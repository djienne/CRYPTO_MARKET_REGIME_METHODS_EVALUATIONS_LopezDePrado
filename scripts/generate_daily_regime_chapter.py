#!/usr/bin/env python3
"""Download Binance daily data and rebuild the regime-change report assets.

The report is intentionally self-contained, so this script does three things:

1. Uses the longest available Binance spot 1d history for each requested pair.
2. Falls back to Binance USD-M futures when spot is unavailable or too short.
3. Computes daily structural-break diagnostics and writes figures/result tables.

The diagnostics are educational implementations aligned with AFML Chapter 17:
log prices, CUSUM-style instability measures, SDFC, SADF/QADF/CADF, and an
SMT-style trend explosiveness score.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crypto-regime")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "binance_daily"
FIG_DIR = ROOT / "regime_chapter_figs"
RESULTS_PATH = ROOT / "regime_chapter_results.json"

ASSETS = ["BTC", "ETH", "ETC", "SOL", "HYPE"]
SYMBOLS = {asset: f"{asset}USDT" for asset in ASSETS}

MARKETS = {
    "spot": {
        "name": "Binance spot",
        "base": "https://api.binance.com/api/v3",
        "exchange_info": "/exchangeInfo",
        "klines": "/klines",
        "limit": 1000,
    },
    "futures": {
        "name": "Binance USD-M futures",
        "base": "https://fapi.binance.com/fapi/v1",
        "exchange_info": "/exchangeInfo",
        "klines": "/klines",
        "limit": 1500,
    },
}

DAY_MS = 24 * 60 * 60 * 1000
TRANSACTION_COST = 0.001
SIGNAL_Q = 0.95
SIGNAL_MIN_HISTORY = 60
MOMENTUM_DAYS = 20
HOLD_DAYS = 20
STRATEGY_NAMES = ["BuyHold", "BDE", "CSW", "SADF", "QADF", "CADF", "SMT", "Consensus"]


@dataclass
class MarketChoice:
    asset: str
    symbol: str
    market: str
    source: str
    reason: str
    spot_status: str
    futures_status: str
    raw_start: str
    raw_end: str
    raw_rows: int
    close_file: str


def utc_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def request_json(url: str, *, params: dict | None = None, max_retries: int = 5):
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code in {418, 429}:
                time.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code >= 400:
                return response.status_code, response.json()
            return response.status_code, response.json()
        except Exception as exc:  # pragma: no cover - diagnostic script
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed for {url}: {last_error}")


def server_time_ms() -> int:
    status, payload = request_json(MARKETS["spot"]["base"] + "/time")
    if status != 200:
        raise RuntimeError(f"could not read Binance server time: {payload}")
    return int(payload["serverTime"])


def symbol_status(market: str, symbol: str) -> str:
    meta = MARKETS[market]
    status, payload = request_json(
        meta["base"] + meta["exchange_info"], params={"symbol": symbol}
    )
    if status != 200:
        return f"unavailable ({payload.get('msg', status)})"
    symbols = payload.get("symbols", [])
    if not symbols:
        return "unavailable (missing from exchangeInfo)"
    item = symbols[0]
    state = item.get("status", "unknown")
    if market == "spot":
        allowed = item.get("isSpotTradingAllowed")
        return f"{state}, spotAllowed={allowed}"
    return state


def fetch_klines(market: str, symbol: str, last_closed_day_start: int) -> pd.DataFrame:
    meta = MARKETS[market]
    status = symbol_status(market, symbol)
    if status.startswith("unavailable"):
        return pd.DataFrame()

    rows: list[list] = []
    start_time = 0
    while True:
        code, payload = request_json(
            meta["base"] + meta["klines"],
            params={
                "symbol": symbol,
                "interval": "1d",
                "startTime": start_time,
                "limit": meta["limit"],
            },
        )
        if code != 200:
            raise RuntimeError(f"{market} kline error for {symbol}: {payload}")
        if not payload:
            break
        rows.extend(payload)
        next_start = int(payload[-1][0]) + DAY_MS
        if len(payload) < meta["limit"] or next_start > last_closed_day_start:
            break
        if next_start <= start_time:
            break
        start_time = next_start
        time.sleep(0.04)

    if not rows:
        return pd.DataFrame()
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.drop_duplicates("open_time").sort_values("open_time")
    frame = frame[frame["open_time"] <= last_closed_day_start].copy()
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.date
    frame = frame[["date", "open_time", "open", "high", "low", "close", "volume", "quote_volume", "trade_count"]]
    frame = frame.dropna(subset=["close"])
    return frame


def choose_market(asset: str, symbol: str, last_closed_day_start: int) -> tuple[MarketChoice, pd.DataFrame]:
    spot_status = symbol_status("spot", symbol)
    futures_status = symbol_status("futures", symbol)
    spot_df = fetch_klines("spot", symbol, last_closed_day_start)
    futures_df = pd.DataFrame()

    reason = "spot is available and has the longest Binance spot history requested"
    market = "spot"
    chosen = spot_df
    if spot_df.empty or len(spot_df) < 365:
        futures_df = fetch_klines("futures", symbol, last_closed_day_start)
        if spot_df.empty:
            reason = "spot is unavailable; used Binance USD-M futures fallback"
            market = "futures"
            chosen = futures_df
        elif len(futures_df) > len(spot_df) + 30:
            reason = "spot history is limited; futures offers a materially longer Binance history"
            market = "futures"
            chosen = futures_df
    if chosen.empty:
        raise RuntimeError(f"no usable Binance daily data for {asset}/{symbol}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / f"{asset}_{symbol}_{market}_1d.csv"
    chosen.to_csv(file_path, index=False)

    choice = MarketChoice(
        asset=asset,
        symbol=symbol,
        market=market,
        source=MARKETS[market]["name"],
        reason=reason,
        spot_status=spot_status,
        futures_status=futures_status,
        raw_start=str(chosen["date"].iloc[0]),
        raw_end=str(chosen["date"].iloc[-1]),
        raw_rows=int(len(chosen)),
        close_file=str(file_path.relative_to(ROOT)),
    )
    return choice, chosen


def min_window(n: int) -> int:
    return max(60, int(math.ceil(0.15 * n)))


def _ols_beta_tstat_intercept_from_sums(
    m: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
) -> np.ndarray:
    m = m.astype(float)
    sxx_c = sxx - sx * sx / m
    sxy_c = sxy - sx * sy / m
    syy_c = syy - sy * sy / m
    beta = np.divide(sxy_c, sxx_c, out=np.full_like(sxy_c, np.nan), where=sxx_c > 1e-14)
    sse = syy_c - beta * sxy_c
    df = m - 2
    sigma2 = np.divide(sse, df, out=np.full_like(sse, np.nan), where=df > 0)
    se = np.sqrt(np.divide(sigma2, sxx_c, out=np.full_like(sigma2, np.nan), where=sxx_c > 1e-14))
    return np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)


def adf_family(logp: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(logp)
    x = logp[:-1]
    dy = np.diff(logp)
    px = np.r_[0.0, np.cumsum(x)]
    py = np.r_[0.0, np.cumsum(dy)]
    pxx = np.r_[0.0, np.cumsum(x * x)]
    pxy = np.r_[0.0, np.cumsum(x * dy)]
    pyy = np.r_[0.0, np.cumsum(dy * dy)]

    sadf = np.full(n, np.nan)
    qadf = np.full(n, np.nan)
    cadf = np.full(n, np.nan)
    for end in range(tau, n):
        starts = np.arange(0, end - tau + 1)
        m = end - starts
        tstats = _ols_beta_tstat_intercept_from_sums(
            m=m,
            sx=px[end] - px[starts],
            sy=py[end] - py[starts],
            sxx=pxx[end] - pxx[starts],
            sxy=pxy[end] - pxy[starts],
            syy=pyy[end] - pyy[starts],
        )
        tstats = tstats[np.isfinite(tstats)]
        if len(tstats) == 0:
            continue
        q = float(np.quantile(tstats, 0.95))
        sadf[end] = float(np.max(tstats))
        qadf[end] = q
        cadf[end] = float(np.mean(tstats[tstats >= q]))
    return sadf, qadf, cadf


def csw_excess(logp: np.ndarray) -> np.ndarray:
    n = len(logp)
    out = np.full(n, np.nan)
    diff2 = np.diff(logp) ** 2
    prefix = np.r_[0.0, np.cumsum(diff2)]
    for end in range(2, n):
        sigma = math.sqrt(prefix[end] / end) if prefix[end] > 0 else np.nan
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        starts = np.arange(0, end)
        gaps = end - starts
        stat = (logp[end] - logp[starts]) / (sigma * np.sqrt(gaps))
        crit = np.sqrt(4.6 + np.log(gaps))
        out[end] = float(np.nanmax(np.abs(stat) - crit))
    return out


def recursive_cusum(logp: np.ndarray) -> np.ndarray:
    n = len(logp)
    resid = np.full(n, np.nan)
    times = np.arange(n, dtype=float)
    for end in range(8, n):
        y = logp[:end]
        x = times[:end]
        x_mean = x.mean()
        y_mean = y.mean()
        sxx = np.sum((x - x_mean) ** 2)
        if sxx <= 0:
            continue
        beta = np.sum((x - x_mean) * (y - y_mean)) / sxx
        alpha = y_mean - beta * x_mean
        fitted = alpha + beta * x
        sse = np.sum((y - fitted) ** 2)
        sigma = math.sqrt(sse / max(1, end - 2))
        if sigma <= 0:
            continue
        x0 = times[end]
        h = 1.0 / end + (x0 - x_mean) ** 2 / sxx
        resid[end] = (logp[end] - (alpha + beta * x0)) / (sigma * math.sqrt(1 + h))
    valid = np.isfinite(resid)
    out = np.full(n, np.nan)
    if valid.sum() < 5:
        return out
    z = (resid[valid] - np.nanmean(resid[valid])) / np.nanstd(resid[valid])
    csum = np.cumsum(z)
    out[np.where(valid)[0]] = csum / np.sqrt(np.arange(1, len(csum) + 1))
    return out


def sdfc(logp: np.ndarray, tau: int) -> tuple[float, int | None]:
    n = len(logp)
    dy = np.diff(logp)
    lag = logp[:-1]
    best_t = -np.inf
    best_break = None
    obs_index = np.arange(1, n)
    for brk in range(tau, n - tau):
        z = lag * (obs_index >= brk)
        denom = float(np.dot(z, z))
        if denom <= 0:
            continue
        beta = float(np.dot(z, dy) / denom)
        err = dy - beta * z
        sigma2 = float(np.dot(err, err) / max(1, len(dy) - 1))
        se = math.sqrt(sigma2 / denom) if sigma2 > 0 else np.nan
        if se > 0:
            tstat = beta / se
            if tstat > best_t:
                best_t = float(tstat)
                best_break = brk
    return best_t, best_break


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(values)]


def smt_exp_and_poly(logp: np.ndarray, price_norm: np.ndarray, tau: int, phi: float = 0.5) -> np.ndarray:
    """SMT-style score from AFML's exponential and quadratic trend forms.

    This evaluates three forms exactly with cumulative sums:
    normalized price quadratic trend, log-price quadratic trend, and log-price
    exponential trend. The local time origin does not affect linear/quadratic slope
    coefficients when the regression includes lower-order terms.
    """

    n = len(logp)
    t = np.arange(1, n + 1, dtype=float)
    p = {
        "1": _prefix(np.ones(n)),
        "t": _prefix(t),
        "t2": _prefix(t**2),
        "t3": _prefix(t**3),
        "t4": _prefix(t**4),
    }

    def linear_tstat(y: np.ndarray, end: int, starts: np.ndarray) -> np.ndarray:
        py = _prefix(y)
        pty = _prefix(t * y)
        pyy = _prefix(y * y)
        stop = end + 1
        m = p["1"][stop] - p["1"][starts]
        sx = p["t"][stop] - p["t"][starts]
        sy = py[stop] - py[starts]
        sxx = p["t2"][stop] - p["t2"][starts]
        sxy = pty[stop] - pty[starts]
        syy = pyy[stop] - pyy[starts]
        return _ols_beta_tstat_intercept_from_sums(m, sx, sy, sxx, sxy, syy)

    def quadratic_tstat(y: np.ndarray, end: int, starts: np.ndarray) -> np.ndarray:
        py = _prefix(y)
        pty = _prefix(t * y)
        pt2y = _prefix((t**2) * y)
        pyy = _prefix(y * y)
        stop = end + 1
        m = p["1"][stop] - p["1"][starts]
        mats = np.empty((len(starts), 3, 3), dtype=float)
        mats[:, 0, 0] = m
        mats[:, 0, 1] = mats[:, 1, 0] = p["t"][stop] - p["t"][starts]
        mats[:, 0, 2] = mats[:, 1, 1] = mats[:, 2, 0] = p["t2"][stop] - p["t2"][starts]
        mats[:, 1, 2] = mats[:, 2, 1] = p["t3"][stop] - p["t3"][starts]
        mats[:, 2, 2] = p["t4"][stop] - p["t4"][starts]
        vecs = np.column_stack(
            [
                py[stop] - py[starts],
                pty[stop] - pty[starts],
                pt2y[stop] - pt2y[starts],
            ]
        )
        out = np.full(len(starts), np.nan)
        good = m > 3
        if not np.any(good):
            return out
        try:
            betas = np.linalg.solve(mats[good], vecs[good][..., None]).squeeze(-1)
            fitted_cross = np.sum(betas * vecs[good], axis=1)
            syy = pyy[stop] - pyy[starts[good]]
            sse = syy - fitted_cross
            sigma2 = sse / (m[good] - 3)
            inv = np.linalg.inv(mats[good])
            var = sigma2 * inv[:, 2, 2]
            out[good] = np.abs(betas[:, 2]) / np.sqrt(var)
        except np.linalg.LinAlgError:
            return out
        return out

    score = np.full(n, np.nan)
    for end in range(tau, n):
        starts = np.arange(0, end - tau + 1)
        m = end - starts + 1
        penalty = m.astype(float) ** phi
        exp_stat = np.abs(linear_tstat(logp, end, starts)) / penalty
        poly_log = quadratic_tstat(logp, end, starts) / penalty
        poly_price = quadratic_tstat(price_norm, end, starts) / penalty
        vals = np.concatenate([exp_stat, poly_log, poly_price])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            score[end] = float(np.max(vals))
    return score


def top_quantile_flag(values: np.ndarray, q: float = 0.95) -> np.ndarray:
    flag = np.zeros(len(values), dtype=bool)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return flag
    threshold = float(np.nanquantile(values[valid], q))
    flag[valid] = values[valid] >= threshold
    return flag


def online_high_signal(values: np.ndarray, *, use_abs: bool = False) -> pd.Series:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    target = series.abs() if use_abs else series
    threshold = target.expanding(min_periods=SIGNAL_MIN_HISTORY).quantile(SIGNAL_Q).shift(1)
    return (target > threshold).fillna(False)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def backtest_position(
    dates: pd.Series,
    close: pd.Series,
    position: pd.Series,
    *,
    transaction_cost: float = TRANSACTION_COST,
    charge_costs: bool = True,
) -> tuple[pd.Series, pd.Series, dict]:
    returns = close.pct_change().fillna(0.0)
    position = position.astype(float).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = transaction_cost * turnover if charge_costs else 0.0
    strategy_returns = position * returns - costs
    equity = (1.0 + strategy_returns).cumprod()
    days = max(1, int((pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days))
    years = days / 365.25
    daily_std = float(strategy_returns.std(ddof=0))
    sharpe = float(np.sqrt(365.0) * strategy_returns.mean() / daily_std) if daily_std > 0 else np.nan
    entries = int(((position > 0) & (position.shift(1).fillna(0) <= 0)).sum())
    metrics = {
        "final_multiple": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else np.nan,
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe,
        "exposure": float(position.mean()),
        "trades": entries,
        "turnover": float(turnover.sum()),
    }
    return strategy_returns, equity, metrics


def compute_strategy_suite(
    asset: str,
    dates: pd.Series,
    close_array: np.ndarray,
    logp: np.ndarray,
    bde: np.ndarray,
    csw: np.ndarray,
    sadf: np.ndarray,
    qadf: np.ndarray,
    cadf: np.ndarray,
    smt: np.ndarray,
) -> tuple[dict, str]:
    close = pd.Series(close_array, dtype=float)
    momentum = pd.Series(logp).diff(MOMENTUM_DAYS).gt(0).fillna(False)
    raw_events = {
        "BDE": online_high_signal(bde, use_abs=True) & momentum,
        "CSW": pd.Series(csw).gt(0).fillna(False) & momentum,
        "SADF": online_high_signal(sadf) & momentum,
        "QADF": online_high_signal(qadf) & momentum,
        "CADF": online_high_signal(cadf) & momentum,
        "SMT": online_high_signal(smt) & momentum,
    }
    consensus_event = (
        raw_events["BDE"].astype(int)
        + raw_events["CSW"].astype(int)
        + raw_events["SADF"].astype(int)
        + raw_events["SMT"].astype(int)
    ) >= 2
    raw_events["Consensus"] = consensus_event

    strategy_frame = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "close": close,
            "return": close.pct_change().fillna(0.0),
            "momentum_20d_positive": momentum,
        }
    )
    metrics: dict[str, dict] = {}

    buyhold_position = pd.Series(1.0, index=close.index)
    buyhold_returns, buyhold_equity, buyhold_metrics = backtest_position(
        dates, close, buyhold_position, charge_costs=False
    )
    strategy_frame["pos_BuyHold"] = buyhold_position
    strategy_frame["ret_BuyHold"] = buyhold_returns
    strategy_frame["equity_BuyHold"] = buyhold_equity
    metrics["BuyHold"] = buyhold_metrics

    for name, event in raw_events.items():
        held = event.rolling(HOLD_DAYS, min_periods=1).max().astype(bool)
        position = held.shift(1).fillna(False).astype(float)
        strategy_returns, equity, stats = backtest_position(dates, close, position)
        strategy_frame[f"event_{name}"] = event.astype(int)
        strategy_frame[f"pos_{name}"] = position
        strategy_frame[f"ret_{name}"] = strategy_returns
        strategy_frame[f"equity_{name}"] = equity
        metrics[name] = stats

    best_signal = max(
        [name for name in STRATEGY_NAMES if name != "BuyHold"],
        key=lambda name: metrics[name]["final_multiple"],
    )
    best_including_buyhold = max(
        STRATEGY_NAMES,
        key=lambda name: metrics[name]["final_multiple"],
    )
    out_csv = DATA_DIR / f"{asset}_strategy_daily.csv"
    strategy_frame.to_csv(out_csv, index=False)
    summary = {
        "strategy_file": str(out_csv.relative_to(ROOT)),
        "parameters": {
            "transaction_cost": TRANSACTION_COST,
            "signal_quantile": SIGNAL_Q,
            "signal_min_history": SIGNAL_MIN_HISTORY,
            "momentum_days": MOMENTUM_DAYS,
            "hold_days": HOLD_DAYS,
        },
        "metrics": metrics,
        "best_signal_strategy": best_signal,
        "best_signal_final_multiple": metrics[best_signal]["final_multiple"],
        "best_including_buyhold": best_including_buyhold,
        "best_including_buyhold_final_multiple": metrics[best_including_buyhold]["final_multiple"],
    }
    return summary, str(out_csv.relative_to(ROOT))


def runs_from_flag(dates: pd.Series, flag: np.ndarray) -> list[dict]:
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return []
    runs: list[dict] = []
    start = prev = idx[0]
    def fmt(pos: int) -> str:
        return pd.Timestamp(dates.iloc[pos]).date().isoformat()

    for item in idx[1:]:
        if item == prev + 1:
            prev = item
            continue
        runs.append(
            {
                "start": fmt(start),
                "end": fmt(prev),
                "days": int((pd.Timestamp(dates.iloc[prev]) - pd.Timestamp(dates.iloc[start])).days + 1),
            }
        )
        start = prev = item
    runs.append(
        {
            "start": fmt(start),
            "end": fmt(prev),
            "days": int((pd.Timestamp(dates.iloc[prev]) - pd.Timestamp(dates.iloc[start])).days + 1),
        }
    )
    return runs


def compute_asset(asset: str, frame: pd.DataFrame, choice: MarketChoice) -> dict:
    dates = pd.Series(pd.to_datetime(frame["date"]))
    close = frame["close"].astype(float).to_numpy()
    logp = np.log(close)
    price_norm = close / close[0]
    tau = min_window(len(close))

    bde = recursive_cusum(logp)
    csw = csw_excess(logp)
    sadf, qadf, cadf = adf_family(logp, tau)
    smt = smt_exp_and_poly(logp, price_norm, tau)
    sdfc_t, sdfc_idx = sdfc(logp, tau)

    bde_flag = top_quantile_flag(np.abs(bde))
    csw_flag = np.nan_to_num(csw, nan=-np.inf) > 0
    sadf_flag = top_quantile_flag(sadf)
    smt_flag = top_quantile_flag(smt)
    family_count = (
        bde_flag.astype(int)
        + csw_flag.astype(int)
        + sadf_flag.astype(int)
        + smt_flag.astype(int)
    )
    consensus = family_count >= 2
    runs = runs_from_flag(dates, consensus)
    strategies, strategy_file = compute_strategy_suite(
        asset, dates, close, logp, bde, csw, sadf, qadf, cadf, smt
    )

    series = {
        "date": dates.dt.strftime("%Y-%m-%d"),
        "close": close,
        "logp": logp,
        "bde": bde,
        "csw_excess": csw,
        "sadf": sadf,
        "qadf": qadf,
        "cadf": cadf,
        "smt": smt,
        "family_count": family_count,
        "consensus": consensus,
    }
    out_csv = DATA_DIR / f"{asset}_diagnostics_daily.csv"
    pd.DataFrame(series).to_csv(out_csv, index=False)

    def max_date(values: np.ndarray) -> str:
        valid = np.isfinite(values)
        if valid.sum() == 0:
            return "n/a"
        pos = int(np.nanargmax(values))
        return str(dates.iloc[pos].date())

    summary = {
        **asdict(choice),
        "clock": "daily close",
        "test_rows": int(len(close)),
        "tau": int(tau),
        "diagnostics_file": str(out_csv.relative_to(ROOT)),
        "sdfc_t": float(sdfc_t),
        "sdfc_candidate": str(dates.iloc[sdfc_idx].date()) if sdfc_idx is not None and sdfc_t > 0 else "none (negative t)",
        "max_sadf": max_date(sadf),
        "max_csw": max_date(csw),
        "max_smt": max_date(smt),
        "runs": runs,
        "run_count": int(len(runs)),
        "first_run": runs[0]["start"] if runs else "none",
        "strategy_file": strategy_file,
        "strategies": strategies,
    }
    return summary


def shade_runs(ax, runs: Iterable[dict]):
    for run in runs:
        ax.axvspan(pd.Timestamp(run["start"]), pd.Timestamp(run["end"]), color="#f4a340", alpha=0.24, lw=0)


def plot_normalized(results: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for asset, result in results.items():
        diag = pd.read_csv(ROOT / result["diagnostics_file"], parse_dates=["date"])
        norm = diag["close"] / diag["close"].iloc[0]
        ax.plot(diag["date"], norm, lw=1.5, label=asset)
    ax.set_yscale("log")
    ax.set_title("Daily Binance close prices, normalized to 1")
    ax.set_ylabel("Normalized close (log scale)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=5, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_normalized_prices.pdf")
    plt.close(fig)


def plot_asset(asset: str, result: dict):
    diag = pd.read_csv(ROOT / result["diagnostics_file"], parse_dates=["date"])
    fig, axes = plt.subplots(5, 1, figsize=(11, 12.5), sharex=True)
    runs = result["runs"]

    axes[0].plot(diag["date"], diag["close"], color="#1f4e79", lw=1.25)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Close")
    axes[0].set_title(f"{asset}: daily {result['source']} diagnostics")
    shade_runs(axes[0], runs)
    if result["sdfc_candidate"] and not result["sdfc_candidate"].startswith("none"):
        axes[0].axvline(pd.Timestamp(result["sdfc_candidate"]), color="#7b3294", ls="--", lw=1)

    axes[1].plot(diag["date"], diag["bde"], color="#276419", lw=1)
    axes[1].axhline(0, color="black", lw=0.6, alpha=0.6)
    axes[1].set_ylabel("BDE CUSUM")
    shade_runs(axes[1], runs)

    axes[2].plot(diag["date"], diag["csw_excess"], color="#b35806", lw=1)
    axes[2].axhline(0, color="black", lw=0.8, alpha=0.7)
    axes[2].set_ylabel("CSW excess")
    shade_runs(axes[2], runs)

    axes[3].plot(diag["date"], diag["sadf"], label="SADF", color="#542788", lw=1)
    axes[3].plot(diag["date"], diag["qadf"], label="QADF 95%", color="#2d708e", lw=1)
    axes[3].plot(diag["date"], diag["cadf"], label="CADF tail mean", color="#5aae61", lw=1)
    axes[3].set_ylabel("ADF t-stat")
    axes[3].legend(fontsize=8, ncol=3, loc="upper left")
    shade_runs(axes[3], runs)

    axes[4].plot(diag["date"], diag["smt"], color="#8c510a", lw=1)
    axes[4].set_ylabel("SMT score")
    axes[4].set_xlabel("Date")
    shade_runs(axes[4], runs)

    for ax in axes:
        ax.grid(True, alpha=0.22)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig_asset_{asset}.pdf")
    plt.close(fig)


def plot_heatmap(results: dict[str, dict]):
    all_months = []
    rows = []
    for asset, result in results.items():
        diag = pd.read_csv(ROOT / result["diagnostics_file"], parse_dates=["date"])
        months = diag["date"].dt.to_period("M").astype(str)
        counts = pd.Series(diag["consensus"].astype(int).to_numpy(), index=months).groupby(level=0).sum()
        rows.append((asset, counts))
        all_months.extend(counts.index.tolist())
    month_index = sorted(set(all_months))
    matrix = np.zeros((len(rows), len(month_index)))
    for i, (_, counts) in enumerate(rows):
        matrix[i, :] = [counts.get(month, 0) for month in month_index]

    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(np.arange(len(rows)), [asset for asset, _ in rows])
    step = max(1, len(month_index) // 12)
    xticks = np.arange(0, len(month_index), step)
    ax.set_xticks(xticks, [month_index[i] for i in xticks], rotation=45, ha="right")
    ax.set_title("Consensus alert observations by month")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Daily alert observations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_alert_heatmap.pdf")
    plt.close(fig)


def plot_strategy_asset(asset: str, result: dict):
    strat = pd.read_csv(ROOT / result["strategy_file"], parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = {
        "BuyHold": "#111111",
        "BDE": "#276419",
        "CSW": "#b35806",
        "SADF": "#542788",
        "QADF": "#2d708e",
        "CADF": "#5aae61",
        "SMT": "#8c510a",
        "Consensus": "#c51b7d",
    }
    for name in STRATEGY_NAMES:
        style = "--" if name == "BuyHold" else "-"
        width = 1.8 if name in {"BuyHold", result["strategies"]["best_signal_strategy"]} else 1.05
        alpha = 0.95 if name in {"BuyHold", result["strategies"]["best_signal_strategy"], "Consensus"} else 0.72
        label = "Buy-and-hold" if name == "BuyHold" else name
        ax.plot(
            strat["date"],
            strat[f"equity_{name}"],
            label=label,
            color=colors[name],
            ls=style,
            lw=width,
            alpha=alpha,
        )
    ax.set_yscale("log")
    ax.set_title(f"{asset}: long/flat PnL curves from daily structure-break signals")
    ax.set_ylabel("Equity multiple, start = 1 (log scale)")
    ax.grid(True, which="both", alpha=0.24)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig_strategy_pnl_{asset}.pdf")
    plt.close(fig)


def plot_strategy_heatmap(results: dict[str, dict]):
    matrix = np.zeros((len(results), len(STRATEGY_NAMES)))
    assets = list(results.keys())
    for i, asset in enumerate(assets):
        metrics = results[asset]["strategies"]["metrics"]
        matrix[i, :] = [metrics[name]["final_multiple"] for name in STRATEGY_NAMES]
    fig, ax = plt.subplots(figsize=(10.8, 4.1))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(assets)), assets)
    ax.set_xticks(np.arange(len(STRATEGY_NAMES)), ["Buy&hold", "BDE", "CSW", "SADF", "QADF", "CADF", "SMT", "Consensus"], rotation=35, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] < np.nanmax(matrix) * 0.55 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}x", ha="center", va="center", fontsize=8, color=color)
    ax.set_title("Final equity multiple by asset and strategy")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Final equity multiple")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_strategy_final_multiple_heatmap.pdf")
    plt.close(fig)


def pct(value: float, digits: int = 0) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.{digits}f}\\%"


def mult(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}x"


def num(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def strict_json_value(value):
    if isinstance(value, dict):
        return {key: strict_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [strict_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_latex_snippets(results: dict[str, dict]):
    summary_lines = []
    for asset, r in results.items():
        sdfc_t = f"{r['sdfc_t']:.2f}"
        summary_lines.append(
            f"{asset} & {r['source']} & {r['raw_start']} & {r['raw_end']} & "
            f"{r['raw_rows']} & {r['clock']} & {r['test_rows']} & {r['tau']} & "
            f"{r['sdfc_candidate']} & {sdfc_t} & {r['max_sadf']} & {r['max_csw']} & "
            f"{r['max_smt']} & {r['run_count']} \\\\"
        )

    run_lines = []
    for asset, r in results.items():
        first = True
        for run in r["runs"][:8]:
            label = asset if first else ""
            first = False
            run_lines.append(f"{label} & {run['start']} & {run['end']} & {run['days']} \\\\")

    notes = []
    for asset, r in results.items():
        notes.append(
            f"{asset}: {r['source']} {r['symbol']} {r['raw_start']} to {r['raw_end']} "
            f"({r['raw_rows']} rows); {r['reason']}. Spot status: {r['spot_status']}. "
            f"Futures status: {r['futures_status']}."
        )

    def strategy_label(name: str) -> str:
        return "Buy-and-hold" if name == "BuyHold" else name

    strategy_best_lines = []
    strategy_detail_lines = []
    for asset, r in results.items():
        strat = r["strategies"]
        metrics = strat["metrics"]
        best_signal = strat["best_signal_strategy"]
        best_any = strat["best_including_buyhold"]
        best_stats = metrics[best_signal]
        strategy_best_lines.append(
            f"{asset} & {strategy_label(best_signal)} & {mult(best_stats['final_multiple'])} & "
            f"{mult(metrics['BuyHold']['final_multiple'])} & {strategy_label(best_any)} & "
            f"{pct(best_stats['max_drawdown'])} & {pct(best_stats['exposure'])} & {best_stats['trades']} \\\\"
        )
        for name in STRATEGY_NAMES:
            stats = metrics[name]
            strategy_detail_lines.append(
                f"{asset} & {strategy_label(name)} & {mult(stats['final_multiple'])} & "
                f"{pct(stats['total_return'])} & {pct(stats['cagr'])} & "
                f"{pct(stats['max_drawdown'])} & {num(stats['sharpe'])} & "
                f"{pct(stats['exposure'])} & {stats['trades']} \\\\"
            )

    snippet = {
        "summary_rows": "\n".join(summary_lines),
        "run_rows": "\n".join(run_lines),
        "strategy_best_rows": "\n".join(strategy_best_lines),
        "strategy_detail_rows": "\n".join(strategy_detail_lines),
        "source_notes": notes,
    }
    (ROOT / "regime_chapter_latex_snippets.json").write_text(json.dumps(snippet, indent=2))


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    now_ms = server_time_ms()
    now_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    today_start = datetime(now_dt.year, now_dt.month, now_dt.day, tzinfo=timezone.utc)
    last_closed_day_start = int(today_start.timestamp() * 1000) - DAY_MS

    choices: dict[str, MarketChoice] = {}
    frames: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        choice, frame = choose_market(asset, SYMBOLS[asset], last_closed_day_start)
        choices[asset] = choice
        frames[asset] = frame
        print(
            f"{asset}: {choice.source} {choice.symbol}, "
            f"{choice.raw_start} to {choice.raw_end}, {choice.raw_rows} rows"
        )

    results: dict[str, dict] = {}
    for asset in ASSETS:
        print(f"Computing diagnostics for {asset}...")
        results[asset] = compute_asset(asset, frames[asset], choices[asset])

    plot_normalized(results)
    for asset in ASSETS:
        plot_asset(asset, results[asset])
        plot_strategy_asset(asset, results[asset])
    plot_heatmap(results)
    plot_strategy_heatmap(results)
    write_latex_snippets(results)

    payload = {
        "generated_at_utc": now_dt.isoformat(),
        "last_closed_day": utc_date(last_closed_day_start),
        "assets": results,
    }
    RESULTS_PATH.write_text(json.dumps(strict_json_value(payload), indent=2, allow_nan=False))
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote figures to {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
