#!/usr/bin/env python3
"""Download Binance daily data and rebuild the crypto regime-change report.

The report is intentionally self-contained. This script downloads or reloads
daily Binance OHLCV data, computes structural-break diagnostics, runs simple
online long/flat strategy conversions, renders figures, and generates the TeX
report from a template so numerical tables cannot go stale.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crypto-regime")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "binance_daily"
CALIBRATION_DIR = ROOT / "data" / "calibration"
FIG_DIR = ROOT / "regime_chapter_figs"
RESULTS_PATH = ROOT / "regime_chapter_results.json"
SNIPPET_PATH = ROOT / "regime_chapter_latex_snippets.json"
REPORT_TEMPLATE_PATH = ROOT / "regime_detection_crypto_report_template.tex"
REPORT_PATH = ROOT / "regime_detection_crypto_report.tex"

ASSETS = ("BTC", "ETH", "ETC", "SOL", "HYPE")
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
DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_SIGNAL_Q = 0.95
DEFAULT_SIGNAL_MIN_HISTORY = 60
DEFAULT_MOMENTUM_DAYS = 20
DEFAULT_HOLD_DAYS = 20
DEFAULT_TAU_FRACTION = 0.15
DEFAULT_CALIBRATION_SIMS = 2000
DEFAULT_CALIBRATION_SEED = 123
DEFAULT_BOOTSTRAP_SIMS = 500
DEFAULT_BOOTSTRAP_SEED = 777
DEFAULT_WALK_FORWARD_TRAIN_MIN = 730
DEFAULT_WALK_FORWARD_TEST_SIZE = 180
DEFAULT_HMM_STATES = 3
DEFAULT_HMM_REFIT_DAYS = 126
DEFAULT_HMM_MIN_HISTORY = 365
DEFAULT_HMM_EM_ITER = 100
TAU_SENSITIVITY_FRACTIONS = (0.10, 0.15, 0.20)
STRATEGY_NAMES = ("BuyHold", "BDE", "CSW", "SADF", "QADF", "CADF", "SMT", "HMM", "Consensus")
STRUCTURE_STRATEGIES = tuple(name for name in STRATEGY_NAMES if name != "BuyHold")
CALIBRATION_VERSION = "adf_zero_lag_v2"

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "crypto-regime-report/1.0"})


@dataclass(frozen=True)
class RunConfig:
    assets: tuple[str, ...] = ASSETS
    transaction_cost: float = DEFAULT_TRANSACTION_COST
    signal_q: float = DEFAULT_SIGNAL_Q
    signal_min_history: int = DEFAULT_SIGNAL_MIN_HISTORY
    momentum_days: int = DEFAULT_MOMENTUM_DAYS
    hold_days: int = DEFAULT_HOLD_DAYS
    tau_fraction: float = DEFAULT_TAU_FRACTION
    calibration_sims: int = DEFAULT_CALIBRATION_SIMS
    calibration_seed: int = DEFAULT_CALIBRATION_SEED
    bootstrap_sims: int = DEFAULT_BOOTSTRAP_SIMS
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    walk_forward_train_min: int = DEFAULT_WALK_FORWARD_TRAIN_MIN
    walk_forward_test_size: int = DEFAULT_WALK_FORWARD_TEST_SIZE
    hmm_states: int = DEFAULT_HMM_STATES
    hmm_refit_days: int = DEFAULT_HMM_REFIT_DAYS
    hmm_min_history: int = DEFAULT_HMM_MIN_HISTORY
    use_cache: bool = False
    skip_download: bool = False
    workers: int = max(1, min(8, os.cpu_count() or 1))


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
            response = HTTP.get(url, params=params, timeout=30)
            try:
                payload = response.json()
            except ValueError:
                payload = {"text": response.text[:500]}
            if response.status_code in {418, 429}:
                last_error = (response.status_code, payload)
                time.sleep(min(30.0, 1.5 * (2**attempt)))
                continue
            if response.status_code >= 400:
                return response.status_code, payload
            return response.status_code, payload
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(30.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"request failed for {url}: {last_error}")


def server_time_ms() -> int:
    status, payload = request_json(MARKETS["spot"]["base"] + "/time")
    if status != 200:
        raise RuntimeError(f"could not read Binance server time: {payload}")
    return int(payload["serverTime"])


@lru_cache(maxsize=128)
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


def raw_data_path(asset: str, symbol: str, market: str) -> Path:
    return DATA_DIR / f"{asset}_{symbol}_{market}_1d.csv"


def validate_ohlcv(frame: pd.DataFrame, symbol: str) -> None:
    if frame.empty:
        raise ValueError(f"{symbol}: no OHLCV rows")
    required = ["date", "open", "high", "low", "close"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")
    numeric = frame[["open", "high", "low", "close"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{symbol}: non-finite OHLC values")
    if (numeric <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive OHLC values")
    bad = frame[
        (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame["low"])
    ]
    if not bad.empty:
        raise ValueError(f"{symbol}: OHLC sanity check failed on {len(bad)} rows")
    dates = pd.to_datetime(frame["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError(f"{symbol}: dates are not sorted")
    if dates.duplicated().any():
        raise ValueError(f"{symbol}: duplicate dates found")


def load_cached_frame(asset: str, symbol: str, market: str) -> pd.DataFrame:
    path = raw_data_path(asset, symbol, market)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    validate_ohlcv(frame, f"{asset}/{symbol}/{market}")
    return frame


def fetch_klines(
    market: str,
    symbol: str,
    last_closed_day_start: int,
    *,
    status: str | None = None,
) -> pd.DataFrame:
    meta = MARKETS[market]
    status = status if status is not None else symbol_status(market, symbol)
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
    frame = frame[
        [
            "date",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
        ]
    ]
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    validate_ohlcv(frame, symbol)
    return frame


def choose_market(
    asset: str,
    symbol: str,
    last_closed_day_start: int,
    config: RunConfig,
) -> tuple[MarketChoice, pd.DataFrame]:
    spot_status = "not checked (--skip-download)" if config.skip_download else symbol_status("spot", symbol)
    futures_status = "not checked (--skip-download)" if config.skip_download else symbol_status("futures", symbol)

    spot_df = load_cached_frame(asset, symbol, "spot") if config.use_cache or config.skip_download else pd.DataFrame()
    futures_df = load_cached_frame(asset, symbol, "futures") if config.use_cache or config.skip_download else pd.DataFrame()

    if not config.skip_download:
        if spot_df.empty:
            spot_df = fetch_klines("spot", symbol, last_closed_day_start, status=spot_status)
        if spot_df.empty or len(spot_df) < 365:
            futures_df = fetch_klines("futures", symbol, last_closed_day_start, status=futures_status)

    reason = "spot is available and has the longest Binance spot history requested"
    market = "spot"
    chosen = spot_df
    if spot_df.empty or len(spot_df) < 365:
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
    file_path = raw_data_path(asset, symbol, market)
    if not ((config.use_cache or config.skip_download) and file_path.exists()):
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


def min_window(n: int, tau_fraction: float = DEFAULT_TAU_FRACTION) -> int:
    return max(60, int(math.ceil(tau_fraction * n)))


def validate_close_series(asset: str, dates: pd.Series, close: np.ndarray) -> None:
    if len(close) == 0:
        raise ValueError(f"{asset}: empty close series")
    if not np.isfinite(close).all():
        raise ValueError(f"{asset}: close contains non-finite values")
    if np.any(close <= 0):
        raise ValueError(f"{asset}: close contains non-positive values")
    if not pd.to_datetime(dates).is_monotonic_increasing:
        raise ValueError(f"{asset}: dates are not sorted")


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


def finite_nanmax(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.nanmax(values[valid])) if valid.any() else np.nan


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
    """BDE-inspired recursive forecast-error CUSUM with online normalization."""

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

    s = pd.Series(resid)
    mu = s.expanding(min_periods=30).mean().shift(1)
    sd = s.expanding(min_periods=30).std(ddof=0).shift(1).replace(0, np.nan)
    z = (s - mu) / sd
    count = z.notna().cumsum()
    csum = z.cumsum()
    out = csum / np.sqrt(count.replace(0, np.nan))
    return out.to_numpy(dtype=float)


def sdfc(logp: np.ndarray, tau: int) -> tuple[float, int | None]:
    n = len(logp)
    if n < 2 * tau + 5:
        return np.nan, None
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
    if best_break is None:
        return np.nan, None
    return best_t, best_break


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(values)]


def smt_trend_score(logp: np.ndarray, price_norm: np.ndarray, tau: int, phi: float = 0.5) -> np.ndarray:
    """SMT-style score from the AFML 17.4.3 trend battery.

    Covers all four book specifications: SM-Poly1 (quadratic trend on raw
    normalized price), SM-Poly2 (quadratic trend on log price), SM-Exp
    (linear trend on log price), and SM-Power (log price on log time). Each
    absolute trend t-statistic is penalized by window length to the power
    ``phi`` before taking the supremum over backward windows.
    """

    n = len(logp)
    t = np.arange(1, n + 1, dtype=float)
    logt = np.log(t)
    ones = _prefix(np.ones(n))
    pt = _prefix(t)
    pt2 = _prefix(t**2)
    pt3 = _prefix(t**3)
    pt4 = _prefix(t**4)
    pu = _prefix(logt)
    pu2 = _prefix(logt**2)

    py_log = _prefix(logp)
    pyy_log = _prefix(logp * logp)
    pty_log = _prefix(t * logp)
    pt2y_log = _prefix((t**2) * logp)
    puy_log = _prefix(logt * logp)
    py_pr = _prefix(price_norm)
    pyy_pr = _prefix(price_norm * price_norm)
    pty_pr = _prefix(t * price_norm)
    pt2y_pr = _prefix((t**2) * price_norm)

    def linear_tstat(
        end: int,
        starts: np.ndarray,
        px: np.ndarray,
        pxx: np.ndarray,
        pxy: np.ndarray,
        py: np.ndarray,
        pyy: np.ndarray,
    ) -> np.ndarray:
        stop = end + 1
        m = ones[stop] - ones[starts]
        return _ols_beta_tstat_intercept_from_sums(
            m,
            px[stop] - px[starts],
            py[stop] - py[starts],
            pxx[stop] - pxx[starts],
            pxy[stop] - pxy[starts],
            pyy[stop] - pyy[starts],
        )

    def quadratic_tstat(
        end: int,
        starts: np.ndarray,
        py: np.ndarray,
        pty: np.ndarray,
        pt2y: np.ndarray,
        pyy: np.ndarray,
    ) -> np.ndarray:
        stop = end + 1
        m = ones[stop] - ones[starts]
        mats = np.empty((len(starts), 3, 3), dtype=float)
        mats[:, 0, 0] = m
        mats[:, 0, 1] = mats[:, 1, 0] = pt[stop] - pt[starts]
        mats[:, 0, 2] = mats[:, 1, 1] = mats[:, 2, 0] = pt2[stop] - pt2[starts]
        mats[:, 1, 2] = mats[:, 2, 1] = pt3[stop] - pt3[starts]
        mats[:, 2, 2] = pt4[stop] - pt4[starts]
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
        exp_stat = np.abs(linear_tstat(end, starts, pt, pt2, pty_log, py_log, pyy_log)) / penalty
        power_stat = np.abs(linear_tstat(end, starts, pu, pu2, puy_log, py_log, pyy_log)) / penalty
        poly_log = quadratic_tstat(end, starts, py_log, pty_log, pt2y_log, pyy_log) / penalty
        poly_price = quadratic_tstat(end, starts, py_pr, pty_pr, pt2y_pr, pyy_pr) / penalty
        vals = np.concatenate([exp_stat, power_stat, poly_log, poly_price])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            score[end] = float(np.max(vals))
    return score


def _gaussian_diag_loglik(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Log density of each row of ``x`` under each diagonal-Gaussian state."""

    diff = x[:, None, :] - means[None, :, :]
    quad = np.sum(diff * diff / variances[None, :, :], axis=2)
    norm = np.sum(np.log(2.0 * np.pi * variances), axis=1)
    return -0.5 * (quad + norm[None, :])


def _hmm_forward_filter(
    x: np.ndarray,
    start: np.ndarray,
    trans: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Scaled forward pass. Row t of the output uses observations 0..t only.

    Returns the filtered state probabilities and the total log-likelihood.
    """

    log_b = _gaussian_diag_loglik(x, means, variances)
    row_max = log_b.max(axis=1)
    b = np.exp(log_b - row_max[:, None])
    n, k = b.shape
    alpha = np.empty((n, k))
    scale = np.empty(n)
    a = start * b[0]
    scale[0] = max(a.sum(), 1e-300)
    alpha[0] = a / scale[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ trans) * b[t]
        scale[t] = max(a.sum(), 1e-300)
        alpha[t] = a / scale[t]
    loglik = float(np.sum(np.log(scale)) + np.sum(row_max))
    return alpha, loglik


def fit_gaussian_hmm(
    x: np.ndarray,
    n_states: int,
    *,
    n_iter: int = DEFAULT_HMM_EM_ITER,
    tol: float = 1e-6,
) -> dict | None:
    """Deterministic EM fit of a diagonal-covariance Gaussian HMM.

    Initialization is deterministic (states seeded from quantile slices of the
    first feature), so repeated fits on identical data give identical models.
    Returns ``None`` when the sample is too short or the fit degenerates.
    """

    x = np.asarray(x, dtype=float)
    n, d = x.shape
    if n < max(8 * n_states, 2 * d + 2) or not np.isfinite(x).all():
        return None
    var_floor = np.maximum(np.var(x, axis=0), 1e-12) * 1e-4 + 1e-12

    order = np.argsort(x[:, 0], kind="mergesort")
    means = np.empty((n_states, d))
    variances = np.empty((n_states, d))
    for state, chunk in enumerate(np.array_split(order, n_states)):
        means[state] = x[chunk].mean(axis=0)
        variances[state] = np.maximum(x[chunk].var(axis=0), var_floor)
    start = np.full(n_states, 1.0 / n_states)
    if n_states > 1:
        trans = np.full((n_states, n_states), 0.1 / (n_states - 1))
        np.fill_diagonal(trans, 0.9)
    else:
        trans = np.ones((1, 1))

    prev_loglik = -np.inf
    for _ in range(n_iter):
        log_b = _gaussian_diag_loglik(x, means, variances)
        row_max = log_b.max(axis=1)
        b = np.exp(log_b - row_max[:, None])

        alpha = np.empty((n, n_states))
        scale = np.empty(n)
        a = start * b[0]
        scale[0] = max(a.sum(), 1e-300)
        alpha[0] = a / scale[0]
        for t in range(1, n):
            a = (alpha[t - 1] @ trans) * b[t]
            scale[t] = max(a.sum(), 1e-300)
            alpha[t] = a / scale[t]

        beta = np.empty((n, n_states))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (trans @ (b[t + 1] * beta[t + 1])) / scale[t + 1]

        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        xi_sum = trans * (alpha[:-1].T @ ((b[1:] * beta[1:]) / scale[1:, None]))
        loglik = float(np.sum(np.log(scale)) + np.sum(row_max))

        start = np.maximum(gamma[0], 1e-12)
        start /= start.sum()
        trans = np.maximum(xi_sum, 1e-12)
        trans /= trans.sum(axis=1, keepdims=True)
        weights = np.maximum(gamma.sum(axis=0), 1e-8)
        means = (gamma.T @ x) / weights[:, None]
        variances = np.maximum((gamma.T @ (x * x)) / weights[:, None] - means**2, var_floor)

        if not np.isfinite(loglik):
            return None
        if abs(loglik - prev_loglik) < tol * max(1.0, abs(prev_loglik)):
            prev_loglik = loglik
            break
        prev_loglik = loglik

    # Final E-step responsibilities under the converged parameters.
    log_b = _gaussian_diag_loglik(x, means, variances)
    row_max = log_b.max(axis=1)
    b = np.exp(log_b - row_max[:, None])
    alpha = np.empty((n, n_states))
    scale = np.empty(n)
    a = start * b[0]
    scale[0] = max(a.sum(), 1e-300)
    alpha[0] = a / scale[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ trans) * b[t]
        scale[t] = max(a.sum(), 1e-300)
        alpha[t] = a / scale[t]
    beta = np.empty((n, n_states))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (trans @ (b[t + 1] * beta[t + 1])) / scale[t + 1]
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    return {
        "start": start,
        "trans": trans,
        "means": means,
        "variances": variances,
        "gamma": gamma,
        "loglik": prev_loglik,
    }


def hmm_regime_signal(
    logp: np.ndarray,
    *,
    n_states: int = DEFAULT_HMM_STATES,
    refit_days: int = DEFAULT_HMM_REFIT_DAYS,
    min_history: int = DEFAULT_HMM_MIN_HISTORY,
    momentum_days: int = DEFAULT_MOMENTUM_DAYS,
    n_iter: int = DEFAULT_HMM_EM_ITER,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward Gaussian HMM regime filter with no forward-looking bias.

    Design guarantees against lookahead:

    - Features at close ``t`` (daily log return and ``momentum_days`` log
      momentum) use prices through ``t`` only.
    - Model parameters used for close ``t`` are fitted on features strictly
      before the most recent refit date, which is itself ``<= t``.
    - The regime at close ``t`` is the argmax of *filtered* probabilities
      from a forward pass over features ``0..t`` (never smoothed with future
      observations).
    - The regime-to-signal map (which state is bullish) is computed from the
      training window only, using EM responsibilities against same-window
      returns.
    - Refit dates are fixed absolute indices, so the value at ``t`` does not
      change when future data is appended.

    Returns a boolean bull-signal array and an integer state array (``-1``
    where no model is available), both aligned to ``logp``.
    """

    n = len(logp)
    signal = np.zeros(n, dtype=bool)
    states = np.full(n, -1, dtype=int)
    if n <= momentum_days + 1:
        return signal, states

    t_idx = np.arange(momentum_days, n)
    features = np.column_stack(
        [
            logp[t_idx] - logp[t_idx - 1],
            logp[t_idx] - logp[t_idx - momentum_days],
        ]
    )
    n_feat = len(features)
    if n_feat <= min_history:
        return signal, states

    model: dict | None = None
    bull_state: int | None = None
    fit_at = min_history
    while fit_at < n_feat:
        block_end = min(fit_at + refit_days, n_feat)
        fitted = fit_gaussian_hmm(features[:fit_at], n_states, n_iter=n_iter)
        if fitted is not None:
            model = fitted
            gamma = fitted["gamma"]
            weights = np.maximum(gamma.sum(axis=0), 1e-8)
            state_mean_return = (gamma * features[:fit_at, [0]]).sum(axis=0) / weights
            best = int(np.argmax(state_mean_return))
            bull_state = best if state_mean_return[best] > 0 else None
        if model is not None:
            filtered, _ = _hmm_forward_filter(
                features[:block_end],
                model["start"],
                model["trans"],
                model["means"],
                model["variances"],
            )
            block_states = np.argmax(filtered[fit_at:block_end], axis=1)
            states[t_idx[fit_at:block_end]] = block_states
            if bull_state is not None:
                signal[t_idx[fit_at:block_end]] = block_states == bull_state
        fit_at = block_end
    return signal, states


def top_quantile_flag(values: np.ndarray, q: float = 0.95) -> np.ndarray:
    flag = np.zeros(len(values), dtype=bool)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return flag
    threshold = float(np.nanquantile(values[valid], q))
    flag[valid] = values[valid] >= threshold
    return flag


def online_high_signal(
    values: np.ndarray,
    *,
    use_abs: bool = False,
    q: float = DEFAULT_SIGNAL_Q,
    min_history: int = DEFAULT_SIGNAL_MIN_HISTORY,
) -> pd.Series:
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    target = series.abs() if use_abs else series
    threshold = target.expanding(min_periods=min_history).quantile(q).shift(1)
    return (target > threshold).fillna(False)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def metrics_from_returns(
    strategy_returns: pd.Series,
    *,
    days: int | None = None,
    position: pd.Series | None = None,
) -> tuple[pd.Series, dict]:
    strategy_returns = strategy_returns.astype(float).fillna(0.0)
    equity = (1.0 + strategy_returns).cumprod()
    days = int(days if days is not None else max(1, len(strategy_returns) - 1))
    years = max(days / 365.25, 1 / 365.25)
    daily_std = float(strategy_returns.std(ddof=0))
    sharpe = float(np.sqrt(365.0) * strategy_returns.mean() / daily_std) if daily_std > 0 else np.nan
    metrics = {
        "final_multiple": float(equity.iloc[-1]) if len(equity) else np.nan,
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else np.nan,
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0)
        if len(equity) and years > 0 and equity.iloc[-1] > 0
        else np.nan,
        "max_drawdown": max_drawdown(equity) if len(equity) else np.nan,
        "sharpe": sharpe,
        "exposure": float(position.mean()) if position is not None else np.nan,
        "trades": int(((position > 0) & (position.shift(1).fillna(0) <= 0)).sum()) if position is not None else 0,
        "turnover": float(position.diff().abs().fillna(position.abs()).sum()) if position is not None else np.nan,
    }
    return equity, metrics


def backtest_position(
    dates: pd.Series,
    close: pd.Series,
    position: pd.Series,
    *,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    charge_costs: bool = True,
) -> tuple[pd.Series, pd.Series, dict]:
    returns = close.pct_change().fillna(0.0)
    position = position.astype(float).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = transaction_cost * turnover if charge_costs else 0.0
    strategy_returns = position * returns - costs
    days = max(1, int((pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days))
    equity, metrics = metrics_from_returns(strategy_returns, days=days, position=position)
    return strategy_returns, equity, metrics


def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    starts = rng.integers(0, n, size=int(math.ceil(n / block_size)))
    pieces = [(start + np.arange(block_size)) % n for start in starts]
    return np.concatenate(pieces)[:n]


def bootstrap_multiple_testing(strategy_frame: pd.DataFrame, config: RunConfig) -> dict:
    sims = config.bootstrap_sims
    if sims <= 0:
        return {"sims": 0, "note": "bootstrap disabled"}
    n = len(strategy_frame)
    rng = np.random.default_rng(config.bootstrap_seed + n)
    buyhold = strategy_frame["ret_BuyHold"].to_numpy(dtype=float)
    structure_returns = {
        name: strategy_frame[f"ret_{name}"].to_numpy(dtype=float) for name in STRUCTURE_STRATEGIES
    }
    final_multiples = {
        name: float(np.prod(1.0 + values)) for name, values in structure_returns.items()
    }
    best_signal = max(final_multiples, key=final_multiples.get)
    best_returns = structure_returns[best_signal]
    excess = np.column_stack([structure_returns[name] - buyhold for name in STRUCTURE_STRATEGIES])
    observed_best_mean_excess = float(np.nanmax(excess.mean(axis=0)))
    centered_excess = excess - excess.mean(axis=0)

    boot_final = []
    boot_sharpe = []
    boot_reality = []
    for _ in range(sims):
        idx = block_bootstrap_indices(n, max(2, config.hold_days), rng)
        sampled = best_returns[idx]
        boot_final.append(float(np.prod(1.0 + sampled)))
        std = float(np.std(sampled, ddof=0))
        boot_sharpe.append(float(np.sqrt(365.0) * np.mean(sampled) / std) if std > 0 else np.nan)
        boot_reality.append(float(np.nanmax(centered_excess[idx].mean(axis=0))))

    boot_final_arr = np.asarray(boot_final, dtype=float)
    boot_sharpe_arr = np.asarray(boot_sharpe, dtype=float)
    boot_reality_arr = np.asarray(boot_reality, dtype=float)
    return {
        "sims": int(sims),
        "block_size": int(max(2, config.hold_days)),
        "best_signal": best_signal,
        "best_signal_final_multiple": final_multiples[best_signal],
        "best_signal_final_ci_05": float(np.nanquantile(boot_final_arr, 0.05)),
        "best_signal_final_ci_95": float(np.nanquantile(boot_final_arr, 0.95)),
        "best_signal_sharpe_ci_05": float(np.nanquantile(boot_sharpe_arr, 0.05)),
        "best_signal_sharpe_ci_95": float(np.nanquantile(boot_sharpe_arr, 0.95)),
        "observed_best_daily_excess_mean": observed_best_mean_excess,
        "reality_check_p_value": float(np.mean(boot_reality_arr >= observed_best_mean_excess)),
    }


def walk_forward_evaluation(strategy_frame: pd.DataFrame, dates: pd.Series, config: RunConfig) -> dict:
    n = len(strategy_frame)
    start = config.walk_forward_train_min
    blocks = []
    selected_returns = []
    selected_dates = []

    while start + config.walk_forward_test_size <= n:
        train = slice(0, start)
        test = slice(start, start + config.walk_forward_test_size)
        train_multiples = {
            name: float(np.prod(1.0 + strategy_frame[f"ret_{name}"].iloc[train].to_numpy(dtype=float)))
            for name in STRUCTURE_STRATEGIES
        }
        selected = max(train_multiples, key=train_multiples.get)
        test_returns = strategy_frame[f"ret_{selected}"].iloc[test].reset_index(drop=True)
        test_days = max(
            1,
            int(
                (
                    pd.Timestamp(dates.iloc[start + config.walk_forward_test_size - 1])
                    - pd.Timestamp(dates.iloc[start])
                ).days
            ),
        )
        _, test_metrics = metrics_from_returns(test_returns, days=test_days)
        blocks.append(
            {
                "train_end": pd.Timestamp(dates.iloc[start - 1]).date().isoformat(),
                "test_start": pd.Timestamp(dates.iloc[start]).date().isoformat(),
                "test_end": pd.Timestamp(dates.iloc[start + config.walk_forward_test_size - 1]).date().isoformat(),
                "selected_signal": selected,
                "test_final_multiple": test_metrics["final_multiple"],
                "test_sharpe": test_metrics["sharpe"],
                "test_max_drawdown": test_metrics["max_drawdown"],
            }
        )
        selected_returns.append(test_returns)
        selected_dates.extend(dates.iloc[test].tolist())
        start += config.walk_forward_test_size

    if not selected_returns:
        return {
            "blocks": [],
            "aggregate": {
                "blocks": 0,
                "test_days": 0,
                "final_multiple": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "selected_signals": "",
            },
        }

    combined = pd.concat(selected_returns, ignore_index=True)
    test_days = max(1, int((pd.Timestamp(selected_dates[-1]) - pd.Timestamp(selected_dates[0])).days))
    _, aggregate = metrics_from_returns(combined, days=test_days)
    aggregate.update(
        {
            "blocks": len(blocks),
            "test_days": int(len(combined)),
            "selected_signals": ", ".join(block["selected_signal"] for block in blocks),
        }
    )
    return {"blocks": blocks, "aggregate": aggregate}


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
    hmm_bull: np.ndarray,
    config: RunConfig,
    *,
    write_csv: bool = True,
    include_validation: bool = True,
) -> tuple[dict, str]:
    close = pd.Series(close_array, dtype=float)
    momentum = pd.Series(logp).diff(config.momentum_days).gt(0).fillna(False)
    raw_events = {
        "BDE": online_high_signal(
            bde, use_abs=True, q=config.signal_q, min_history=config.signal_min_history
        )
        & momentum,
        "CSW": pd.Series(csw).gt(0).fillna(False) & momentum,
        "SADF": online_high_signal(sadf, q=config.signal_q, min_history=config.signal_min_history) & momentum,
        "QADF": online_high_signal(qadf, q=config.signal_q, min_history=config.signal_min_history) & momentum,
        "CADF": online_high_signal(cadf, q=config.signal_q, min_history=config.signal_min_history) & momentum,
        "SMT": online_high_signal(smt, q=config.signal_q, min_history=config.signal_min_history) & momentum,
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
        dates, close, buyhold_position, transaction_cost=config.transaction_cost, charge_costs=True
    )
    strategy_frame["pos_BuyHold"] = buyhold_position
    strategy_frame["ret_BuyHold"] = buyhold_returns
    strategy_frame["equity_BuyHold"] = buyhold_equity
    metrics["BuyHold"] = buyhold_metrics

    for name, event in raw_events.items():
        held = event.rolling(config.hold_days, min_periods=1).max().astype(bool)
        position = held.shift(1).fillna(False).astype(float)
        strategy_returns, equity, stats = backtest_position(
            dates, close, position, transaction_cost=config.transaction_cost
        )
        strategy_frame[f"event_{name}"] = event.astype(int)
        strategy_frame[f"pos_{name}"] = position
        strategy_frame[f"ret_{name}"] = strategy_returns
        strategy_frame[f"equity_{name}"] = equity
        metrics[name] = stats

    # HMM is a persistent regime state, not an alert event: hold the position
    # exactly while the filtered bull state is active, with the same one-day
    # execution shift as the alert strategies. No hold window or momentum gate.
    hmm_event = pd.Series(np.asarray(hmm_bull, dtype=bool), index=close.index)
    hmm_position = hmm_event.shift(1).fillna(False).astype(float)
    hmm_returns, hmm_equity, hmm_stats = backtest_position(
        dates, close, hmm_position, transaction_cost=config.transaction_cost
    )
    strategy_frame["event_HMM"] = hmm_event.astype(int)
    strategy_frame["pos_HMM"] = hmm_position
    strategy_frame["ret_HMM"] = hmm_returns
    strategy_frame["equity_HMM"] = hmm_equity
    metrics["HMM"] = hmm_stats

    best_signal = max(STRUCTURE_STRATEGIES, key=lambda name: metrics[name]["final_multiple"])
    best_including_buyhold = max(STRATEGY_NAMES, key=lambda name: metrics[name]["final_multiple"])
    out_csv = DATA_DIR / f"{asset}_strategy_daily.csv"
    if write_csv:
        strategy_frame.to_csv(out_csv, index=False)

    summary = {
        "strategy_file": str(out_csv.relative_to(ROOT)) if write_csv else "",
        "parameters": {
            "transaction_cost": config.transaction_cost,
            "signal_quantile": config.signal_q,
            "signal_min_history": config.signal_min_history,
            "momentum_days": config.momentum_days,
            "hold_days": config.hold_days,
            "hmm_states": config.hmm_states,
            "hmm_refit_days": config.hmm_refit_days,
            "hmm_min_history": config.hmm_min_history,
            "buyhold_entry_cost_charged": True,
        },
        "metrics": metrics,
        "best_signal_strategy": best_signal,
        "best_signal_final_multiple": metrics[best_signal]["final_multiple"],
        "best_including_buyhold": best_including_buyhold,
        "best_including_buyhold_final_multiple": metrics[best_including_buyhold]["final_multiple"],
    }
    if include_validation:
        summary["walk_forward"] = walk_forward_evaluation(strategy_frame, dates, config)
        summary["multiple_testing"] = bootstrap_multiple_testing(strategy_frame, config)
    return summary, str(out_csv.relative_to(ROOT)) if write_csv else ""


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


def max_date(dates: pd.Series, values: np.ndarray) -> str:
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return "n/a"
    pos = int(np.nanargmax(values))
    return str(pd.Timestamp(dates.iloc[pos]).date())


def calibration_cache_path(n: int, tau: int, sims: int, seed: int) -> Path:
    return CALIBRATION_DIR / f"{CALIBRATION_VERSION}_n{n}_tau{tau}_sims{sims}_seed{seed}.json"


def _simulate_adf_worker(args: tuple[int, int, int, int]) -> dict[str, list[float]]:
    n, tau, sims, seed = args
    rng = np.random.default_rng(seed)
    out = {"sadf": [], "qadf": [], "cadf": []}
    for _ in range(sims):
        y = np.cumsum(rng.normal(size=n))
        sadf, qadf, cadf = adf_family(y, tau)
        out["sadf"].append(finite_nanmax(sadf))
        out["qadf"].append(finite_nanmax(qadf))
        out["cadf"].append(finite_nanmax(cadf))
    return out


def simulate_adf_critical_values(n: int, tau: int, config: RunConfig) -> dict:
    sims = int(config.calibration_sims)
    seed = int(config.calibration_seed)
    if sims <= 0:
        return {
            "version": CALIBRATION_VERSION,
            "n": n,
            "tau": tau,
            "sims": 0,
            "seed": seed,
            "critical_values": {},
        }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    path = calibration_cache_path(n, tau, sims, seed)
    if path.exists():
        return json.loads(path.read_text())

    workers = max(1, min(config.workers, sims))
    counts = [sims // workers] * workers
    for i in range(sims % workers):
        counts[i] += 1
    tasks = [(n, tau, count, seed + 1009 * (i + 1)) for i, count in enumerate(counts) if count > 0]
    if workers == 1:
        parts = [_simulate_adf_worker(task) for task in tasks]
    else:
        with Pool(processes=workers) as pool:
            parts = pool.map(_simulate_adf_worker, tasks)

    combined = {"sadf": [], "qadf": [], "cadf": []}
    for part in parts:
        for key in combined:
            combined[key].extend(part[key])

    critical_values = {}
    for key, values in combined.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        critical_values[f"{key}_90"] = float(np.quantile(arr, 0.90)) if len(arr) else np.nan
        critical_values[f"{key}_95"] = float(np.quantile(arr, 0.95)) if len(arr) else np.nan
        critical_values[f"{key}_99"] = float(np.quantile(arr, 0.99)) if len(arr) else np.nan

    payload = {
        "version": CALIBRATION_VERSION,
        "n": n,
        "tau": tau,
        "sims": sims,
        "seed": seed,
        "critical_values": critical_values,
    }
    path.write_text(json.dumps(strict_json_value(payload), indent=2, allow_nan=False))
    return payload


def flag_summary(
    bde: np.ndarray,
    csw: np.ndarray,
    sadf: np.ndarray,
    smt: np.ndarray,
    config: RunConfig,
) -> dict[str, np.ndarray]:
    bde_retro = top_quantile_flag(np.abs(bde))
    csw_retro = np.nan_to_num(csw, nan=-np.inf) > 0
    sadf_retro = top_quantile_flag(sadf)
    smt_retro = top_quantile_flag(smt)
    retro_count = bde_retro.astype(int) + csw_retro.astype(int) + sadf_retro.astype(int) + smt_retro.astype(int)
    bde_online = online_high_signal(
        bde, use_abs=True, q=config.signal_q, min_history=config.signal_min_history
    ).to_numpy(dtype=bool)
    csw_online = np.nan_to_num(csw, nan=-np.inf) > 0
    sadf_online = online_high_signal(sadf, q=config.signal_q, min_history=config.signal_min_history).to_numpy(dtype=bool)
    smt_online = online_high_signal(smt, q=config.signal_q, min_history=config.signal_min_history).to_numpy(dtype=bool)
    online_count = (
        bde_online.astype(int) + csw_online.astype(int) + sadf_online.astype(int) + smt_online.astype(int)
    )
    return {
        "bde_retro_flag": bde_retro,
        "csw_retro_flag": csw_retro,
        "sadf_retro_flag": sadf_retro,
        "smt_retro_flag": smt_retro,
        "family_count_retro": retro_count,
        "consensus_retro": retro_count >= 2,
        "bde_online_flag": bde_online,
        "csw_online_flag": csw_online,
        "sadf_online_flag": sadf_online,
        "smt_online_flag": smt_online,
        "family_count_online": online_count,
        "consensus_online": online_count >= 2,
    }


def compute_tau_sensitivity(
    dates: pd.Series,
    close: np.ndarray,
    logp: np.ndarray,
    bde: np.ndarray,
    csw: np.ndarray,
    smt_default: np.ndarray,
    hmm_bull: np.ndarray,
    config: RunConfig,
) -> list[dict]:
    rows = []
    for frac in TAU_SENSITIVITY_FRACTIONS:
        tau = min_window(len(close), frac)
        sadf, qadf, cadf = adf_family(logp, tau)
        flags = flag_summary(bde, csw, sadf, smt_default, config)
        runs = runs_from_flag(dates, flags["consensus_retro"])
        strategies, _ = compute_strategy_suite(
            "sensitivity",
            dates,
            close,
            logp,
            bde,
            csw,
            sadf,
            qadf,
            cadf,
            smt_default,
            hmm_bull,
            config,
            write_csv=False,
            include_validation=False,
        )
        rows.append(
            {
                "tau_fraction": frac,
                "tau": tau,
                "max_sadf_date": max_date(dates, sadf),
                "run_count": len(runs),
                "best_signal_strategy": strategies["best_signal_strategy"],
                "best_signal_final_multiple": strategies["best_signal_final_multiple"],
            }
        )
    return rows


def sdfc_label(dates: pd.Series, sdfc_t: float, sdfc_idx: int | None) -> str:
    if sdfc_idx is None or not np.isfinite(sdfc_t):
        return "insufficient sample"
    if sdfc_t <= 0:
        return "none (non-positive t)"
    return str(pd.Timestamp(dates.iloc[sdfc_idx]).date())


def compute_asset(asset: str, frame: pd.DataFrame, choice: MarketChoice, config: RunConfig) -> dict:
    dates = pd.Series(pd.to_datetime(frame["date"]))
    close = frame["close"].astype(float).to_numpy()
    validate_close_series(asset, dates, close)
    logp = np.log(close)
    price_norm = close / close[0]
    tau = min_window(len(close), config.tau_fraction)

    bde = recursive_cusum(logp)
    csw = csw_excess(logp)
    sadf, qadf, cadf = adf_family(logp, tau)
    smt = smt_trend_score(logp, price_norm, tau)
    sdfc_t, sdfc_idx = sdfc(logp, tau)
    hmm_bull, hmm_states = hmm_regime_signal(
        logp,
        n_states=config.hmm_states,
        refit_days=config.hmm_refit_days,
        min_history=config.hmm_min_history,
        momentum_days=config.momentum_days,
    )
    flags = flag_summary(bde, csw, sadf, smt, config)
    runs = runs_from_flag(dates, flags["consensus_retro"])
    strategies, strategy_file = compute_strategy_suite(
        asset, dates, close, logp, bde, csw, sadf, qadf, cadf, smt, hmm_bull, config
    )
    calibration = simulate_adf_critical_values(len(close), tau, config)
    critical_values = calibration.get("critical_values", {})
    tau_sensitivity = compute_tau_sensitivity(dates, close, logp, bde, csw, smt, hmm_bull, config)

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
        "hmm_state": hmm_states,
        "hmm_bull_flag": hmm_bull.astype(int),
        **{key: value.astype(int) if value.dtype == bool else value for key, value in flags.items()},
    }
    out_csv = DATA_DIR / f"{asset}_diagnostics_daily.csv"
    pd.DataFrame(series).to_csv(out_csv, index=False)

    def exceed_count(values: np.ndarray, key: str) -> int:
        threshold = critical_values.get(key)
        if threshold is None or not np.isfinite(threshold):
            return 0
        return int(np.sum(values > threshold))

    summary = {
        **asdict(choice),
        "clock": "daily close",
        "test_rows": int(len(close)),
        "tau": int(tau),
        "tau_fraction": config.tau_fraction,
        "diagnostics_file": str(out_csv.relative_to(ROOT)),
        "sdfc_t": float(sdfc_t),
        "sdfc_candidate": sdfc_label(dates, sdfc_t, sdfc_idx),
        "max_sadf": max_date(dates, sadf),
        "max_sadf_value": finite_nanmax(sadf),
        "max_qadf_value": finite_nanmax(qadf),
        "max_cadf_value": finite_nanmax(cadf),
        "max_csw": max_date(dates, csw),
        "max_csw_value": finite_nanmax(csw),
        "max_smt": max_date(dates, smt),
        "max_smt_value": finite_nanmax(smt),
        "hmm": {
            "states": int(config.hmm_states),
            "refit_days": int(config.hmm_refit_days),
            "min_history": int(config.hmm_min_history),
            "modeled_days": int(np.sum(hmm_states >= 0)),
            "bull_days": int(np.sum(hmm_bull)),
            "first_modeled_day": (
                str(pd.Timestamp(dates.iloc[int(np.argmax(hmm_states >= 0))]).date())
                if np.any(hmm_states >= 0)
                else "none (insufficient history)"
            ),
        },
        "runs": runs,
        "run_count": int(len(runs)),
        "first_run": runs[0]["start"] if runs else "none",
        "strategy_file": strategy_file,
        "strategies": strategies,
        "calibration": calibration,
        "critical_exceedances": {
            "sadf_95_count": exceed_count(sadf, "sadf_95"),
            "qadf_95_count": exceed_count(qadf, "qadf_95"),
            "cadf_95_count": exceed_count(cadf, "cadf_95"),
        },
        "tau_sensitivity": tau_sensitivity,
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
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(11, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1, 1, 1, 1, 0.75]},
    )
    runs = result["runs"]

    axes[0].plot(diag["date"], diag["close"], color="#1f4e79", lw=1.25)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Close")
    axes[0].set_title(f"{asset}: daily {result['source']} diagnostics")
    shade_runs(axes[0], runs)
    if result["sdfc_candidate"] and not result["sdfc_candidate"].startswith(("none", "insufficient")):
        axes[0].axvline(pd.Timestamp(result["sdfc_candidate"]), color="#7b3294", ls="--", lw=1)

    axes[1].plot(diag["date"], diag["bde"], color="#276419", lw=1)
    axes[1].axhline(0, color="black", lw=0.6, alpha=0.6)
    axes[1].set_ylabel("BDE-inspired")
    shade_runs(axes[1], runs)

    axes[2].plot(diag["date"], diag["csw_excess"], color="#b35806", lw=1)
    axes[2].axhline(0, color="black", lw=0.8, alpha=0.7)
    axes[2].set_ylabel("2-sided CSW excess")
    shade_runs(axes[2], runs)

    axes[3].plot(diag["date"], diag["sadf"], label="SADF", color="#542788", lw=1)
    axes[3].plot(diag["date"], diag["qadf"], label="QADF 95%", color="#2d708e", lw=1)
    axes[3].plot(diag["date"], diag["cadf"], label="CADF tail mean", color="#5aae61", lw=1)
    axes[3].set_ylabel("ADF t-stat")
    axes[3].legend(fontsize=8, ncol=3, loc="upper left")
    shade_runs(axes[3], runs)

    axes[4].plot(diag["date"], diag["smt"], color="#8c510a", lw=1)
    axes[4].set_ylabel("SMT score")
    shade_runs(axes[4], runs)

    hmm_states = int(result.get("hmm", {}).get("states", DEFAULT_HMM_STATES))
    state = diag["hmm_state"].astype(float).where(diag["hmm_state"] >= 0)
    bull = diag["hmm_bull_flag"].astype(bool)
    axes[5].fill_between(
        diag["date"],
        -0.5,
        hmm_states - 0.5,
        where=bull,
        color="#1a9850",
        alpha=0.22,
        step="mid",
        lw=0,
    )
    axes[5].plot(diag["date"], state, color="#d73027", lw=1, drawstyle="steps-mid")
    axes[5].set_ylim(-0.6, hmm_states - 0.4)
    axes[5].set_yticks(range(hmm_states))
    axes[5].set_ylabel("HMM state")
    axes[5].set_xlabel("Date")
    if not (diag["hmm_state"] >= 0).any():
        axes[5].text(
            0.5,
            0.5,
            "insufficient history for HMM",
            transform=axes[5].transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
        )

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
        counts = pd.Series(diag["consensus_retro"].astype(int).to_numpy(), index=months).groupby(level=0).sum()
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
    ax.set_title("Retrospective consensus alert observations by month")
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
        "HMM": "#d73027",
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
    fig.savefig(FIG_DIR / f"fig_strategy_pnl_{asset}.png", dpi=180)
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
    ax.set_xticks(
        np.arange(len(STRATEGY_NAMES)),
        ["Buy&hold", "BDE", "CSW", "SADF", "QADF", "CADF", "SMT", "HMM", "Consensus"],
        rotation=35,
        ha="right",
    )
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


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def compress_signal_path(path: str) -> str:
    """Compress consecutive repeats: 'CSW, CSW, HMM' -> 'CSW x2, HMM'."""

    names = [part.strip() for part in path.split(",") if part.strip()]
    if not names:
        return "n/a"
    parts: list[str] = []
    run_name, run_len = names[0], 1
    for name in names[1:]:
        if name == run_name:
            run_len += 1
            continue
        parts.append(run_name if run_len == 1 else f"{run_name} x{run_len}")
        run_name, run_len = name, 1
    parts.append(run_name if run_len == 1 else f"{run_name} x{run_len}")
    return ", ".join(parts)


def strategy_label(name: str) -> str:
    return "Buy-and-hold" if name == "BuyHold" else name


def render_rows(results: dict[str, dict], config: RunConfig) -> dict[str, str | list[str]]:
    summary_lines = []
    critical_lines = []
    run_lines = []
    tau_lines = []
    strategy_best_lines = []
    strategy_detail_lines = []
    walk_forward_lines = []
    bootstrap_lines = []
    asset_sections = []
    source_notes = []

    for asset, r in results.items():
        summary_lines.append(
            f"{asset} & {latex_escape(r['source'])} & {r['raw_start']} & {r['raw_end']} & "
            f"{r['raw_rows']} & {r['tau']} & {latex_escape(r['sdfc_candidate'])} & "
            f"{num(r['sdfc_t'])} & {r['max_sadf']} & {r['max_csw']} & {r['max_smt']} & {r['run_count']} \\\\"
        )
        crit = r["calibration"]["critical_values"]
        exceed = r["critical_exceedances"]
        critical_lines.append(
            f"{asset} & {r['calibration']['sims']} & {r['tau']} & "
            f"{num(crit.get('sadf_95', np.nan))} & {num(r['max_sadf_value'])} & {exceed['sadf_95_count']} & "
            f"{num(crit.get('qadf_95', np.nan))} & {num(r['max_qadf_value'])} & {exceed['qadf_95_count']} & "
            f"{num(crit.get('cadf_95', np.nan))} & {num(r['max_cadf_value'])} & {exceed['cadf_95_count']} \\\\"
        )
        first = True
        for run in r["runs"][:8]:
            label = asset if first else ""
            first = False
            run_lines.append(f"{label} & {run['start']} & {run['end']} & {run['days']} \\\\")
        for row in r["tau_sensitivity"]:
            tau_lines.append(
                f"{asset} & {int(row['tau_fraction'] * 100)}\\% & {row['tau']} & "
                f"{row['max_sadf_date']} & {row['run_count']} & "
                f"{strategy_label(row['best_signal_strategy'])} & {mult(row['best_signal_final_multiple'])} \\\\"
            )
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
        wf = strat["walk_forward"]["aggregate"]
        walk_forward_lines.append(
            f"{asset} & {wf['blocks']} & {wf['test_days']} & {latex_escape(compress_signal_path(wf['selected_signals']))} & "
            f"{mult(wf['final_multiple'])} & {num(wf['sharpe'])} & {pct(wf['max_drawdown'])} \\\\"
        )
        mt = strat["multiple_testing"]
        bootstrap_lines.append(
            f"{asset} & {mt.get('sims', 0)} & {strategy_label(mt.get('best_signal', best_signal))} & "
            f"{mult(mt.get('best_signal_final_multiple', np.nan))} & "
            f"[{mult(mt.get('best_signal_final_ci_05', np.nan))}, {mult(mt.get('best_signal_final_ci_95', np.nan))}] & "
            f"[{num(mt.get('best_signal_sharpe_ci_05', np.nan))}, {num(mt.get('best_signal_sharpe_ci_95', np.nan))}] & "
            f"{num(mt.get('reality_check_p_value', np.nan), 3)} \\\\"
        )
        sdfc_sentence = (
            f"The SDFC candidate is {latex_escape(r['sdfc_candidate'])} with t-statistic {num(r['sdfc_t'])}."
            if not r["sdfc_candidate"].startswith("insufficient")
            else "The SDFC sample guard marks this series as insufficient for that statistic."
        )
        hmm_info = r.get("hmm", {})
        if hmm_info.get("modeled_days", 0) > 0:
            hmm_sentence = (
                f" The walk-forward HMM covers {hmm_info['modeled_days']} daily observations from "
                f"{hmm_info['first_modeled_day']} and spends {hmm_info['bull_days']} of them in its bull state."
            )
        else:
            hmm_sentence = " The series is too short for the walk-forward HMM, so its strategy stays flat."
        hype_note = (
            " HYPE uses Binance USD-M futures price klines; strategy returns exclude funding, carry, basis, and futures-specific execution costs."
            if asset == "HYPE"
            else ""
        )
        asset_sections.append(
            "\n".join(
                [
                    f"\\subsection{{{asset}}}",
                    "",
                    f"{r['run_count']} retrospective consensus alert run(s) were flagged on the daily clock. "
                    f"The first listed run begins on {r['first_run']}. {sdfc_sentence} "
                    f"SADF peaks on {r['max_sadf']}, the two-sided CSW heuristic peaks on {r['max_csw']}, "
                    f"and the SMT score peaks on {r['max_smt']}.{hmm_sentence}{hype_note}",
                    "",
                    "\\begin{figure}[H]",
                    "\\centering",
                    f"\\includegraphics[width=0.98\\textwidth]{{regime_chapter_figs/fig_asset_{asset}.pdf}}",
                    f"\\caption{{{asset} structural-break diagnostics. Orange shading is retrospective consensus, not an online trading signal. Green shading in the bottom panel marks the filtered HMM bull state, which is the online HMM strategy state.}}",
                    "\\end{figure}",
                ]
            )
        )
        source_notes.append(
            f"{asset}: {r['source']} {r['symbol']} {r['raw_start']} to {r['raw_end']} "
            f"({r['raw_rows']} rows); {r['reason']}. Spot status: {r['spot_status']}. "
            f"Futures status: {r['futures_status']}."
        )

    parameter_lines = [
        f"Transaction cost & {pct(config.transaction_cost, 2)} per exposure change \\\\",
        f"Online signal quantile & {config.signal_q:.2f} prior expanding quantile \\\\",
        f"Minimum online history & {config.signal_min_history} finite observations \\\\",
        f"Momentum gate & trailing {config.momentum_days}-day log return must be positive \\\\",
        f"Holding window & {config.hold_days} calendar daily observations after an alert \\\\",
        f"Minimum diagnostic window & $\\max(60, \\lceil {config.tau_fraction:.2f} n \\rceil)$ \\\\",
        f"HMM regime model & {config.hmm_states} states, refit every {config.hmm_refit_days} obs, min {config.hmm_min_history} training obs \\\\",
        f"ADF Monte Carlo calibration & {config.calibration_sims} random-walk simulations, seed {config.calibration_seed} \\\\",
        f"Walk-forward validation & expanding {config.walk_forward_train_min}-day train, {config.walk_forward_test_size}-day test blocks \\\\",
    ]

    inference_lines = [
        "BDE-inspired CUSUM & shifted expanding residual z-score & no & yes for strategy, no for retrospective plot flags \\\\",
        "Two-sided CSW excess & book-inspired curve applied symmetrically & heuristic & yes \\\\",
        "SADF/QADF/CADF visual flags & full-sample top 5\\% & no & no \\\\",
        "SADF/QADF/CADF critical exceedances & simulated random-walk finite-sample critical values & approximate & no, retrospective inference only \\\\",
        "SDFC & full-sample unknown-break scan & no in this report & no \\\\",
        "HMM regime filter & walk-forward Gaussian HMM, filtered (causal) state probabilities & no & yes \\\\",
        "Long/flat strategies & prior expanding thresholds plus one-day execution shift & no & yes \\\\",
    ]

    return {
        "SUMMARY_ROWS": "\n".join(summary_lines),
        "CRITICAL_ROWS": "\n".join(critical_lines),
        "RUN_ROWS": "\n".join(run_lines),
        "TAU_SENSITIVITY_ROWS": "\n".join(tau_lines),
        "STRATEGY_BEST_ROWS": "\n".join(strategy_best_lines),
        "STRATEGY_DETAIL_ROWS": "\n".join(strategy_detail_lines),
        "WALK_FORWARD_ROWS": "\n".join(walk_forward_lines),
        "BOOTSTRAP_ROWS": "\n".join(bootstrap_lines),
        "PARAMETER_ROWS": "\n".join(parameter_lines),
        "INFERENCE_ROWS": "\n".join(inference_lines),
        "ASSET_SECTIONS": "\n\n".join(asset_sections),
        "SOURCE_NOTES": source_notes,
    }


def render_report(results: dict[str, dict], payload: dict, config: RunConfig) -> None:
    if not REPORT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"missing report template: {REPORT_TEMPLATE_PATH}")
    rows = render_rows(results, config)
    template = REPORT_TEMPLATE_PATH.read_text()
    replacements = {
        "GENERATED_DATE": datetime.now(timezone.utc).date().isoformat(),
        "LAST_CLOSED_DAY": payload["last_closed_day"],
        **{key: value for key, value in rows.items() if isinstance(value, str)},
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"%%{key}%%", str(value))
    unresolved = [part for part in rendered.split() if part.startswith("%%") and part.endswith("%%")]
    if unresolved:
        raise RuntimeError(f"unresolved template placeholders: {unresolved[:5]}")
    REPORT_PATH.write_text(rendered)
    snippet = {key.lower(): value for key, value in rows.items()}
    SNIPPET_PATH.write_text(json.dumps(strict_json_value(snippet), indent=2, allow_nan=False))


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="Regenerate crypto regime-change report artifacts.")
    parser.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    parser.add_argument("--signal-q", type=float, default=DEFAULT_SIGNAL_Q)
    parser.add_argument("--signal-min-history", type=int, default=DEFAULT_SIGNAL_MIN_HISTORY)
    parser.add_argument("--hold-days", type=int, default=DEFAULT_HOLD_DAYS)
    parser.add_argument("--momentum-days", type=int, default=DEFAULT_MOMENTUM_DAYS)
    parser.add_argument("--transaction-cost", type=float, default=DEFAULT_TRANSACTION_COST)
    parser.add_argument("--tau-fraction", type=float, default=DEFAULT_TAU_FRACTION)
    parser.add_argument("--calibration-sims", type=int, default=DEFAULT_CALIBRATION_SIMS)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    parser.add_argument("--bootstrap-sims", type=int, default=DEFAULT_BOOTSTRAP_SIMS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--walk-forward-train-min", type=int, default=DEFAULT_WALK_FORWARD_TRAIN_MIN)
    parser.add_argument("--walk-forward-test-size", type=int, default=DEFAULT_WALK_FORWARD_TEST_SIZE)
    parser.add_argument("--hmm-states", type=int, default=DEFAULT_HMM_STATES)
    parser.add_argument("--hmm-refit-days", type=int, default=DEFAULT_HMM_REFIT_DAYS)
    parser.add_argument("--hmm-min-history", type=int, default=DEFAULT_HMM_MIN_HISTORY)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    return RunConfig(
        assets=tuple(args.assets),
        transaction_cost=args.transaction_cost,
        signal_q=args.signal_q,
        signal_min_history=args.signal_min_history,
        momentum_days=args.momentum_days,
        hold_days=args.hold_days,
        tau_fraction=args.tau_fraction,
        calibration_sims=args.calibration_sims,
        calibration_seed=args.calibration_seed,
        bootstrap_sims=args.bootstrap_sims,
        bootstrap_seed=args.bootstrap_seed,
        walk_forward_train_min=args.walk_forward_train_min,
        walk_forward_test_size=args.walk_forward_test_size,
        hmm_states=max(1, args.hmm_states),
        hmm_refit_days=max(1, args.hmm_refit_days),
        hmm_min_history=max(30, args.hmm_min_history),
        use_cache=args.use_cache,
        skip_download=args.skip_download,
        workers=max(1, args.workers),
    )


def main(argv: Sequence[str] | None = None):
    config = parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if config.skip_download:
        now_dt = datetime.now(timezone.utc)
        last_closed_day_start = int(now_dt.timestamp() * 1000)
    else:
        now_ms = server_time_ms()
        now_dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        today_start = datetime(now_dt.year, now_dt.month, now_dt.day, tzinfo=timezone.utc)
        last_closed_day_start = int(today_start.timestamp() * 1000) - DAY_MS

    choices: dict[str, MarketChoice] = {}
    frames: dict[str, pd.DataFrame] = {}
    for asset in config.assets:
        choice, frame = choose_market(asset, SYMBOLS[asset], last_closed_day_start, config)
        choices[asset] = choice
        frames[asset] = frame
        print(
            f"{asset}: {choice.source} {choice.symbol}, "
            f"{choice.raw_start} to {choice.raw_end}, {choice.raw_rows} rows"
        )

    results: dict[str, dict] = {}
    for asset in config.assets:
        print(f"Computing diagnostics for {asset}...")
        results[asset] = compute_asset(asset, frames[asset], choices[asset], config)

    plot_normalized(results)
    for asset in config.assets:
        plot_asset(asset, results[asset])
        plot_strategy_asset(asset, results[asset])
    plot_heatmap(results)
    plot_strategy_heatmap(results)

    last_closed_day = max(str(frame["date"].iloc[-1]) for frame in frames.values())
    payload = {
        "generated_at_utc": now_dt.isoformat(),
        "last_closed_day": last_closed_day if config.skip_download else utc_date(last_closed_day_start),
        "config": asdict(config),
        "assets": results,
    }
    RESULTS_PATH.write_text(json.dumps(strict_json_value(payload), indent=2, allow_nan=False))
    render_report(results, strict_json_value(payload), config)
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote figures to {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
