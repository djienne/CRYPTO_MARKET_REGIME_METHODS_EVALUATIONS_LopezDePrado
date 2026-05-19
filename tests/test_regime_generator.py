from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_daily_regime_chapter.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("regime_generator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["regime_generator"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


g = load_generator()


def test_ols_tstat_matches_bruteforce_lstsq():
    x = np.arange(1.0, 9.0)
    y = 0.25 * x + np.array([0.10, -0.05, 0.02, 0.04, -0.01, 0.03, -0.02, 0.01])
    got = g._ols_beta_tstat_intercept_from_sums(
        m=np.array([len(x)]),
        sx=np.array([x.sum()]),
        sy=np.array([y.sum()]),
        sxx=np.array([(x * x).sum()]),
        sxy=np.array([(x * y).sum()]),
        syy=np.array([(y * y).sum()]),
    )[0]

    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    sigma2 = float(np.dot(resid, resid) / (len(x) - 2))
    inv = np.linalg.inv(design.T @ design)
    expected = beta[1] / math.sqrt(sigma2 * inv[1, 1])
    assert np.isclose(got, expected)


def test_adf_family_matches_bruteforce_for_one_endpoint():
    logp = np.array([1.00, 1.02, 1.01, 1.04, 1.08, 1.07, 1.11, 1.15, 1.14, 1.18])
    tau = 4
    sadf, qadf, cadf = g.adf_family(logp, tau)
    end = 8
    brute = []
    for start in range(0, end - tau + 1):
        x = logp[:-1][start:end]
        dy = np.diff(logp)[start:end]
        design = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(design, dy, rcond=None)[0]
        resid = dy - design @ beta
        sigma2 = float(np.dot(resid, resid) / (len(x) - 2))
        inv = np.linalg.inv(design.T @ design)
        brute.append(beta[1] / math.sqrt(sigma2 * inv[1, 1]))
    brute = np.asarray(brute)
    q = np.quantile(brute, 0.95)
    assert np.isclose(sadf[end], np.max(brute))
    assert np.isclose(qadf[end], q)
    assert np.isclose(cadf[end], np.mean(brute[brute >= q]))


def test_recursive_cusum_does_not_change_when_future_data_is_added():
    rng = np.random.default_rng(42)
    logp = np.cumsum(rng.normal(0.001, 0.03, size=140))
    full = g.recursive_cusum(logp)
    for end in (70, 95, 139):
        prefix = g.recursive_cusum(logp[: end + 1])
        assert np.isclose(full[end], prefix[end], equal_nan=True)


def test_csw_flat_series_has_no_positive_excess_and_jump_flags():
    flat = np.zeros(30)
    flat_csw = g.csw_excess(flat)
    assert not np.any(flat_csw[np.isfinite(flat_csw)] > 0)

    jump = np.r_[np.zeros(25), np.full(5, 2.0)]
    jump_csw = g.csw_excess(jump)
    assert g.finite_nanmax(jump_csw) > 0


def test_backtest_position_shift_convention_can_trade_next_return():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    close = pd.Series([100.0, 110.0, 121.0])
    position = pd.Series([0.0, 1.0, 0.0])
    returns, equity, metrics = g.backtest_position(
        dates, close, position, transaction_cost=0.0, charge_costs=False
    )
    assert np.allclose(returns.to_numpy(), [0.0, 0.10, 0.0])
    assert np.isclose(equity.iloc[-1], 1.10)
    assert metrics["trades"] == 1


def test_strict_json_value_converts_nonfinite_values():
    payload = {"a": np.float64(np.nan), "b": [np.inf, -np.inf, 1.0]}
    assert g.strict_json_value(payload) == {"a": None, "b": [None, None, 1.0]}


def test_sdfc_insufficient_sample_is_explicit():
    value, idx = g.sdfc(np.arange(20.0), tau=12)
    assert np.isnan(value)
    assert idx is None


def test_report_template_placeholders_are_known():
    text = (ROOT / "regime_detection_crypto_report_template.tex").read_text()
    placeholders = set(re.findall(r"%%([A-Z_]+)%%", text))
    assert placeholders == {
        "GENERATED_DATE",
        "LAST_CLOSED_DAY",
        "SUMMARY_ROWS",
        "PARAMETER_ROWS",
        "INFERENCE_ROWS",
        "ASSET_SECTIONS",
        "RUN_ROWS",
        "CRITICAL_ROWS",
        "TAU_SENSITIVITY_ROWS",
        "STRATEGY_BEST_ROWS",
        "STRATEGY_DETAIL_ROWS",
        "WALK_FORWARD_ROWS",
        "BOOTSTRAP_ROWS",
    }
