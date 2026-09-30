import numpy as np
import pandas as pd
from scipy.stats import norm
from conftest import inputs, returns
from portfolio_risk import config as cfg, risk_metrics as rm


def test_volatility_and_sharpe_formulas():
    w = np.array([0.5, 0.5]); cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    assert abs(rm.portfolio_volatility(w, cov) - np.sqrt(0.25 * 0.04 + 0.25 * 0.09)) < 1e-12
    assert abs(rm.sharpe_ratio(0.08, 0.10, rf=0.02) - 0.6) < 1e-12


def test_es_is_at_least_var_and_matches_manual_tail_average():
    losses = np.arange(1, 101, dtype=float)            # 1..100
    assert rm.value_at_risk(losses, 0.95) == np.quantile(losses, 0.95)
    assert abs(rm.expected_shortfall(losses, 0.95) - np.mean([96, 97, 98, 99, 100])) < 1e-12
    port = rm.portfolio_returns(returns(), cfg.POLICY_WEIGHTS)
    for a in (0.95, 0.99):
        v, e = rm.historical_var_es(port, a)
        assert e >= v > 0


def test_monte_carlo_is_reproducible_and_matches_analytic_normal():
    R = returns(); w = cfg.POLICY_WEIGHTS
    a = rm.monte_carlo_var_es(R, w, 0.99); b = rm.monte_carlo_var_es(R, w, 0.99)
    assert a == b                                        # same seed -> identical
    c = rm.monte_carlo_var_es(R, w, 0.99, seed=1)
    assert a != c                                        # different seed -> different draws
    wv = w.reindex(R.columns).values
    mu_p, sd_p = R.mean().values @ wv, np.sqrt(wv @ R.cov().values @ wv)
    analytic = -(mu_p - norm.ppf(0.99) * sd_p)
    assert abs(a[0] - analytic) / analytic < 0.03        # MC ~ closed form for normal


def test_drawdown_known_example():
    r = pd.Series([0.10, -0.20, 0.05, 0.10], index=pd.bdate_range("2024-01-01", periods=4))
    dd = rm.drawdown_series(r)
    assert abs(dd["Drawdown"].min() - (-0.20)) < 1e-12
    assert dd["Drawdown"].iloc[0] == 0
    assert rm.max_drawdown(r)["trough_date"] == "2024-01-02"


def test_risk_summary_has_all_metrics_and_no_nans():
    s = rm.risk_summary(returns(), cfg.POLICY_WEIGHTS)
    for k in ("annual_return", "annual_volatility", "sharpe", "hist_var_95", "hist_es_99", "mc_var_95", "mc_es_99",
              "max_drawdown"):
        assert k in s and np.isfinite(s[k])
