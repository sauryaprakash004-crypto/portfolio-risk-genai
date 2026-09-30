import numpy as np
from conftest import returns
from portfolio_risk import config as cfg, scenarios as sc


def test_crash_return_equals_weights_dot_shock_and_baseline_is_zero():
    df = sc.run_scenarios(returns(), cfg.POLICY_WEIGHTS).set_index("Scenario")
    expected = float((cfg.POLICY_WEIGHTS * cfg.CRASH_SHOCK).sum())
    assert abs(df.loc["Market Crash", "Return_Impact"] - expected) < 1e-12
    assert abs(df.loc["Market Crash", "Loss_Amount"] - (-expected) * cfg.PORTFOLIO_VALUE) < 1e-6
    assert (df.loc["Baseline", [c for c in df.columns if c.startswith("Change_")]] == 0).all()
    assert df.loc["Volatility Shock", "Return_Impact"] == 0


def test_vol_shock_scales_volatility_and_corr_shock_raises_it():
    df = sc.run_scenarios(returns(), cfg.POLICY_WEIGHTS).set_index("Scenario")
    base = df.loc["Baseline", "Stressed_Vol"]
    assert abs(df.loc["Volatility Shock", "Stressed_Vol"] - cfg.VOL_MULTIPLIER * base) < 1e-9
    assert df.loc["Correlation Shock", "Stressed_Vol"] > base
    assert df.loc["Combined Stress", "Stressed_ES_99"] > df.loc["Volatility Shock", "Stressed_ES_99"]
    assert (df["Stressed_ES_99"] >= df["Stressed_VaR_99"]).all()


def test_stressed_covariance_is_valid():
    cov = returns().cov().values
    s = sc.stressed_covariance(cov, 2.0, 0.5)
    assert np.allclose(s, s.T) and np.linalg.eigvalsh(s).min() > 0
    assert np.allclose(np.sqrt(np.diag(s)), 2.0 * np.sqrt(np.diag(cov)))      # volatilities scaled
    assert np.allclose(sc.stressed_covariance(cov, 1.0, 0.0), cov)             # no shock -> unchanged
