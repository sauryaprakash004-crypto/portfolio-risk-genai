import numpy as np
import pandas as pd
from conftest import pipeline_output, returns
from portfolio_risk import backtesting as bt, config as cfg, risk_metrics as rm


def test_kupiec_known_values():
    ok = bt.kupiec_pof_test(1000, 10, 0.99)               # exactly the expected number of breaches
    assert ok["kupiec_lr"] < 1e-9 and not ok["model_rejected_5pct"]
    bad = bt.kupiec_pof_test(1000, 30, 0.99)              # 3x too many breaches
    assert bad["model_rejected_5pct"] and bad["kupiec_p_value"] < 0.001
    assert bt.kupiec_pof_test(250, 0, 0.99)["kupiec_lr"] > 0   # zero breaches handled (no log(0))


def test_breach_counting_and_no_lookahead():
    pr = rm.portfolio_returns(returns(), cfg.POLICY_WEIGHTS)
    f = bt.rolling_var_forecasts(pr, 0.99, window=250)
    assert len(f) == len(pr) - 250
    assert f["Breach"].sum() == ((-f["Realized_Return"]) > f["VaR_Forecast"]).sum()
    # first forecast uses ONLY the first 250 returns
    assert abs(f["VaR_Forecast"].iloc[0] - np.quantile(-pr.values[:250], 0.99)) < 1e-12


def test_summary_rates_consistent():
    _, art = pipeline_output()
    b = art["backtest"]
    assert np.allclose(b["breach_rate"], b["breaches"] / b["days_tested"])
    assert np.allclose(b["expected_breach_rate"], 1 - b["confidence"])
    assert set(b["method"]) == {"historical", "monte_carlo"}
