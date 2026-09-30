"""
VaR backtesting: compare each day's VaR forecast with the loss that actually happened next.

For every day t we forecast a 1-day VaR using ONLY the previous `window` days (no look-ahead),
then check whether the realised loss on day t was larger than that forecast (a "breach").
If the model is right, breaches should occur on (1 - confidence) of the days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from . import config as cfg
from .risk_metrics import simulate_portfolio_losses, value_at_risk


def rolling_var_forecasts(port_returns: pd.Series, alpha: float = 0.99,
                          window: int = cfg.BACKTEST_WINDOW, method: str = "historical") -> pd.DataFrame:
    """method='historical': VaR = alpha-quantile of the last `window` losses.
       method='monte_carlo': fit mean/vol on the window, simulate 5,000 losses (fixed seed), take the quantile."""
    vals = port_returns.values
    z = np.random.default_rng(cfg.MC_SEED).standard_normal(5000)     # same draws every day
    var_fc = []
    for t in range(window, len(vals)):
        hist = vals[t - window:t]
        if method == "historical":
            var_fc.append(value_at_risk(-hist, alpha))
        elif method == "monte_carlo":
            sim_losses = -(hist.mean() + hist.std(ddof=1) * z)
            var_fc.append(value_at_risk(sim_losses, alpha))
        else:
            raise ValueError("method must be 'historical' or 'monte_carlo'")
    out = pd.DataFrame({"VaR_Forecast": var_fc, "Realized_Return": vals[window:]},
                       index=port_returns.index[window:])
    out["Breach"] = (-out["Realized_Return"]) > out["VaR_Forecast"]
    return out


def kupiec_pof_test(n_obs: int, n_breaches: int, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures test.
    H0: true breach probability = 1 - alpha.   LR ~ chi-square(1) under H0.
    Small p-value (< 5%) => breach rate is statistically different from expected => model rejected."""
    p, x, n = 1 - alpha, n_breaches, n_obs
    ll_null = (n - x) * np.log(1 - p) + (x * np.log(p) if x else 0.0)
    q = x / n
    ll_alt = (n - x) * np.log(1 - q) + (x * np.log(q) if x else 0.0)
    lr = float(-2 * (ll_null - ll_alt))
    pval = float(1 - chi2.cdf(lr, df=1))
    return {"kupiec_lr": lr, "kupiec_p_value": pval, "model_rejected_5pct": pval < 0.05}


def backtest_summary(port_returns: pd.Series, alphas=cfg.CONFIDENCE_LEVELS,
                     window: int = cfg.BACKTEST_WINDOW, method: str = "historical"):
    """Return (summary DataFrame, {alpha: forecast DataFrame})."""
    rows, detail = [], {}
    for a in alphas:
        bt = rolling_var_forecasts(port_returns, a, window, method)
        n, x = len(bt), int(bt["Breach"].sum())
        k = kupiec_pof_test(n, x, a)
        rows.append({"method": method, "confidence": a, "days_tested": n, "breaches": x,
                     "breach_rate": x / n, "expected_breach_rate": 1 - a,
                     "expected_breaches": n * (1 - a), **k,
                     "interpretation": interpret_backtest(x, n, a, k["model_rejected_5pct"])})
        detail[a] = bt
    return pd.DataFrame(rows), detail


def interpret_backtest(x: int, n: int, alpha: float, rejected: bool) -> str:
    rate, exp = x / n, 1 - alpha
    if not rejected:
        return (f"{x} breaches in {n} days ({rate:.2%}) vs {exp:.2%} expected: statistically consistent "
                f"with a well-calibrated {alpha:.0%} VaR.")
    direction = "UNDERESTIMATES risk (too many breaches)" if rate > exp else "OVERESTIMATES risk (too few breaches)"
    return (f"{x} breaches in {n} days ({rate:.2%}) vs {exp:.2%} expected: the {alpha:.0%} VaR "
            f"{direction}; the Kupiec test rejects the model.")
