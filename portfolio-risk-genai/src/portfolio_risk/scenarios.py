"""
Scenario engine: "what happens to my portfolio if market conditions suddenly deteriorate?"

Every scenario is described by three simple knobs:
    return shock   : instant % price move per asset            (default: none)
    vol multiplier : every asset's volatility x k              (default: 1)
    corr blend     : move correlations toward 0.90 by this %   (default: 0)

Outputs per scenario:
    Return_Impact  - instant portfolio return  = w' shock          (0 for scenarios with no price shock)
    Loss_Amount    - money lost from that shock (portfolio_value x max(-Return_Impact, 0))
    Stressed_Vol / Stressed_VaR_99 / Stressed_ES_99 - risk after the vol/correlation change
                     (Monte Carlo, same seed as the baseline, so changes come ONLY from the scenario)
    Change_*       - stressed value minus baseline value
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .risk_metrics import expected_shortfall, simulate_portfolio_losses, value_at_risk

SCENARIOS = {
    "Baseline":           dict(shock=None,             vol_mult=1.0,             corr_blend=0.0),
    "Market Crash":       dict(shock=cfg.CRASH_SHOCK,  vol_mult=1.0,             corr_blend=0.0),
    "Volatility Shock":   dict(shock=None,             vol_mult=cfg.VOL_MULTIPLIER, corr_blend=0.0),
    "Correlation Shock":  dict(shock=None,             vol_mult=1.0,             corr_blend=cfg.CORR_BLEND),
    "Combined Stress":    dict(shock=cfg.CRASH_SHOCK,  vol_mult=cfg.VOL_MULTIPLIER, corr_blend=cfg.CORR_BLEND),
}


def stressed_covariance(cov_daily: np.ndarray, vol_mult: float = 1.0, corr_blend: float = 0.0,
                        corr_target: float = cfg.CORR_TARGET) -> np.ndarray:
    """Sigma_stressed = D C_stressed D  with  D = diag(vol_mult * sigma_i)  and
    C_stressed = (1 - b) * C + b * C_target   (C_target: all off-diagonals = corr_target).
    A blend of two valid correlation matrices is itself a valid correlation matrix."""
    sd = np.sqrt(np.diag(cov_daily))
    C = cov_daily / np.outer(sd, sd)
    n = len(sd)
    C_target = np.full((n, n), corr_target) + np.eye(n) * (1 - corr_target)
    C_s = (1 - corr_blend) * C + corr_blend * C_target
    sd_s = sd * vol_mult
    return C_s * np.outer(sd_s, sd_s)


def run_scenarios(returns: pd.DataFrame, weights: pd.Series,
                  portfolio_value: float = cfg.PORTFOLIO_VALUE) -> pd.DataFrame:
    w = weights.reindex(returns.columns)
    mu_d, cov_d = returns.mean().values, returns.cov().values
    rows = []
    for name, sc in SCENARIOS.items():
        ret_impact = 0.0 if sc["shock"] is None else float(w @ sc["shock"].reindex(returns.columns))
        cov_s = stressed_covariance(cov_d, sc["vol_mult"], sc["corr_blend"])
        losses = simulate_portfolio_losses(mu_d, cov_s, w.values)
        rows.append({
            "Scenario": name,
            "Return_Impact": ret_impact,
            "Loss_Amount": portfolio_value * max(0.0, -ret_impact),
            "Stressed_Vol": float(np.sqrt(w.values @ cov_s @ w.values) * np.sqrt(cfg.TRADING_DAYS)),
            "Stressed_VaR_99": value_at_risk(losses, 0.99),
            "Stressed_ES_99": expected_shortfall(losses, 0.99),
        })
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    for col in ("Return_Impact", "Stressed_Vol", "Stressed_VaR_99", "Stressed_ES_99"):
        df[f"Change_{col.replace('Stressed_', '')}"] = df[col] - base[col]
    return df


def run_scenarios_all(returns: pd.DataFrame, portfolios: dict) -> pd.DataFrame:
    """Same scenarios for several portfolios -> one long table with a 'Portfolio' column."""
    frames = [run_scenarios(returns, w).assign(Portfolio=name) for name, w in portfolios.items()]
    df = pd.concat(frames, ignore_index=True)
    return df[["Portfolio"] + [c for c in df.columns if c != "Portfolio"]]
