"""
Portfolio risk metrics.  Conventions used everywhere in this project:
  * returns are simple daily returns; a LOSS is  -return  (so losses are positive numbers)
  * VaR / ES are 1-day, expressed as a fraction of portfolio value (0.02 = 2%)
  * annualised return = mean daily return x 252, annualised vol = daily vol x sqrt(252)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------- basic inputs
def estimate_inputs(returns: pd.DataFrame):
    """Annualised expected returns (mu) and covariance matrix (Sigma) from daily data."""
    mu = returns.mean() * cfg.TRADING_DAYS
    cov = returns.cov() * cfg.TRADING_DAYS
    return mu, cov


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    return returns @ weights.reindex(returns.columns)


# ------------------------------------------------------- return / vol / Sharpe
def portfolio_volatility(weights, cov) -> float:
    """sigma_p = sqrt(w' Sigma w)   (cov and result both annualised)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ np.asarray(cov) @ w))


def portfolio_return(weights, mu) -> float:
    """R_p = w' mu   (annualised)."""
    return float(np.asarray(weights, dtype=float) @ np.asarray(mu))


def sharpe_ratio(ret: float, vol: float, rf: float = cfg.RISK_FREE_RATE) -> float:
    """Sharpe = (R_p - R_f) / sigma_p"""
    return float((ret - rf) / vol)


# ------------------------------------------------------------------ VaR and ES
def value_at_risk(losses, alpha: float) -> float:
    """VaR_alpha = the alpha-quantile of the loss distribution: 'losses stay below this
    value alpha% of the time'."""
    return float(np.quantile(np.asarray(losses), alpha))


def expected_shortfall(losses, alpha: float) -> float:
    """ES_alpha (= CVaR) = average loss in the worst (1-alpha) tail.
    Computed exactly as the mean of the worst k=(1-alpha)*N losses (fractional last
    observation), which is the same quantity the CVaR optimiser minimises."""
    x = np.sort(np.asarray(losses, dtype=float))[::-1]
    k = (1 - alpha) * len(x)
    m = int(np.floor(k))
    tail = x[:m].sum() + (k - m) * (x[m] if m < len(x) else 0.0)
    return float(tail / k)


def historical_var_es(port_returns: pd.Series, alpha: float = 0.95):
    """Historical simulation: use the portfolio's actual past returns as the distribution."""
    losses = -np.asarray(port_returns)
    return value_at_risk(losses, alpha), expected_shortfall(losses, alpha)


def simulate_portfolio_losses(mu_daily, cov_daily, weights, n_sims: int = cfg.MC_SIMS,
                              seed: int = cfg.MC_SEED) -> np.ndarray:
    """Monte Carlo core: draw asset returns ~ N(mu, Sigma) (Cholesky), return portfolio losses.
    Fixed seed => reproducible; the same seed gives the same random numbers, so two runs with
    different Sigma (e.g. a stress scenario) differ ONLY because of Sigma."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, len(weights)))
    L = np.linalg.cholesky(np.asarray(cov_daily))
    sim_asset_returns = np.asarray(mu_daily) + z @ L.T
    return -(sim_asset_returns @ np.asarray(weights, dtype=float))


def monte_carlo_var_es(returns: pd.DataFrame, weights: pd.Series, alpha: float = 0.95,
                       n_sims: int = cfg.MC_SIMS, seed: int = cfg.MC_SEED):
    """Monte Carlo VaR / ES:
       1. estimate mean vector and covariance matrix from historical returns
       2. simulate n_sims correlated asset-return scenarios
       3. turn them into a distribution of portfolio losses
       4. VaR = alpha-quantile of simulated losses, ES = average of the tail beyond it."""
    w = weights.reindex(returns.columns).values
    losses = simulate_portfolio_losses(returns.mean().values, returns.cov().values, w, n_sims, seed)
    return value_at_risk(losses, alpha), expected_shortfall(losses, alpha)


# ------------------------------------------------------------------- drawdown
def drawdown_series(port_returns: pd.Series) -> pd.DataFrame:
    """Wealth curve, running maximum of wealth, and drawdown = wealth / running_max - 1."""
    wealth = (1 + port_returns).cumprod()
    running_max = wealth.cummax()
    return pd.DataFrame({"Wealth": wealth, "RunningMax": running_max,
                         "Drawdown": wealth / running_max - 1.0})


def max_drawdown(port_returns: pd.Series) -> dict:
    dd = drawdown_series(port_returns)
    trough = dd["Drawdown"].idxmin()
    peak = dd.loc[:trough, "Wealth"].idxmax()
    return {"max_drawdown": float(dd["Drawdown"].min()),
            "peak_date": str(peak.date()), "trough_date": str(trough.date())}


# -------------------------------------------------------------- one-line report
def risk_summary(returns: pd.DataFrame, weights: pd.Series) -> dict:
    """Every risk metric for one portfolio (all numbers here are plain Python floats)."""
    mu, cov = estimate_inputs(returns)
    w = weights.reindex(returns.columns)
    pr = portfolio_returns(returns, w)
    ret, vol = portfolio_return(w, mu), portfolio_volatility(w, cov)
    out = {"annual_return": ret, "annual_volatility": vol, "sharpe": sharpe_ratio(ret, vol)}
    for a in cfg.CONFIDENCE_LEVELS:
        tag = int(round(a * 100))
        hv, hes = historical_var_es(pr, a)
        mv, mes = monte_carlo_var_es(returns, w, a)
        out.update({f"hist_var_{tag}": hv, f"hist_es_{tag}": hes,
                    f"mc_var_{tag}": mv, f"mc_es_{tag}": mes})
    out.update(max_drawdown(pr))
    return out
