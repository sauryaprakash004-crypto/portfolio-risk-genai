"""
Three portfolio optimisers. All share the SAME constraints so results are comparable:
    sum(w) = 1     (fully invested)
    0 <= w_i <= w_max   (long-only, no single asset above w_max)

1. Mean-Variance (Markowitz):   min  w' Sigma w   (GMV)   |   max  (w'mu - rf) / sqrt(w' Sigma w)  (Max Sharpe)
2. L2-regularised MVO:          min  gamma * w' Sigma w  -  w' mu  +  lambda * ||w||^2
3. CVaR optimisation:           min  CVaR_alpha(portfolio losses)   s.t.  w'mu >= target return
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix, hstack, identity, vstack

from . import config as cfg
from .risk_metrics import portfolio_volatility


def _slsqp(objective, n: int, max_weight: float, extra_constraints=()):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *extra_constraints]
    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP", bounds=[(0.0, max_weight)] * n,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Optimiser failed: {res.message}")
    return np.clip(res.x, 0.0, max_weight)


# ------------------------------------------------------------ 1. mean-variance
def global_min_variance(mu: pd.Series, cov: pd.DataFrame, max_weight: float = cfg.MAX_WEIGHT) -> pd.Series:
    """Lowest-volatility portfolio: min w' Sigma w."""
    S = cov.values
    w = _slsqp(lambda w: w @ S @ w, len(mu), max_weight)
    return pd.Series(w, index=mu.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = cfg.RISK_FREE_RATE,
               max_weight: float = cfg.MAX_WEIGHT) -> pd.Series:
    """Best return per unit of risk: max (w'mu - rf) / sqrt(w' Sigma w)."""
    m, S = mu.values, cov.values
    w = _slsqp(lambda w: -(w @ m - rf) / np.sqrt(w @ S @ w), len(mu), max_weight)
    return pd.Series(w, index=mu.index)


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 30,
                       max_weight: float = cfg.MAX_WEIGHT) -> pd.DataFrame:
    """For each target return between the GMV return and the highest achievable return,
    find the minimum-variance portfolio. Returns (Return, Vol) pairs."""
    m, S, n = mu.values, cov.values, len(mu)
    lo = float(global_min_variance(mu, cov, max_weight) @ mu)
    hi, left = 0.0, 1.0                        # highest return reachable under the weight cap
    for i in np.argsort(-m):
        take = min(max_weight, left)
        hi, left = hi + take * m[i], left - take
    rows = []
    for target in np.linspace(lo, hi * 0.999, n_points):
        w = _slsqp(lambda w: w @ S @ w, n, max_weight,
                   [{"type": "eq", "fun": lambda w, t=target: w @ m - t}])
        rows.append({"Return": float(w @ m), "Vol": portfolio_volatility(w, S)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- 2. L2-regularised
def l2_regularized_mvo(mu: pd.Series, cov: pd.DataFrame, lam: float = cfg.L2_LAMBDA,
                       risk_aversion: float = cfg.RISK_AVERSION,
                       max_weight: float = cfg.MAX_WEIGHT) -> pd.Series:
    """min  gamma * w' Sigma w  -  w' mu  +  lam * sum(w_i^2)
    lam = 0 is plain mean-variance utility; larger lam pushes weights toward equal (1/N)."""
    m, S = mu.values, cov.values
    w = _slsqp(lambda w: risk_aversion * (w @ S @ w) - w @ m + lam * (w @ w), len(mu), max_weight)
    return pd.Series(w, index=mu.index)


# ---------------------------------------------------------------- 3. CVaR
def cvar_optimization(returns: pd.DataFrame, alpha: float = cfg.CVAR_ALPHA,
                      target_return: float | None = None,
                      max_weight: float = cfg.MAX_WEIGHT) -> pd.Series:
    """Minimise CVaR (expected shortfall) of the historical portfolio losses.

    Rockafellar-Uryasev (2000): CVaR_alpha(w) = min_z  z + 1/((1-alpha) T) * sum_t max(0, loss_t(w) - z)
    which becomes a linear program with extra variables z (~VaR) and u_t >= 0:
        minimise    z + 1/((1-alpha) T) * sum u_t
        subject to  u_t >= -r_t' w - z,   u_t >= 0,   sum w = 1,   0 <= w <= w_max,
                    (optional)  mu' w >= target_return
    Because it is a linear program it is solved exactly (no local minima).
    """
    R = returns.values
    T, n = R.shape
    c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / ((1 - alpha) * T))])
    # -R w - z - u <= 0
    A_ub = hstack([csr_matrix(-R), csr_matrix(-np.ones((T, 1))), -identity(T, format="csr")], format="csr")
    b_ub = np.zeros(T)
    if target_return is not None:
        row = np.concatenate([-returns.mean().values * cfg.TRADING_DAYS, np.zeros(1 + T)])
        A_ub = vstack([A_ub, csr_matrix(row[None, :])], format="csr")
        b_ub = np.append(b_ub, -target_return)
    A_eq = csr_matrix(np.concatenate([np.ones(n), np.zeros(1 + T)])[None, :])
    bounds = [(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CVaR optimisation failed: {res.message}")
    w = np.clip(res.x[:n], 0.0, max_weight)
    return pd.Series(w / w.sum(), index=returns.columns)


# ------------------------------------------------------------------- helpers
def concentration(weights: pd.Series) -> dict:
    """HHI = sum w_i^2 (1/N = perfectly spread, 1 = all in one asset); effective N = 1/HHI."""
    hhi = float((weights ** 2).sum())
    return {"hhi": hhi, "effective_n": 1.0 / hhi, "max_weight": float(weights.max())}
