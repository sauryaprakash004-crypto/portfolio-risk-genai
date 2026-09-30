"""
Market data.

`simulate_market_data()` generates 6 years of daily returns for an 8-asset
multi-asset universe (fat-tailed multivariate Student-t, fixed seed, fixed dates
=> fully reproducible). One short "stress regime" (high volatility, falling
equities, correlations jumping toward 1) is placed in the sample so that VaR
backtesting and drawdown analysis have a real tail event to detect.

To use real data, replace it with `load_returns_csv("prices_or_returns.csv")`.
Everything downstream only needs a DataFrame of daily returns (rows = days,
columns = assets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSET_NAMES = ["US_Equity", "Intl_Equity", "EM_Equity", "US_Bonds",
               "HY_Bonds", "Gold", "REIT", "Commodities"]

ANNUAL_RETURN = {"US_Equity": 0.10, "Intl_Equity": 0.07, "EM_Equity": 0.08, "US_Bonds": 0.03,
                 "HY_Bonds": 0.05, "Gold": 0.05, "REIT": 0.08, "Commodities": 0.04}
ANNUAL_VOL = {"US_Equity": 0.16, "Intl_Equity": 0.17, "EM_Equity": 0.22, "US_Bonds": 0.05,
              "HY_Bonds": 0.09, "Gold": 0.15, "REIT": 0.20, "Commodities": 0.18}

CORR = pd.DataFrame(
    [  # USEq  IntlEq EMEq  USBd  HYBd  Gold  REIT  Cmdty
        [1.00, 0.82, 0.72, -0.15, 0.55, 0.02, 0.65, 0.30],
        [0.82, 1.00, 0.78, -0.10, 0.58, 0.05, 0.60, 0.35],
        [0.72, 0.78, 1.00, -0.05, 0.55, 0.10, 0.55, 0.45],
        [-0.15, -0.10, -0.05, 1.00, 0.20, 0.15, 0.05, -0.05],
        [0.55, 0.58, 0.55, 0.20, 1.00, 0.10, 0.50, 0.30],
        [0.02, 0.05, 0.10, 0.15, 0.10, 1.00, 0.05, 0.35],
        [0.65, 0.60, 0.55, 0.05, 0.50, 0.05, 1.00, 0.25],
        [0.30, 0.35, 0.45, -0.05, 0.30, 0.35, 0.25, 1.00],
    ],
    index=ASSET_NAMES, columns=ASSET_NAMES,
)

TRADING_DAYS = 252


def _daily_params(annual_return, annual_vol, corr):
    vol = np.array([annual_vol[a] for a in ASSET_NAMES]) / np.sqrt(TRADING_DAYS)
    mu = np.array([annual_return[a] for a in ASSET_NAMES]) / TRADING_DAYS
    return mu, corr.values * np.outer(vol, vol)


def simulate_market_data(n_days: int = 1512, seed: int = 42, t_dof: float = 5.0,
                         stress_start: int = 1000, stress_len: int = 40,
                         end_date: str = "2025-12-31") -> pd.DataFrame:
    """Return a DataFrame of simulated daily returns (n_days x 8 assets)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n_days)

    mu_n, cov_n = _daily_params(ANNUAL_RETURN, ANNUAL_VOL, CORR)

    # stress regime: 3x volatility, negative drift on risky assets, correlations -> 0.9
    stress_ret = {k: v - 0.35 for k, v in ANNUAL_RETURN.items()}
    stress_ret["US_Bonds"] += 0.10          # flight to quality
    stress_ret["Gold"] += 0.05
    stress_vol = {k: v * 3.0 for k, v in ANNUAL_VOL.items()}
    n = len(ASSET_NAMES)
    equicorr = np.full((n, n), 0.90) + np.eye(n) * 0.10
    stress_corr = pd.DataFrame(0.15 * CORR.values + 0.85 * equicorr, index=ASSET_NAMES, columns=ASSET_NAMES)
    mu_s, cov_s = _daily_params(stress_ret, stress_vol, stress_corr)

    def draw(k, mu, cov):
        L = np.linalg.cholesky(cov)
        g = rng.standard_normal((k, n)) @ L.T
        chi2 = rng.chisquare(t_dof, size=(k, 1))
        return mu + g * np.sqrt(t_dof / chi2) * np.sqrt((t_dof - 2) / t_dof)   # multivariate t, same covariance

    r = np.vstack([draw(stress_start, mu_n, cov_n),
                   draw(stress_len, mu_s, cov_s),
                   draw(n_days - stress_start - stress_len, mu_n, cov_n)])
    return pd.DataFrame(r, index=dates, columns=ASSET_NAMES)


def load_returns_csv(path: str, is_prices: bool = True) -> pd.DataFrame:
    """Load real data: a CSV with a date index column and one column per asset."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().dropna()
    return df.pct_change().dropna() if is_prices else df
