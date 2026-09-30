"""All project assumptions in ONE place (easy to explain, easy to change)."""
import pandas as pd

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02          # annual, used in the Sharpe ratio
PORTFOLIO_VALUE = 1_000_000    # illustrative $ portfolio, only used to show $ losses
CONFIDENCE_LEVELS = (0.95, 0.99)

# --- optimisation (identical constraints for all three methods) ---------------
MAX_WEIGHT = 0.40              # long-only, fully invested, no asset above 40%
RISK_AVERSION = 5.0            # gamma in the L2 objective
L2_LAMBDA = 0.20               # lambda: strength of the L2 penalty
CVAR_ALPHA = 0.95              # tail level used by the CVaR optimiser

# --- Monte Carlo VaR ----------------------------------------------------------
MC_SIMS = 100_000
MC_SEED = 42

# --- backtesting --------------------------------------------------------------
BACKTEST_WINDOW = 250          # trading days of history used for each VaR forecast

# Reference "policy" portfolio (what an investor might hold today, un-optimised)
POLICY_WEIGHTS = pd.Series({
    "US_Equity": 0.30, "Intl_Equity": 0.15, "EM_Equity": 0.10, "US_Bonds": 0.20,
    "HY_Bonds": 0.05, "Gold": 0.10, "REIT": 0.05, "Commodities": 0.05,
})

# --- scenarios ----------------------------------------------------------------
# Instant price shock used by "Market Crash" and "Combined Stress"
CRASH_SHOCK = pd.Series({
    "US_Equity": -0.30, "Intl_Equity": -0.28, "EM_Equity": -0.35, "US_Bonds": 0.04,
    "HY_Bonds": -0.15, "Gold": 0.08, "REIT": -0.32, "Commodities": -0.18,
})
VOL_MULTIPLIER = 2.0           # volatility shock: every asset's volatility x 2
CORR_TARGET = 0.90             # correlation shock: correlations move toward 0.90 ...
CORR_BLEND = 0.50              # ... by this fraction (0 = no change, 1 = all pairs = 0.90)
