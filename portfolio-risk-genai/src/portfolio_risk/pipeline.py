"""
Quantitative pipeline: data -> optimisation -> risk -> backtest -> scenarios.
Everything here is deterministic Python (fixed seeds). Its output (`results`) is the ONLY thing the
GenAI layer ever sees.
"""
from __future__ import annotations

import pandas as pd

from . import backtesting, optimization as opt, risk_metrics as rm, scenarios, simulate_data
from . import config as cfg

POLICY, GMV, MAXSHARPE, L2, CVAR = ("Policy (reference)", "Min Variance (MVO)", "Max Sharpe (MVO)",
                                    "L2-Regularized MVO", "CVaR-Optimized")
PRIMARY = MAXSHARPE
LAMBDA_GRID = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def run_pipeline(returns: pd.DataFrame | None = None):
    """Returns (results, artifacts).
    results   : plain dict of numbers (JSON-safe, raw fractions) -> saved to outputs/results.json
    artifacts : DataFrames / Series used for tables and charts."""
    if returns is None:
        returns = simulate_data.simulate_market_data()
    mu, cov = rm.estimate_inputs(returns)

    # ---- 1. optimisation ---------------------------------------------------
    w_gmv = opt.global_min_variance(mu, cov)
    w_ms = opt.max_sharpe(mu, cov)
    w_l2 = opt.l2_regularized_mvo(mu, cov)
    cvar_target = rm.portfolio_return(w_ms, mu)          # same expected return as Max Sharpe
    w_cvar = opt.cvar_optimization(returns, target_return=cvar_target)
    weights = pd.DataFrame({POLICY: cfg.POLICY_WEIGHTS.reindex(returns.columns), GMV: w_gmv,
                            MAXSHARPE: w_ms, L2: w_l2, CVAR: w_cvar})

    frontier = opt.efficient_frontier(mu, cov)
    sweep_w = pd.DataFrame({lam: opt.l2_regularized_mvo(mu, cov, lam=lam) for lam in LAMBDA_GRID}).T
    sweep_w.index.name = "lambda"
    sweep_rows = []
    for lam, w in sweep_w.iterrows():
        r_, v_ = rm.portfolio_return(w, mu), rm.portfolio_volatility(w, cov)
        sweep_rows.append({"lambda": lam, "annual_return": r_, "annual_volatility": v_,
                           "sharpe": rm.sharpe_ratio(r_, v_), **opt.concentration(w)})

    # ---- 2. risk, 3. backtest, 4. scenarios (all portfolios) ------------------
    risk = {name: rm.risk_summary(returns, weights[name]) for name in weights}
    for name in weights:
        risk[name].update(opt.concentration(weights[name]))
    port_ret = {name: rm.portfolio_returns(returns, weights[name]) for name in weights}

    bt_frames, bt_detail = [], {}
    for name in weights:
        for method in ("historical", "monte_carlo"):
            df, detail = backtesting.backtest_summary(port_ret[name], method=method)
            bt_frames.append(df.assign(portfolio=name))
            bt_detail[(name, method)] = detail
    backtest = pd.concat(bt_frames, ignore_index=True)
    scen = scenarios.run_scenarios_all(returns, {n: weights[n] for n in weights})

    results = {
        "meta": {
            "n_days": len(returns), "start": str(returns.index[0].date()), "end": str(returns.index[-1].date()),
            "assets": list(returns.columns), "risk_free_rate": cfg.RISK_FREE_RATE,
            "max_weight": cfg.MAX_WEIGHT, "l2_lambda": cfg.L2_LAMBDA, "risk_aversion": cfg.RISK_AVERSION,
            "cvar_alpha": cfg.CVAR_ALPHA, "cvar_min_return_constraint": cvar_target,
            "mc_sims": cfg.MC_SIMS, "mc_seed": cfg.MC_SEED, "backtest_window": cfg.BACKTEST_WINDOW,
            "portfolio_value": cfg.PORTFOLIO_VALUE, "primary_portfolio": PRIMARY,
        },
        "weights": {n: weights[n].to_dict() for n in weights},
        "risk": risk,
        "l2_lambda_sweep": sweep_rows,
        "backtest": backtest.to_dict(orient="records"),
        "scenarios": scen.to_dict(orient="records"),
        "scenario_definitions": {
            "Market Crash": {k: float(v) for k, v in cfg.CRASH_SHOCK.items()},
            "Volatility Shock": f"all volatilities x {cfg.VOL_MULTIPLIER}",
            "Correlation Shock": f"correlations moved {cfg.CORR_BLEND:.0%} of the way toward {cfg.CORR_TARGET}",
            "Combined Stress": "Market Crash price shock + Volatility Shock + Correlation Shock",
        },
    }
    artifacts = {"returns": returns, "mu": mu, "cov": cov, "weights": weights, "frontier": frontier,
                 "l2_sweep_weights": sweep_w, "port_returns": port_ret, "backtest": backtest,
                 "backtest_detail": bt_detail, "scenarios": scen}
    return results, artifacts
