import numpy as np
import pandas as pd
from conftest import inputs, returns
from portfolio_risk import config as cfg, optimization as opt, risk_metrics as rm


def _check_constraints(w):
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= -1e-9).all() and (w <= cfg.MAX_WEIGHT + 1e-9).all()


def test_all_methods_respect_constraints():
    mu, cov = inputs()
    ms = opt.max_sharpe(mu, cov)
    for w in (opt.global_min_variance(mu, cov), ms, opt.l2_regularized_mvo(mu, cov),
              opt.cvar_optimization(returns(), target_return=rm.portfolio_return(ms, mu))):
        _check_constraints(w)


def test_gmv_has_lowest_volatility_and_max_sharpe_highest_sharpe():
    mu, cov = inputs()
    gmv, ms = opt.global_min_variance(mu, cov), opt.max_sharpe(mu, cov)
    assert rm.portfolio_volatility(gmv, cov) <= rm.portfolio_volatility(ms, cov)
    sharpe = lambda w: rm.sharpe_ratio(rm.portfolio_return(w, mu), rm.portfolio_volatility(w, cov))
    assert sharpe(ms) >= sharpe(gmv)
    rng = np.random.default_rng(0)                      # no random feasible portfolio beats the optimum
    for _ in range(3000):
        w = pd.Series(rng.dirichlet(np.ones(len(mu))), index=mu.index)
        if w.max() <= cfg.MAX_WEIGHT:
            assert sharpe(w) <= sharpe(ms) + 1e-6
            assert rm.portfolio_volatility(w, cov) >= rm.portfolio_volatility(gmv, cov) - 1e-6


def test_frontier_is_increasing_and_above_gmv():
    mu, cov = inputs()
    f = opt.efficient_frontier(mu, cov, n_points=10)
    assert (np.diff(f["Vol"]) > 0).all() and (np.diff(f["Return"]) > 0).all()


def test_l2_penalty_spreads_weights_monotonically():
    mu, cov = inputs()
    eff = [opt.concentration(opt.l2_regularized_mvo(mu, cov, lam=l))["effective_n"] for l in (0, 0.05, 0.2, 1.0)]
    assert all(a < b for a, b in zip(eff, eff[1:]))


def test_l2_lambda_zero_is_plain_mean_variance_utility():
    mu, cov = inputs()
    w = opt.l2_regularized_mvo(mu, cov, lam=0.0)
    u = lambda x: cfg.RISK_AVERSION * (x @ cov.values @ x) - x @ mu.values
    rng = np.random.default_rng(1)
    for _ in range(2000):
        r = rng.dirichlet(np.ones(len(mu)))
        if r.max() <= cfg.MAX_WEIGHT:
            assert u(w.values) <= u(r) + 1e-8


def test_cvar_optimum_beats_random_portfolios_and_meets_return_target():
    mu, cov = inputs(); R = returns()
    target = rm.portfolio_return(opt.max_sharpe(mu, cov), mu)
    w = opt.cvar_optimization(R, target_return=target)
    assert rm.portfolio_return(w, mu) >= target - 1e-6
    best = rm.expected_shortfall(-(R.values @ w.values), cfg.CVAR_ALPHA)
    rng = np.random.default_rng(2)
    for _ in range(3000):
        r = rng.dirichlet(np.ones(len(mu)))
        if r.max() <= cfg.MAX_WEIGHT and r @ mu.values >= target:
            assert best <= rm.expected_shortfall(-(R.values @ r), cfg.CVAR_ALPHA) + 1e-9


def test_cvar_optimum_is_no_worse_than_max_sharpe_in_tail():
    mu, cov = inputs(); R = returns()
    ms = opt.max_sharpe(mu, cov)
    cv = opt.cvar_optimization(R, target_return=rm.portfolio_return(ms, mu))
    es = lambda w: rm.expected_shortfall(-(R.values @ w.values), cfg.CVAR_ALPHA)
    assert es(cv) <= es(ms) + 1e-9
