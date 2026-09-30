# Interview guide (30-second answers)

**What is this project?** A pipeline that optimises a portfolio three ways, measures its risk, backtests VaR, stress-tests it, and uses GenAI to explain the calculated results in plain English.

**Why GenAI, and does it calculate anything?** The quantitative engine produces portfolio, risk, optimisation and scenario results, but interpreting them is hard for non-specialists. GenAI is only a natural-language explanation layer: it receives the calculated results and explains them. All maths is deterministic Python, and a number-checker flags any figure in an answer that isn't in the results.

**Why is it not agentic?** The task is "explain fixed results", which needs one prompt, not autonomy. No tool calls, planning loops or retrieval, so it is simple, cheap, testable and cannot wander.

**Mean-variance in one line.** Minimise `wᵀΣw` (or maximise `(wᵀμ − r_f)/√wᵀΣw`) with `Σw = 1`, `0 ≤ w ≤ 40%`. GMV = lowest risk; Max Sharpe = best return per unit of risk.

**Why L2 regularisation?** MVO chases noisy expected-return estimates and concentrates. Adding `λ‖w‖²` penalises big weights; larger λ → more even weights (effective N rises from 3.2 to 7.6 as λ goes 0 → 1) at the cost of some Sharpe.

**VaR vs CVaR.** VaR = loss threshold (95% of days are better than this). CVaR/ES = average loss *beyond* the threshold, so it measures how bad the tail is. ES ≥ VaR always.

**How does CVaR optimisation work?** Rockafellar–Uryasev turns "minimise average of the worst 5% of losses" into a linear program (variables: weights, threshold z, and slack u_t ≥ loss_t − z, u_t ≥ 0). I fix the expected return equal to the Max-Sharpe portfolio's so the tail comparison is like-for-like.

**Monte Carlo VaR steps.** Estimate μ and Σ → simulate correlated returns with Cholesky (fixed seed) → portfolio losses → quantile (VaR) and tail mean (ES).

**Why is Monte Carlo ES below historical ES?** It assumes normal returns; real (and simulated) returns are fat-tailed.

**Backtest.** Forecast VaR each day from the previous 250 days, count days where the realised loss exceeded it, compare the breach rate with 1−confidence, and use Kupiec's likelihood-ratio test (χ², 1 dof). Here the 99% historical VaR is breached 1.66% of days (p=0.03): it underestimates risk around the stress regime.

**Scenarios.** Three knobs (price shock, volatility multiplier, correlation shock). Crash = instant `wᵀshock`; vol/corr shocks change Σ (correlations move toward 0.9 by a blend, which keeps Σ valid) and I re-run Monte Carlo with the same seed, so any change is purely from the scenario.

**Biggest weakness?** Simulated data and in-sample optimisation. Next step would be a walk-forward out-of-sample test on real prices (intentionally out of scope here).

**CVaR and Max Sharpe look nearly identical - why?** The simulated returns are roughly elliptical, where variance and CVaR rank portfolios similarly. CVaR optimisation matters when returns are skewed / fat-tailed.
