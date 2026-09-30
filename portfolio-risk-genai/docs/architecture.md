# Architecture

```text
simulate_data.py ──► returns (DataFrame)
                         │
      ┌──────────────────┼───────────────────────────┐
      ▼                  ▼                           ▼
optimization.py     risk_metrics.py            scenarios.py
 MVO / L2 / CVaR    vol, Sharpe, VaR, ES,      crash / vol / corr / combined
                    Monte Carlo, drawdown
      └────────► pipeline.py ◄──── backtesting.py (rolling VaR, Kupiec)
                     │
                     ▼
           results dict → outputs/results.json     (100% deterministic Python)
                     │
                     ▼
        genai_explainer.build_context()  → compact JSON, percent units
                     │
                     ▼
        ONE LLM request: system rules + results + question
                     │
                     ▼
        answer  →  check_numbers() flags any untraceable figure
```

Design rules
- `config.py` is the single place for assumptions.
- Only `genai_explainer.py` touches an LLM, and it never calculates: it receives finished numbers.
- No loops, planners, tool calls, retrieval or multiple agents anywhere.
- Same seed ⇒ same results (data, Monte Carlo, backtests).
