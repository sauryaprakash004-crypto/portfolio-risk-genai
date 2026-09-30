#!/usr/bin/env python3
"""Run the whole quantitative pipeline and write tables, charts and results.json to ./outputs.

    python run_analysis.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from portfolio_risk import pipeline, plots  # noqa: E402

OUT, TABLES, FIGS = ROOT / "outputs", ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    results, art = pipeline.run_pipeline()
    pd.set_option("display.width", 200, "display.max_columns", 30)

    # ---- tables ----
    art["weights"].round(4).to_csv(TABLES / "01_portfolio_weights.csv")
    risk = pd.DataFrame(results["risk"]).T
    risk.to_csv(TABLES / "02_risk_metrics.csv")
    art["frontier"].to_csv(TABLES / "03_efficient_frontier.csv", index=False)
    pd.DataFrame(results["l2_lambda_sweep"]).to_csv(TABLES / "04_l2_lambda_sweep.csv", index=False)
    art["backtest"].to_csv(TABLES / "05_var_backtest.csv", index=False)
    art["scenarios"].to_csv(TABLES / "06_scenarios.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # ---- charts ----
    plots.efficient_frontier(art, results, FIGS / "01_efficient_frontier.png")
    plots.weights_comparison(art, FIGS / "02_weights_comparison.png")
    plots.l2_sweep(art, results, FIGS / "03_l2_lambda_sweep.png")
    plots.var_distribution(art, FIGS / "04_var_es_distribution.png")
    plots.drawdown(art, FIGS / "05_drawdown.png")
    plots.backtest(art, FIGS / "06_var_backtest.png")
    plots.scenario_chart(art, FIGS / "07_scenarios.png")

    # ---- console summary ----
    print(f"Data: {results['meta']['n_days']} days, {results['meta']['start']} -> {results['meta']['end']}\n")
    print("WEIGHTS\n", art["weights"].round(3).to_string(), "\n")
    cols = ["annual_return", "annual_volatility", "sharpe", "hist_var_95", "hist_es_95", "mc_var_95", "hist_var_99",
            "hist_es_99", "mc_var_99", "max_drawdown", "effective_n"]
    print("RISK\n", risk[cols].astype(float).round(4).to_string(), "\n")
    print("VaR BACKTEST\n", art["backtest"].drop(columns=["interpretation"]).round(4).to_string(), "\n")
    print("SCENARIOS (Max Sharpe)\n",
          art["scenarios"][art["scenarios"].Portfolio == pipeline.PRIMARY].drop(columns="Portfolio").round(4).to_string())
    print(f"\nWrote {OUT}/results.json, tables/ and figures/. Next: python ask.py \"Explain my portfolio risk.\"")


if __name__ == "__main__":
    main()
