"""
GenAI explanation layer.

    calculated results (Python)  ->  build_context()  ->  ONE prompt to the LLM  ->  plain-English answer

This is NOT an agent: there is no loop, no planner, no tool calling, no retrieval. The model gets the
already-calculated results as text and explains them. It never calculates anything.

"The quantitative calculations are performed deterministically using Python. Generative AI is used
as an interpretation and natural-language explanation layer."

Safeguards against invented numbers:
  1. The system prompt forbids using any figure that is not in the supplied results.
  2. `check_numbers()` scans the answer and flags every figure that cannot be traced to the results.
Without an API key, `offline_explain()` answers from fixed templates filled with the same results
(useful for demos/tests; it is plain string formatting, not GenAI).
"""
from __future__ import annotations

import json
import os
import re

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")

EXAMPLE_QUESTIONS = [
    "Explain my portfolio risk.",
    "Why is the maximum Sharpe portfolio different from the minimum variance portfolio?",
    "Explain the Monte Carlo VaR.",
    "What does my Expected Shortfall mean?",
    "Compare MVO and CVaR optimization.",
    "Why did the L2 portfolio have different weights?",
    "Which portfolio has lower tail risk?",
    "Explain the VaR backtesting result.",
    "What happens under the market crash scenario?",
    "Explain the difference between VaR and CVaR.",
]

SYSTEM_PROMPT = """You are a portfolio-risk explainer inside a quantitative-finance project.
A Python engine has ALREADY calculated every number (optimisation, VaR, Expected Shortfall, backtests,
scenarios). You receive those results inside <calculated_results>. Your job is to explain them.

Rules:
1. Use ONLY numbers that appear in <calculated_results>. Never estimate, recompute, extrapolate or invent
   a number. If the answer needs a number that is not there, say it was not calculated.
2. Quote numbers exactly as given (same units: values ending in _pct are percentages).
3. Explain in plain English for someone new to quantitative finance; define a term the first time you use it.
4. Refer to portfolios and scenarios by the names used in the results.
5. VaR/ES are 1-day losses as a % of portfolio value. Results are in-sample on simulated data: say so when
   it matters for the conclusion. Do not give investment advice.
6. Be concise: about 120-250 words, short paragraphs, at most one short list."""


# ------------------------------------------------------------------ context
def _pct(x):
    return round(100 * x, 2)


def build_context(results: dict) -> dict:
    """Turn raw results (fractions) into a compact, LLM-friendly dict (percent units, 2 decimals)."""
    m = results["meta"]
    ctx = {
        "note": "Simulated daily data. All *_pct values are percentages. VaR/ES are 1-day losses as % of "
                "portfolio value. Optimisation and risk numbers are in-sample.",
        "assumptions": {
            "period": f"{m['start']} to {m['end']} ({m['n_days']} trading days)",
            "risk_free_rate_pct": _pct(m["risk_free_rate"]), "max_weight_per_asset_pct": _pct(m["max_weight"]),
            "l2_lambda": m["l2_lambda"], "l2_risk_aversion": m["risk_aversion"],
            "cvar_confidence_pct": _pct(m["cvar_alpha"]),
            "cvar_min_expected_return_pct": _pct(m["cvar_min_return_constraint"]),
            "monte_carlo_simulations": m["mc_sims"], "monte_carlo_seed": m["mc_seed"],
            "backtest_window_days": m["backtest_window"], "portfolio_value": m["portfolio_value"],
            "monte_carlo_distribution": "multivariate normal (mean/covariance estimated from history)",
        },
        "portfolios": {}, "l2_lambda_sweep": [], "backtest": [], "scenario_definitions": results["scenario_definitions"],
        "scenarios": {},
    }
    for name, r in results["risk"].items():
        p = {"weights_pct": {a: _pct(w) for a, w in results["weights"][name].items() if w >= 0.0005},
             "annual_return_pct": _pct(r["annual_return"]), "annual_volatility_pct": _pct(r["annual_volatility"]),
             "sharpe": round(r["sharpe"], 2), "max_drawdown_pct": _pct(r["max_drawdown"]),
             "max_drawdown_peak": r["peak_date"], "max_drawdown_trough": r["trough_date"],
             "effective_number_of_assets": round(r["effective_n"], 2), "largest_weight_pct": _pct(r["max_weight"])}
        for tag in (95, 99):
            for k in ("hist_var", "hist_es", "mc_var", "mc_es"):
                p[f"{k}_{tag}_pct"] = _pct(r[f"{k}_{tag}"])
        ctx["portfolios"][name] = p
    for r in results["l2_lambda_sweep"]:
        ctx["l2_lambda_sweep"].append({"lambda": r["lambda"], "annual_return_pct": _pct(r["annual_return"]),
                                       "annual_volatility_pct": _pct(r["annual_volatility"]),
                                       "sharpe": round(r["sharpe"], 2), "largest_weight_pct": _pct(r["max_weight"]),
                                       "effective_number_of_assets": round(r["effective_n"], 2)})
    for r in results["backtest"]:
        ctx["backtest"].append({
            "portfolio": r["portfolio"], "var_method": r["method"], "confidence_pct": _pct(r["confidence"]),
            "days_tested": r["days_tested"], "breaches": r["breaches"], "breach_rate_pct": _pct(r["breach_rate"]),
            "expected_breach_rate_pct": _pct(r["expected_breach_rate"]),
            "expected_breaches": round(r["expected_breaches"], 1),
            "kupiec_p_value": round(r["kupiec_p_value"], 4), "model_rejected_at_5pct": bool(r["model_rejected_5pct"]),
            "interpretation": r["interpretation"]})
    for r in results["scenarios"]:
        ctx["scenarios"].setdefault(r["Portfolio"], {})[r["Scenario"]] = {
            "return_impact_pct": _pct(r["Return_Impact"]), "loss_amount": round(r["Loss_Amount"]),
            "stressed_volatility_pct": _pct(r["Stressed_Vol"]), "stressed_var_99_pct": _pct(r["Stressed_VaR_99"]),
            "stressed_es_99_pct": _pct(r["Stressed_ES_99"]),
            "change_in_volatility_pct_points": _pct(r["Change_Vol"]),
            "change_in_var_99_pct_points": _pct(r["Change_VaR_99"]),
            "change_in_es_99_pct_points": _pct(r["Change_ES_99"])}
    return ctx


# --------------------------------------------------------- number safeguard
_NUM = re.compile(r"(?<![\w.])-?\$?\d[\d,]*(?:\.\d+)?%?")


def _leaves(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _leaves(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _leaves(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(abs(float(obj)))
    return out


def check_numbers(answer: str, ctx: dict) -> list:
    """Return the figures in `answer` that cannot be traced to `ctx` (empty list = all traceable).
    A figure matches if some result value rounds to it at the precision the answer used."""
    leaves = _leaves(ctx, [])
    ctx_text = json.dumps(ctx)
    bad = []
    for tok in _NUM.findall(answer):
        clean = tok.replace("$", "").replace(",", "").rstrip("%").lstrip("-")
        if not clean:
            continue
        has_decimal, is_pct = "." in clean, tok.endswith("%")
        if not (has_decimal or is_pct or len(clean) >= 3):
            continue                                     # ignore small counting numbers like "two", "1", "5"
        if clean in ctx_text:
            continue
        d = len(clean.split(".")[1]) if has_decimal else 0
        val = float(clean)
        if any(abs(x - val) <= 0.5 * 10 ** (-d) + 1e-9 for x in leaves):
            continue
        bad.append(tok)
    return sorted(set(bad))


# --------------------------------------------------------------- LLM call
def _call_llm(question: str, ctx: dict, model: str, client=None) -> str:
    if client is None:
        import anthropic                                   # imported lazily: only needed for real LLM use
        client = anthropic.Anthropic()
    user = (f"<calculated_results>\n{json.dumps(ctx, indent=1)}\n</calculated_results>\n\n"
            f"Question: {question}")
    msg = client.messages.create(model=model, max_tokens=900, system=SYSTEM_PROMPT,
                                 messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def explain(question: str, results: dict, client=None, model: str = DEFAULT_MODEL) -> dict:
    """Answer a question about the calculated results.
    Returns {"answer", "source" ('llm' | 'offline_template'), "unverified_numbers": [...], "note"}."""
    ctx = build_context(results)
    if client is not None or os.environ.get("ANTHROPIC_API_KEY"):
        try:
            answer = _call_llm(question, ctx, model, client)
            return {"answer": answer, "source": "llm", "unverified_numbers": check_numbers(answer, ctx), "note": ""}
        except Exception as exc:                            # network / auth / SDK problem -> fall back
            note = f"LLM call failed ({type(exc).__name__}); showing the offline template answer instead."
    else:
        note = "No ANTHROPIC_API_KEY set: showing the offline template answer (fixed templates, not GenAI)."
    return {"answer": offline_explain(question, ctx), "source": "offline_template",
            "unverified_numbers": [], "note": note}


# ------------------------------------------------------- offline templates
def _p(ctx, name):
    return ctx["portfolios"][name]


def _has(q, *words):
    return any(w in q for w in words)


def offline_explain(question: str, ctx: dict) -> str:
    q = question.lower()
    names = list(ctx["portfolios"])
    primary = "Max Sharpe (MVO)"
    ms, gmv, l2, cv = (_p(ctx, n) for n in ("Max Sharpe (MVO)", "Min Variance (MVO)", "L2-Regularized MVO", "CVaR-Optimized"))

    def top(p, k=3):
        return ", ".join(f"{a} {w:.1f}%" for a, w in sorted(p["weights_pct"].items(), key=lambda x: -x[1])[:k])

    def eq(p):
        return sum(w for a, w in p["weights_pct"].items() if "Equity" in a)

    if _has(q, "difference") and _has(q, "var") and _has(q, "cvar", "expected shortfall", "es"):
        return (f"VaR is a threshold: the 99% 1-day historical VaR of {primary} is {ms['hist_var_99_pct']}%, meaning losses "
                f"were smaller than this on 99% of days. CVaR / Expected Shortfall looks INSIDE the worst 1% of days and "
                f"averages them: {ms['hist_es_99_pct']}%. So VaR says 'how bad is the edge of the bad zone', ES says "
                f"'how bad is it once we are in it'. ES is always at least as large as VaR, and unlike VaR it reacts to "
                f"how extreme the tail losses are.")
    if _has(q, "monte carlo"):
        return (f"Monte Carlo VaR: (1) estimate each asset's average return and the covariance matrix from history, "
                f"(2) simulate {ctx['assumptions']['monte_carlo_simulations']:,} correlated return scenarios (fixed seed "
                f"{ctx['assumptions']['monte_carlo_seed']}), (3) turn them into simulated portfolio losses, (4) read off the "
                f"quantile. For {primary} the 95% / 99% Monte Carlo VaR is {ms['mc_var_95_pct']}% / {ms['mc_var_99_pct']}%, "
                f"versus historical {ms['hist_var_95_pct']}% / {ms['hist_var_99_pct']}%. The simulated 99% Expected Shortfall is "
                f"{ms['mc_es_99_pct']}% vs {ms['hist_es_99_pct']}% historically"
                + (", lower because the simulation assumes a normal distribution while real returns have fatter tails."
                   if ms['mc_es_99_pct'] < ms['hist_es_99_pct'] else "."))
    if _has(q, "expected shortfall", "cvar") and not _has(q, "compare", "mvo"):
        return (f"Expected Shortfall (ES, also called CVaR) is the average loss on the worst days beyond the VaR threshold. "
                f"For {primary}: 95% ES = {ms['hist_es_95_pct']}% (the average loss on the worst 5% of days) and 99% ES = "
                f"{ms['hist_es_99_pct']}% (worst 1%), compared with the 99% VaR of {ms['hist_var_99_pct']}%. The gap between "
                f"VaR and ES shows how heavy the tail is. These are 1-day losses as a % of portfolio value "
                f"(${ctx['assumptions']['portfolio_value']:,} portfolio).")
    if _has(q, "compare") and _has(q, "mvo") and _has(q, "cvar"):
        return (f"MVO minimises variance (volatility) and treats gains and losses symmetrically; CVaR optimisation minimises the "
                f"average loss in the worst {100 - ctx['assumptions']['cvar_confidence_pct']:.0f}% of days, so it targets the tail directly. "
                f"Here the CVaR portfolio is forced to earn the same expected return as Max Sharpe "
                f"({ctx['assumptions']['cvar_min_expected_return_pct']}%). Result: 95% ES {cv['hist_es_95_pct']}% (CVaR-Optimized) vs "
                f"{ms['hist_es_95_pct']}% (Max Sharpe); volatility {cv['annual_volatility_pct']}% vs {ms['annual_volatility_pct']}%. "
                f"Weights: CVaR-Optimized {top(cv)}; Max Sharpe {top(ms)}. On this near-symmetric simulated data the two are similar; "
                f"they diverge more when returns are skewed or very fat-tailed.")
    if _has(q, "l2", "regulari", "lambda"):
        s0, s1 = ctx["l2_lambda_sweep"][0], ctx["l2_lambda_sweep"][-1]
        return (f"Standard MVO tends to concentrate: Max Sharpe puts up to {ms['largest_weight_pct']}% in one asset "
                f"({top(ms)}), effective number of assets {ms['effective_number_of_assets']}. The L2 version adds a penalty "
                f"lambda x (sum of squared weights) (lambda = {ctx['assumptions']['l2_lambda']}), which makes big weights expensive, "
                f"so the optimiser spreads capital: {top(l2)}, largest weight {l2['largest_weight_pct']}%, effective assets "
                f"{l2['effective_number_of_assets']}. The trade-off: volatility {l2['annual_volatility_pct']}% vs {ms['annual_volatility_pct']}%, "
                f"return {l2['annual_return_pct']}% vs {ms['annual_return_pct']}%, Sharpe {l2['sharpe']:.2f} vs {ms['sharpe']:.2f}. Across the lambda sweep, effective assets go from {s0['effective_number_of_assets']} "
                f"at lambda={s0['lambda']} to {s1['effective_number_of_assets']} at lambda={s1['lambda']}.")
    if _has(q, "sharpe") and _has(q, "minimum", "min var", "min-var", "gmv", "minimum variance"):
        return (f"They optimise different goals. Min Variance only minimises risk: volatility {gmv['annual_volatility_pct']}%, "
                f"return {gmv['annual_return_pct']}%, Sharpe {gmv['sharpe']:.2f} ({top(gmv)}). Max Sharpe maximises return per unit of risk: "
                f"return {ms['annual_return_pct']}%, volatility {ms['annual_volatility_pct']}%, Sharpe {ms['sharpe']:.2f} ({top(ms)}). "
                f"Equity share: {eq(ms):.1f}% in Max Sharpe vs {eq(gmv):.1f}% in Min Variance. Max Sharpe accepts more risk "
                f"in exchange for more return per unit of risk (Sharpe {ms['sharpe']:.2f} vs {gmv['sharpe']:.2f}).")
    if _has(q, "tail risk", "lower tail", "which portfolio"):
        rows = sorted(((n, _p(ctx, n)["hist_es_99_pct"], _p(ctx, n)["hist_es_95_pct"]) for n in names), key=lambda x: x[1])
        lines = "; ".join(f"{n}: ES99 {a}%, ES95 {b}%" for n, a, b in rows)
        return (f"Ranking by 99% Expected Shortfall (lowest = least tail risk): {lines}. {rows[0][0]} has the lowest tail risk, "
                f"but note it may also have lower expected return - see the return column in the portfolio table.")
    if _has(q, "backtest", "breach", "kupiec"):
        rows = [r for r in ctx["backtest"] if r["portfolio"] == primary]
        txt = " ".join(f"[{r['var_method']}, {r['confidence_pct']:.0f}%] {r['interpretation']} (Kupiec p-value {r['kupiec_p_value']})"
                       for r in rows)
        return ("Backtest: each day I compared the VaR forecast (using only the previous "
                f"{ctx['assumptions']['backtest_window_days']} days) with the loss that actually occurred, and counted breaches. "
                f"For {primary}: {txt}")
    if _has(q, "crash", "scenario", "stress", "shock"):
        lines = []
        for n in names:
            sc = ctx["scenarios"][n]
            lines.append(f"{n}: crash {sc['Market Crash']['return_impact_pct']}% (loss ${sc['Market Crash']['loss_amount']:,}), "
                         f"combined-stress ES99 {sc['Combined Stress']['stressed_es_99_pct']}%")
        return ("Market Crash is an instant price shock (equities about -30%, bonds and gold up). Portfolio impacts: "
                + "; ".join(lines) + ". Volatility and Correlation shocks do not move prices; they raise risk, e.g. for "
                f"{primary} the 99% ES rises from {ctx['scenarios'][primary]['Baseline']['stressed_es_99_pct']}% to "
                f"{ctx['scenarios'][primary]['Volatility Shock']['stressed_es_99_pct']}% under the Volatility Shock.")
    # default: overall risk explanation
    return (f"{primary}: expected return {ms['annual_return_pct']}%, volatility {ms['annual_volatility_pct']}%, Sharpe {ms['sharpe']:.2f}. "
            f"On a typical bad day (95% VaR) it loses at most {ms['hist_var_95_pct']}%; on the worst 5% of days it loses "
            f"{ms['hist_es_95_pct']}% on average (Expected Shortfall). At 99% these are {ms['hist_var_99_pct']}% and {ms['hist_es_99_pct']}%. "
            f"Its worst peak-to-trough fall (maximum drawdown) was {ms['max_drawdown_pct']}%, between {ms['max_drawdown_peak']} and "
            f"{ms['max_drawdown_trough']}. Ask about a specific topic (VaR backtest, scenarios, L2, CVaR) for more detail.")
