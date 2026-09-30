"""Simple matplotlib charts (validated categorical palette, fixed order, light grid)."""
from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import risk_metrics as rm
from .pipeline import CVAR, GMV, L2, MAXSHARPE, POLICY, PRIMARY

COLORS = {POLICY: "#898781", GMV: "#2a78d6", MAXSHARPE: "#eb6834", L2: "#1baf7a", CVAR: "#e87ba4"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlesize": 11, "font.size": 9, "legend.frameon": False})


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=140, facecolor="white"); plt.close(fig)


def efficient_frontier(art, results, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    f = art["frontier"]
    ax.plot(f["Vol"], f["Return"], color=INK, lw=1.5, label="Efficient frontier (MVO)")
    for a in art["returns"].columns:
        ax.scatter(np.sqrt(art["cov"].loc[a, a]), art["mu"][a], s=14, color="#c9c8c2", zorder=2)
        ax.annotate(a, (np.sqrt(art["cov"].loc[a, a]), art["mu"][a]), fontsize=7, color=MUTED, xytext=(3, 3),
                    textcoords="offset points")
    size = {CVAR: 190, MAXSHARPE: 60}       # CVaR ~ Max Sharpe here: draw the big one first so both stay visible
    for name in sorted(results["risk"], key=lambda n: -size.get(n, 80)):
        r = results["risk"][name]
        ax.scatter(r["annual_volatility"], r["annual_return"], s=size.get(name, 80), color=COLORS[name],
                   edgecolor="white", linewidth=1.5, zorder=4, label=name)
    ax.set_xlabel("Annualised volatility"); ax.set_ylabel("Annualised expected return")
    ax.set_title("Efficient frontier and the five portfolios"); ax.legend(fontsize=8)
    _save(fig, path)


def weights_comparison(art, path):
    w = art["weights"]; x = np.arange(len(w.index)); k = len(w.columns); bw = 0.8 / k
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(w.columns):
        ax.bar(x + (i - (k - 1) / 2) * bw, w[name], bw * 0.9, color=COLORS[name], label=name)
    ax.set_xticks(x); ax.set_xticklabels(w.index, rotation=30, ha="right"); ax.set_ylabel("Weight")
    ax.set_title("Portfolio weights by method"); ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def l2_sweep(art, results, path):
    sw = art["l2_sweep_weights"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    a1.stackplot(range(len(sw)), [sw[c].values for c in sw.columns], labels=sw.columns, colors=pal,
                 edgecolor="white", linewidth=0.8)
    a1.set_xticks(range(len(sw))); a1.set_xticklabels([f"{l:g}" for l in sw.index])
    a1.set_xlabel("lambda (L2 penalty strength)"); a1.set_ylabel("Weight"); a1.set_ylim(0, 1)
    a1.set_title("Larger lambda spreads the weights"); a1.legend(fontsize=7, ncol=2, loc="lower center")
    s = results["l2_lambda_sweep"]
    a2.plot(range(len(s)), [r["effective_n"] for r in s], color="#1baf7a", marker="o", lw=2)
    a2.set_xticks(range(len(s))); a2.set_xticklabels([f"{r['lambda']:g}" for r in s])
    a2.set_xlabel("lambda"); a2.set_ylabel("Effective number of assets (1/HHI)")
    a2.set_title("Diversification rises with lambda")
    _save(fig, path)


def var_distribution(art, path, name=PRIMARY):
    losses = -art["port_returns"][name].values
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(losses * 100, bins=80, color="#9ec5f4", edgecolor="white", linewidth=0.3)
    for a, c in ((0.95, "#eb6834"), (0.99, "#e34948")):
        v, e = rm.historical_var_es(-losses, a)
        ax.axvline(v * 100, color=c, ls="--", lw=1.5, label=f"{a:.0%} VaR = {v:.2%}")
        ax.axvline(e * 100, color=c, ls=":", lw=1.5, label=f"{a:.0%} ES = {e:.2%}")
    ax.set_xlabel("Daily loss (%)  [negative = gain]"); ax.set_ylabel("Days")
    ax.set_title(f"Daily loss distribution, VaR -> tail -> Expected Shortfall  ({name})"); ax.legend(fontsize=8)
    _save(fig, path)


def drawdown(art, path):
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for name, pr in art["port_returns"].items():
        ax.plot(rm.drawdown_series(pr)["Drawdown"] * 100, color=COLORS[name], lw=1.3, label=name)
    ax.set_ylabel("Drawdown (%)"); ax.set_title("Drawdown = wealth / running maximum - 1"); ax.legend(fontsize=8)
    _save(fig, path)


def backtest(art, path, name=PRIMARY, method="historical", alpha=0.99):
    bt = art["backtest_detail"][(name, method)][alpha]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(-bt["Realized_Return"] * 100, color="#c9c8c2", lw=0.8, label="Realised daily loss")
    ax.plot(bt["VaR_Forecast"] * 100, color="#eb6834", lw=1.6, label=f"{alpha:.0%} VaR forecast")
    b = bt[bt["Breach"]]
    ax.scatter(b.index, -b["Realized_Return"] * 100, color="#e34948", s=22, zorder=5, label=f"Breach ({len(b)})")
    ax.set_ylabel("Loss (%)"); ax.set_title(f"VaR backtest: {method.replace('_', ' ')} {alpha:.0%} VaR ({name})")
    ax.legend(fontsize=8)
    _save(fig, path)


def scenario_chart(art, path):
    s = art["scenarios"]; names = list(dict.fromkeys(s["Portfolio"])); scs = list(dict.fromkeys(s["Scenario"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, n in enumerate(names):
        d = s[s.Portfolio == n]
        a1.bar(i, d[d.Scenario == "Market Crash"]["Return_Impact"].iloc[0] * 100, color=COLORS[n])
    a1.set_xticks(range(len(names))); a1.set_xticklabels([textwrap.fill(n, 12) for n in names], fontsize=7)
    a1.set_ylabel("Instant return (%)"); a1.set_title("Market Crash: portfolio return impact")
    bw = 0.8 / len(names)
    for i, n in enumerate(names):
        d = s[s.Portfolio == n].set_index("Scenario").loc[scs]
        a2.bar(np.arange(len(scs)) + (i - (len(names) - 1) / 2) * bw, d["Stressed_ES_99"] * 100, bw * 0.9,
               color=COLORS[n], label=n)
    a2.set_xticks(range(len(scs))); a2.set_xticklabels([x.replace(" ", "\n") for x in scs], fontsize=7)
    a2.set_ylabel("1-day 99% Expected Shortfall (%)"); a2.set_title("Tail risk under each scenario")
    a2.legend(fontsize=7)
    _save(fig, path)
