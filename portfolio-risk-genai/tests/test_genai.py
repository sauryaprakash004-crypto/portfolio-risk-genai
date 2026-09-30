import json
from pathlib import Path
from conftest import pipeline_output
from portfolio_risk import genai_explainer as g

ROOT = Path(__file__).resolve().parent.parent


class FakeClient:
    """Stands in for anthropic.Anthropic(): records the request, returns a canned answer."""
    def __init__(self, text):
        self.text, self.calls = text, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        block = type("B", (), {"type": "text", "text": self.text})()
        return type("M", (), {"content": [block]})()


def test_context_contains_results_and_no_nan():
    res, _ = pipeline_output()
    ctx = g.build_context(res)
    txt = json.dumps(ctx)
    assert "NaN" not in txt and "Infinity" not in txt
    assert set(ctx) >= {"assumptions", "portfolios", "l2_lambda_sweep", "backtest", "scenarios"}
    assert ctx["portfolios"]["Max Sharpe (MVO)"]["hist_es_99_pct"] > ctx["portfolios"]["Max Sharpe (MVO)"]["hist_var_99_pct"]


def test_llm_receives_results_and_rules_and_answer_is_verified():
    res, _ = pipeline_output()
    ctx = g.build_context(res)
    v = ctx["portfolios"]["Max Sharpe (MVO)"]["hist_var_99_pct"]
    fake = FakeClient(f"The 99% VaR is {v}% of portfolio value.")
    out = g.explain("Explain my portfolio risk.", res, client=fake)
    call = fake.calls[0]
    assert out["source"] == "llm" and out["unverified_numbers"] == []
    assert "ONLY numbers" in call["system"] and "<calculated_results>" in call["messages"][0]["content"]
    assert "tools" not in call                            # no tool calling: plain prompt -> answer


def test_invented_numbers_are_flagged():
    res, _ = pipeline_output()
    out = g.explain("q", res, client=FakeClient("Your 99% VaR is 7.77% and ES is 12.34%."))
    assert "7.77%" in out["unverified_numbers"] and "12.34%" in out["unverified_numbers"]


def test_rounded_real_numbers_are_accepted():
    res, _ = pipeline_output()
    ctx = g.build_context(res)
    v = ctx["portfolios"]["Max Sharpe (MVO)"]["annual_volatility_pct"]
    assert g.check_numbers(f"volatility is about {round(v, 1)}% ({v}%)", ctx) == []


def test_offline_answers_every_example_question_with_real_numbers():
    res, _ = pipeline_output()
    ctx = g.build_context(res)
    for q in g.EXAMPLE_QUESTIONS:
        out = g.explain(q, res)                           # no API key in tests -> offline
        assert out["source"] == "offline_template" and len(out["answer"]) > 100
    ms = ctx["portfolios"]["Max Sharpe (MVO)"]
    assert str(ms["hist_es_99_pct"]) in g.explain("What does my Expected Shortfall mean?", res)["answer"]


def test_llm_failure_falls_back_gracefully():
    res, _ = pipeline_output()
    class Boom:
        messages = None
        def __init__(self): self.messages = self
        def create(self, **kw): raise RuntimeError("network down")
    out = g.explain("Explain my portfolio risk.", res, client=Boom())
    assert out["source"] == "offline_template" and "failed" in out["note"]


def test_project_has_no_agentic_components():
    banned = ["agents", "rag", "tools", "ui"]
    for d in banned:
        assert not (ROOT / d).exists(), f"{d}/ should not exist"
    src = "\n".join(p.read_text() for p in (ROOT / "src").rglob("*.py")) + (ROOT / "ask.py").read_text()
    for word in ("langgraph", "langchain", "tool_use", "tools=", "faiss", "chromadb", "vector database", "vectorstore"):
        assert word not in src.lower(), word
