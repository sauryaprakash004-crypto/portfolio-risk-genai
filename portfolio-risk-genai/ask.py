#!/usr/bin/env python3
"""Ask the GenAI explainer a question about the calculated results.

    python ask.py "Explain my portfolio risk."
    python ask.py --list          # show the example questions
    export ANTHROPIC_API_KEY=...  # optional: use the real LLM (otherwise offline templates)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from portfolio_risk import genai_explainer, pipeline  # noqa: E402


def load_results():
    f = ROOT / "outputs" / "results.json"
    if f.exists():
        return json.loads(f.read_text())
    return pipeline.run_pipeline()[0]                      # not run yet: compute (takes a few seconds)


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        print("Example questions:\n  " + "\n  ".join(genai_explainer.EXAMPLE_QUESTIONS))
        return
    out = genai_explainer.explain(" ".join(sys.argv[1:]), load_results())
    print(out["answer"])
    if out["note"]:
        print(f"\n[{out['note']}]")
    if out["unverified_numbers"]:
        print(f"\n[WARNING: figures not found in the calculated results: {out['unverified_numbers']}]")


if __name__ == "__main__":
    main()
