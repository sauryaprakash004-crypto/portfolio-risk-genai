"""Tiny stand-in for `pytest tests/` (for machines without pytest): python tests/run_tests.py"""
import importlib, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
failed = 0
for f in sorted(Path(__file__).parent.glob("test_*.py")):
    mod = importlib.import_module(f.stem)
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        try:
            getattr(mod, name)(); print(f"PASS {f.stem}::{name}")
        except Exception:
            failed += 1; print(f"FAIL {f.stem}::{name}"); traceback.print_exc()
print("\nfailures:", failed); sys.exit(1 if failed else 0)
