import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfolio_risk import pipeline, simulate_data  # noqa: E402
from portfolio_risk import risk_metrics as rm  # noqa: E402


@lru_cache(maxsize=1)
def returns():
    return simulate_data.simulate_market_data()


@lru_cache(maxsize=1)
def inputs():
    return rm.estimate_inputs(returns())


@lru_cache(maxsize=1)
def pipeline_output():
    return pipeline.run_pipeline(returns())
