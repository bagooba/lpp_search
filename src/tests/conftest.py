
# tests/conftest.py
import numpy as np
import pytest
from search_core import LightCurve, Candidate

@pytest.fixture
def lc_small():
    # 3-day baseline, ~2-min cadence
    time = np.arange(0.0, 3.0, 0.0014)  # days
    flux = np.ones_like(time)
    # Inject a simple box transit at t0=1.0, duration=0.08 d, depth=0.005
    t0, dur, depth = 1.0, 0.08, 0.005
    in_box = np.abs(time - t0) < dur
    flux[in_box] -= depth
    flux_err = np.full_like(time, 1e-4)
    return LightCurve(time=time, flux=flux, flux_err=flux_err)

@pytest.fixture
def candidate_basic():
    # Candidate consistent with the injected transit above
    return Candidate(period=2.0, t0=1.0, duration=0.08, depth=0.005)

@pytest.fixture
def rng():
    return np.random.default_rng(12345)
