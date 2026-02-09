
import sys
import importlib
import numpy as np
from tests.helpers_fakes import make_fake_functions_all
from search_core import Candidate, LightCurve
from transit_fit import PeriodicFitter, FittingConfig

def test_periodic_fitter_calls_canonical(monkeypatch, lc_small):
    # Inject fake Functions_all to avoid heavy sampling
    fake = make_fake_functions_all()
    monkeypatch.setitem(sys.modules, "Functions_all", fake)
    importlib.reload(importlib.import_module("transit_fit"))

    from transit_fit import PeriodicFitter, FittingConfig

    candidates = [Candidate(period=2.0, t0=1.0, duration=0.08, depth=0.005)]
    fitter = PeriodicFitter(lc_small, ab=(0.3, 0.3), cfg=FittingConfig(verbose=True))

    result = fitter.fit(candidates)
    assert isinstance(result, tuple) and len(result) == 7
    T0_vals, periods_fit, depth_fit, Tdur_fit, SNR_vals, intransit, params_df = result
    assert np.allclose(T0_vals, [1.0])
    assert np.allclose(periods_fit, [2.0])
    assert intransit.shape == lc_small.time.shape

def test_hint_builder_ols_or_duration(monkeypatch, lc_small):
    # Ensure hint builder returns one per candidate and has required keys
    candidates = [Candidate(period=2.0, t0=1.0, duration=0.08, depth=0.005)]
    from transit_fit import PeriodicFitter, FittingConfig
    fitter = PeriodicFitter(lc_small, ab=(0.3, 0.3), cfg=FittingConfig(use_ols_hints=False))
    hints = fitter.build_prior_hints(candidates)
    assert len(hints) == 1
    h = hints[0]
    for key in ("t0", "Per", "rp_rs"):
        assert key in h and all(k in h[key] for k in ("mu", "low", "high"))
