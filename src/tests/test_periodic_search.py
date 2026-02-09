
import sys
import importlib
import numpy as np

from tests.helpers_fakes import make_fake_functions_all

def test_periodic_search_detect_and_fit(monkeypatch, lc_small):
    # Inject fake Functions_all before import
    fake = make_fake_functions_all()
    monkeypatch.setitem(sys.modules, "Functions_all", fake)

    # Now import periodic_search which will use our fake Functions_all
    periodic_search = importlib.import_module("periodic_search")
    from search_core import Candidate

    # Limb darkening
    ab = (0.3, 0.3)
    cfg = periodic_search.PeriodicSearchConfig(verbose=True, save_phasefold=False, run_chunked=False)
    ps = periodic_search.PeriodicSearch(lc=lc_small, ab=ab, cfg=cfg)

    candidates, mask_from_bls = ps.detect_full()
    assert isinstance(candidates, list)
    assert len(candidates) == 1
    assert isinstance(candidates[0], Candidate)
    assert mask_from_bls.shape == lc_small.time.shape

    masks, union = ps.build_intransit_masks(candidates, buffer=0.0)
    assert union.any()

    result = ps.fit_full(candidates, intransit_union=union)
    # Verify canonical tuple shape
    assert isinstance(result, tuple) and len(result) == 7
    T0_vals, periods_fit, depth_fit, Tdur_fit, SNR_vals, intransit, params_df = result
    assert np.allclose(T0_vals, [candidates[0].t0])
    assert np.allclose(periods_fit, [candidates[0].period])
    assert np.allclose(depth_fit, [candidates[0].depth])
    assert intransit.shape == lc_small.time.shape
