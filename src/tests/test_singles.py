
import sys
import importlib
import numpy as np
from search_core import Candidate
from tests.helpers_fakes import make_fake_functions_all

def fake_DT_analysis(time, flux, flux_err, confidence, DT_Quite, is_flat):
    # One box: idx layout [dummy, t0=1.1, dummy, dur=0.07, comp=0.995 -> depth=0.005]
    return [[0, 1.1, 0, 0.07, 0.995]]

def test_singles_search_and_fit(monkeypatch, lc_small):
    # Build a fake Functions_all that includes DT_analysis and pymc builder
    fake = make_fake_functions_all()
    # Add a fake pymc builder that just echoes inputs
    def fake_pymc_new_general_function(time, flux, unc, T0, other_pars, type_fn,
                                       verbose=True, keep_ld_fixed=True, phase_fold=False, prior_hint=None):
        assert type_fn == 'Single'
        return {"T0": T0, "Per": other_pars[0], "depth": other_pars[2], "hint_t0_low": prior_hint["t0"]["low"]}
    fake.pymc_new_general_function = fake_pymc_new_general_function
    fake.DT_analysis = fake_DT_analysis
    monkeypatch.setitem(sys.modules, "Functions_all", fake)

    # Reload modules that import Functions_all
    singles_search = importlib.import_module("singles_search")
    transit_fit = importlib.import_module("transit_fit")

    ss = singles_search.SinglesSearch(lc_small, singles_search.SinglesSearchConfig(confidence=0.6, verbose=True))
    cands = ss.run()
    assert len(cands) == 1 and isinstance(cands[0], Candidate)

    sf = transit_fit.SinglesFitter(lc_small, ab=(0.3, 0.3))
    hints = sf.build_prior_hints(cands)
    out_list = sf.fit(cands, prior_hints=hints)
    assert isinstance(out_list, list) and len(out_list) == 1
    out = out_list[0]
    # Check the fake builder saw the right inputs
    assert out["T0"] == pytest.approx(1.1)
    assert out["Per"] == pytest.approx(lc_small.baseline_days) or out["Per"] == 27.8
