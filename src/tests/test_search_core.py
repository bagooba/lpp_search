
import numpy as np
from search_core import (
    ephemeris_uncertainties_from_transit_times,
    prior_hint_from_duration,
    prior_hint_from_midtimes,
    compute_intransit_masks,
    Candidate,
)

def test_ols_ephemeris_perfect_times():
    # Perfect linear times: t0=100, P=3.2
    t0_true, P_true = 100.0, 3.2
    E = np.arange(5)  # 0..4
    times = t0_true + E * P_true
    P_hat, t0_hat, sP, st0 = ephemeris_uncertainties_from_transit_times(times)
    assert np.isfinite(P_hat) and np.isfinite(t0_hat)
    assert abs(P_hat - P_true) < 1e-12
    assert abs(t0_hat - t0_true) < 1e-12
    # Zero residuals imply near-zero estimated sigma values for this synthetic case
    assert sP >= 0.0 and st0 >= 0.0

def test_prior_hint_from_duration_bounds(lc_small, candidate_basic):
    hint = prior_hint_from_duration(lc_small.time, lc_small.flux, candidate_basic, beta=0.75)
    # t0 bounds include at least cadence floor and duration-derived width
    t0_low, t0_high = hint["t0"]["low"], hint["t0"]["high"]
    assert t0_low < candidate_basic.t0 < t0_high
    # Period bounds are symmetric around the period guess
    P_mu = hint["Per"]["mu"]
    assert hint["Per"]["low"] < P_mu < hint["Per"]["high"]
    # rp_rs window looks sane and within [0, 1]
    rp = hint["rp_rs"]
    assert 0.0 <= rp["low"] < rp["mu"] < rp["high"] <= 1.0

def test_prior_hint_from_midtimes_ols(lc_small, candidate_basic):
    # Build mid-times from a perfect ephemeris: t0=1.0, P=2.0, 3 transits
    t0_true, P_true = 1.0, 2.0
    midtimes = np.array([t0_true + k * P_true for k in range(3)], dtype=float)
    cand = Candidate(
        period=P_true, t0=t0_true, duration=0.08, depth=0.005, midtimes=midtimes
    )
    hint = prior_hint_from_midtimes(lc_small.time, lc_small.flux, cand, beta=0.75)
    # Bounds should be centered near t0_true and P_true
    assert abs(hint["t0"]["mu"] - t0_true) < 1e-8
    assert abs(hint["Per"]["mu"] - P_true) < 1e-8

def test_compute_intransit_masks_union(lc_small, candidate_basic):
    # Two candidates at different t0 to ensure union expands
    c1 = candidate_basic
    c2 = Candidate(period=2.0, t0=1.6, duration=0.08, depth=0.005)
    masks, union = compute_intransit_masks(lc_small.time, [c1, c2], buffer=0.0)
    assert len(masks) == 2
    assert union.any()
    # Union should include both windows
    assert masks[0].sum() > 0 and masks[1].sum() > 0
    assert union.sum() >= max(masks[0].sum(), masks[1].sum())
