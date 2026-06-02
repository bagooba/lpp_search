# utils/periodic_event_dedup.py
from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

from core.planet_candidate import PlanetCandidate




def _get_summary(c: PlanetCandidate, var: str) -> Optional[dict]:
    d = getattr(c, "pymc_summary", None)
    if isinstance(d, dict):
        vv = d.get(var, None)
        return vv if isinstance(vv, dict) else None
    return None


def _median(c: PlanetCandidate, var_name: str, fallback=None):
    var_summary = _get_summary(c, var_name)

    if var_summary is None:
        return fallback

    if "median" not in var_summary:
        return fallback

    value = var_summary["median"]

    try:
        return float(value)
    except Exception:
        return fallback


def _hdi16(c: PlanetCandidate, var_name: str, fallback=None):
    var_summary = _get_summary(c, var_name)

    if var_summary is None:
        return fallback

    if "hdi_16%" not in var_summary:
        return fallback

    value = var_summary["hdi_16%"]

    try:
        return float(value)
    except Exception:
        return fallback

def _period(c: PlanetCandidate) -> Optional[float]:
    p = getattr(c, "period_days", None)
    return None if p is None else float(p)


def _duration_days(c: PlanetCandidate) -> Optional[float]:
    d = getattr(c, "duration_days", None)
    if d is None:
        return None
    try:
        return float(d)
    except Exception:
        return None

def _depth_med(c: PlanetCandidate) -> Optional[float]:
    d = _median(c, "depth", getattr(c, "depth", None))
    return None if d is None else float(d)


def _snr_med(c: PlanetCandidate) -> float:
    s = _median(c, "SNR", getattr(c, "snr", 0.0))
    try:
        return float(s)
    except Exception:
        return 0.0


def _max_rhat(c: PlanetCandidate) -> Optional[float]:
    d = getattr(c, "pymc_summary", None)
    if not isinstance(d, dict):
        return None

    vals = []
    for vv in d.values():
        if isinstance(vv, dict) and "r_hat" in vv:
            try:
                vals.append(float(vv["r_hat"]))
            except Exception:
                pass

    return max(vals) if vals else None


def _observed_transit_times_days(c: PlanetCandidate) -> np.ndarray:
    """
    Returns observed transit times (days) for clustering/dedup:
      - Single: [t0_days]
      - Periodic: transit_times_days if present
    Returns None if not available for periodic.
    """

    # Singles: always just the one observed epoch
    if getattr(c, "ptype", None) == "Single":
        return np.array([float(getattr(c, "t0_days"))], dtype=float)


    # Periodic: otherwise look for a candidate field (if/when you add it)
    transit_times_days = getattr(c, "transit_times_days", None)
    if transit_times_days is None:
        return np.array([], dtype=float)

    values = np.asarray(transit_times_days, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([], dtype=float)

    return np.sort(np.unique(values))


def _event_match_tolerance_days(
    a: PlanetCandidate,
    b: PlanetCandidate,
    floor_days: float = 0.1,
    frac_of_duration: float = 0.5,) -> float:
    da = _duration_days(a)
    db = _duration_days(b)

    valid = [x for x in (da, db) if x is not None and np.isfinite(x) and x > 0]

    if len(valid) == 0:
        return floor_days
    return max(floor_days, frac_of_duration * min(valid))



def _shared_event_counts(
    a: PlanetCandidate,
    b: PlanetCandidate,
    tol_days: float,
) -> Tuple[int, int, int]:
    ta = _observed_transit_times_days(a)
    tb = _observed_transit_times_days(b)

    if len(ta) == 0 or len(tb) == 0:
        return 0, int(ta.size), int(tb.size)

    i = 0
    j = 0
    overlap = 0

    while i < ta.size and j < tb.size:
        dt = ta[i] - tb[j]

        if abs(dt) <= tol_days:
            overlap += 1
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1

    return overlap, int(ta.size), int(tb.size)

def _containment_fractions(
    a: PlanetCandidate,
    b: PlanetCandidate,
    tol_days: float,
) -> Tuple[float, float, int]:
    overlap, na, nb = _shared_event_counts(a, b, tol_days=tol_days)

    frac_a_in_b = 0.0 if na == 0 else overlap / na
    frac_b_in_a = 0.0 if nb == 0 else overlap / nb
    return frac_a_in_b, frac_b_in_a, overlap


def _duration_consistent(
    a: PlanetCandidate,
    b: PlanetCandidate,
    ratio_max: float = 2.0,
) -> bool:
    da = _duration_days(a)
    db = _duration_days(b)

    valid = [x for x in (da, db) if x is not None and np.isfinite(x) and x > 0]

    if len(valid) != 2:
        return False

    dmax = max(valid)
    dmin = min(valid)

    return (dmax / dmin) <= ratio_max


def _depth_consistent(
    a: PlanetCandidate,
    b: PlanetCandidate,
    ratio_max: float = 2.0,
) -> bool:
    da = _depth_med(a)
    db = _depth_med(b)

    valid = [x for x in (da, db) if x is not None and np.isfinite(x) and x > 0]

    if len(valid) != 2:
        return False

    dmax = max(valid)
    dmin = min(valid)

    return (dmax / dmin) <= ratio_max


def _signal_not_zero(c: PlanetCandidate, min_lower_to_median_ratio: float = 0.25) -> bool:
    depth_median = _median(c, "depth", getattr(c, "depth", None))
    depth_lower = _hdi16(c, "depth", None)

    if depth_median is None or depth_lower is None:
        return False

    if not np.isfinite(depth_median) or not np.isfinite(depth_lower):
        return False

    if depth_median <= 0:
        return False

    return (depth_lower / depth_median) >= min_lower_to_median_ratio


def _winner_key(c: PlanetCandidate) -> tuple:
    signal_not_zero = 1 if _signal_not_zero(c) else 0

    observed_transit_times_days = _observed_transit_times_days(c)
    n_observed_events = len(observed_transit_times_days)

    rhat = _max_rhat(c)
    rhat_ok = 1
    rhat_score = 0.0

    if rhat is not None:
        rhat_ok = 1 if rhat <= 1.1 else 0
        rhat_score = -float(rhat)

    snr_value = _snr_med(c)

    period_days = _period(c)
    if period_days is None:
        period_score = 0.0
    else:
        period_score = -float(period_days)

    return (
        signal_not_zero,
        n_observed_events,
        rhat_ok,
        rhat_score,
        snr_value,
        period_score,
    )








# def dedup_periodic_candidates_by_shared_events(
#     periodic_candidates: List[PlanetCandidate],
#     *,
#     containment_threshold: float = 0.75,
#     min_shared_events: int = 2,
#     duration_ratio_max: float = 2.0,
#     floor_tol_days: float = 0.1,
#     frac_of_duration_tol: float = 0.25,
# ) -> List[PlanetCandidate]:
#     """
#     Mutates periodic candidates in-place.

#     Main idea:
#       - if one candidate's observed transit-event set is mostly contained in
#         another candidate's event set, they are treated as the same signal family
#       - choose the better winner and mark the loser as default=False

#     This is intended to replace period-ratio alias dedup as the primary criterion.
#     """

#     cands = [
#         c for c in periodic_candidates
#         if getattr(c, "ptype", None) == "Periodic"
#         and _period(c) is not None
#     ]

#     n = len(cands)
#     if n < 2:
#         return periodic_candidates

#     for i in range(n):
#         a = cands[i]
#         if not getattr(a, "default", True):
#             continue

#         for j in range(i + 1, n):
#             b = cands[j]
#             if not getattr(b, "default", True):
#                 continue

#             tol_days = _event_match_tolerance_days(
#                 a,
#                 b,
#                 floor_days=floor_tol_days,
#                 frac_of_duration=frac_of_duration_tol,
#             )

#             frac_a_in_b, frac_b_in_a, overlap = _containment_fractions(
#                 a,
#                 b,
#                 tol_days=tol_days,
#             )

#             if overlap < min_shared_events:
#                 continue

#             # Main alias criterion:
#             # one candidate mostly reuses the observed transit events of the other
#             if max(frac_a_in_b, frac_b_in_a) < containment_threshold:
#                 continue

#             # Duration veto: if the same claimed events imply wildly different
#             # durations, do not merge automatically.
#             if not _duration_consistent(a, b, ratio_max=duration_ratio_max):
#                 continue

#             winner = a if _winner_key(a) >= _winner_key(b) else b
#             loser = b if winner is a else a

#             loser.default = False
#             note = (
#                 f"periodic_event_dedup: shared claimed transit events with "
#                 f"{winner.candidate_id()}"
#             )
#             loser.notes = (loser.notes + "; " + note) if getattr(loser, "notes", "") else note

#     return periodic_candidates