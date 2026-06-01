# utils/alias_dedup.py
from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Optional

from core.planet_candidate import PlanetCandidate
from utils.singles_periodicity import potential_periods_from_single_dt_events

# ---------- small helpers ----------
def _group_surviving_periodic_candidates_by_near_equal_period(
    periodic_candidates: List[PlanetCandidate],
    same_period_frac_tol: float,
) -> List[List[PlanetCandidate]]:
    """
    Group surviving default periodic candidates by near-equal reported period.
    This is intentionally NOT a harmonic-ratio grouping.
    """
    survivors = [
        c for c in periodic_candidates
        if getattr(c, "ptype", None) == "Periodic"
        and _period(c) is not None
        and getattr(c, "default", True)
    ]

    if len(survivors) < 2:
        return []

    survivors = sorted(survivors, key=lambda c: _period(c))

    groups: List[List[PlanetCandidate]] = []
    current_group: List[PlanetCandidate] = [survivors[0]]

    for c in survivors[1:]:
        current_period_days = _period(c)
        reference_period_days = np.median([_period(x) for x in current_group])

        frac_diff = abs(current_period_days - reference_period_days) / max(reference_period_days, 1e-12)

        if frac_diff <= same_period_frac_tol:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]

    groups.append(current_group)
    return groups


def _count_family_members_supporting_period(
    members: List[PlanetCandidate],
    proposed_period_days: float,
    proposed_t0_days: float,
    tol_days: float,
) -> int:
    """
    Count how many family members contribute at least one observed transit time
    that lands on the proposed shorter-period grid.
    """
    supporting_members = 0

    for c in members:
        observed_transit_times_days = _get_observed_transit_times(c)
        if observed_transit_times_days is None or observed_transit_times_days.size == 0:
            continue

        phase_residual_days = (
            (observed_transit_times_days - proposed_t0_days + 0.5 * proposed_period_days)
            % proposed_period_days
        ) - 0.5 * proposed_period_days

        if np.any(np.abs(phase_residual_days) <= tol_days):
            supporting_members += 1

    return supporting_members


def _flag_missing_base_period_families(
    periodic_candidates: List[PlanetCandidate],
    *,
    same_period_frac_tol: float = 0.02,
    min_group_size: int = 3,
    min_support: int = 3,
    min_supporting_members: int = 3,
    shorter_period_fraction_max: float = 0.8,
    support_tol_days: float = 0.05,
) -> None:
    """
    Second pass:
      - look only at surviving default periodic candidates
      - group by near-equal reported period
      - check whether a family implies a shorter missing base period
      - annotate notes only; do not mark default=False here
    """
    period_families = _group_surviving_periodic_candidates_by_near_equal_period(
        periodic_candidates=periodic_candidates,
        same_period_frac_tol=same_period_frac_tol,
    )

    for members in period_families:
        if len(members) < min_group_size:
            continue

        family_reported_period_days = np.array([_period(c) for c in members], dtype=float)
        family_median_period_days = float(np.median(family_reported_period_days))

        proposal = _infer_missing_base_period_from_cluster(
            members,
            min_support=min_support,
            perc_tol=0.02,
            max_ratio=5,
            P_min=0.25,
            merge_tol=same_period_frac_tol,
            support_tol_days=support_tol_days,
        )

        if proposal is None:
            continue

        proposed_period_days = float(proposal["period_days"])
        proposed_t0_days = float(proposal["t0_days"])
        proposed_support = int(proposal["support"])

        if proposed_period_days > shorter_period_fraction_max * family_median_period_days:
            continue

        supporting_members = _count_family_members_supporting_period(
            members=members,
            proposed_period_days=proposed_period_days,
            proposed_t0_days=proposed_t0_days,
            tol_days=support_tol_days,
        )

        if supporting_members < min_supporting_members:
            continue

        add_str = ''
        if min_supporting_members<3:
            add_str = ' weakly? '
        note = (
            "alias_dedup: near-equal-period family"+add_str+"implies missing base period "
            f"~{proposed_period_days:.6f} d "
            f"(family_period~{family_median_period_days:.6f} d, "
            f"support={proposed_support}, "
            f"supporting_members={supporting_members})"
        )

        for c in members:
            c.notes = (c.notes + "; " + note) if getattr(c, "notes", "") else note

def _fraction_observed_transit_times_explained_by_ephemeris(
    observed_transit_times_days: np.ndarray,
    ephemeris_t0_days: float,
    ephemeris_period_days: float,
    tol_days: float,
) -> float:
    """
    Fraction of observed transit times that fall on the ephemeris
    (ephemeris_t0_days, ephemeris_period_days) within tol_days.
    """
    if (
        observed_transit_times_days is None
        or observed_transit_times_days.size == 0
        or not np.isfinite(ephemeris_period_days)
        or ephemeris_period_days <= 0
    ):
        return 0.0

    phase_residual_days = (
        (observed_transit_times_days - ephemeris_t0_days + 0.5 * ephemeris_period_days)
        % ephemeris_period_days
    ) - 0.5 * ephemeris_period_days

    return float(np.mean(np.abs(phase_residual_days) <= tol_days))


def _mutual_ephemeris_explainability(
    a: PlanetCandidate,
    b: PlanetCandidate,
    tol_days: float,
) -> Tuple[float, float]:
    """
    Returns
    -------
    fraction_a_explained_by_b : float
        Fraction of a's observed transit times explained by b's ephemeris.
    fraction_b_explained_by_a : float
        Fraction of b's observed transit times explained by a's ephemeris.
    """
    a_observed_transit_times_days = _get_observed_transit_times(a)
    b_observed_transit_times_days = _get_observed_transit_times(b)

    if a_observed_transit_times_days is None or b_observed_transit_times_days is None:
        return 0.0, 0.0

    a_period_days = _period(a)
    b_period_days = _period(b)

    if a_period_days is None or b_period_days is None:
        return 0.0, 0.0

    fraction_a_explained_by_b = _fraction_observed_transit_times_explained_by_ephemeris(
        observed_transit_times_days=a_observed_transit_times_days,
        ephemeris_t0_days=_t0(b),
        ephemeris_period_days=b_period_days,
        tol_days=tol_days,
    )

    fraction_b_explained_by_a = _fraction_observed_transit_times_explained_by_ephemeris(
        observed_transit_times_days=b_observed_transit_times_days,
        ephemeris_t0_days=_t0(a),
        ephemeris_period_days=a_period_days,
        tol_days=tol_days,
    )

    return fraction_a_explained_by_b, fraction_b_explained_by_a




def _implied_base_from_equal_period_t0s(
    t0s: np.ndarray,
    P: float,
    tol_days: float = 0.1,
    max_div: int = 5,
) -> Optional[float]:
    """
    Given multiple t0 values for candidates with nearly the same reported period P,
    check whether their phase offsets imply a smaller base period P/k.
    Returns the best implied base period or None.
    """
    if t0s.size < 2 or not np.isfinite(P) or P <= 0:
        return None

    # phases in [0, P)
    phases = np.sort(np.mod(t0s, P))

    # try divisors k = 2..max_div, meaning base period = P/k
    for k in range(2, max_div + 1):
        Pbase = P / k

        # take the first phase as reference
        ref = phases[0]
        resid = ((phases - ref + 0.5 * Pbase) % Pbase) - 0.5 * Pbase

        if np.all(np.abs(resid) <= tol_days):
            return float(Pbase)

    return None

def _best_support_for_period(
    t0s: np.ndarray,
    P: float,
    tol_days: float,
) -> Tuple[int, Optional[float]]:
    """
    For a trial period P, try each observed t0 as the anchor epoch and return:
      - maximum number of observed times supported
      - best anchor t0
    """
    if t0s.size == 0 or not np.isfinite(P) or P <= 0:
        return 0, None

    best_support = 0
    best_t0 = None

    for t0 in t0s:
        resid = ((t0s - t0 + 0.5 * P) % P) - 0.5 * P
        support = int(np.sum(np.abs(resid) <= tol_days))

        if support > best_support:
            best_support = support
            best_t0 = float(t0)

    return best_support, best_t0


def _infer_missing_base_period_from_cluster(
    members: List[PlanetCandidate],
    *,
    min_support: int = 3,
    perc_tol: float = 0.02,
    max_ratio: int = 5,
    P_min: float = 0.25,
    merge_tol: float = 0.02,
    support_tol_days: float = 0.05,
) -> Optional[dict]:
    """
    Look at all observed transit times in a cluster and ask whether they imply
    a plausible base period that is NOT already one of the candidate periods.

    Returns
    -------
    proposal : dict or None
        {
            "period_days": best proposed missing base period,
            "t0_days": best anchor epoch,
            "support": number of observed epochs explained
        }
    """
    all_t0s = []
    existing_periods = []

    for c in members:
        tt = _get_observed_transit_times(c)
        if tt is not None and tt.size > 0:
            all_t0s.extend(tt.tolist())

        P = _period(c)
        if P is not None and np.isfinite(P) and P > 0:
            existing_periods.append(float(P))

    if len(all_t0s) < min_support:
        return None

    t0s = np.unique(np.asarray(all_t0s, dtype=float))
    if t0s.size < min_support:
        return None

    trial_periods = potential_periods_from_single_dt_events(
        t0s,
        min_support=min_support,
        perc_tol=perc_tol,
        max_ratio=max_ratio,
        P_min=P_min,
    )

    if trial_periods.size == 0:
        return None

    # Remove any period already represented by an existing candidate
    truly_missing = []
    for P in trial_periods:
        already_present = any(
            abs(P - P0) / max(P0, 1e-12) < merge_tol
            for P0 in existing_periods
        )
        if not already_present:
            truly_missing.append(float(P))

    if len(truly_missing) == 0:
        return None

    best = None
    for P in truly_missing:
        support, best_t0 = _best_support_for_period(
            t0s,
            P,
            tol_days=support_tol_days,
        )

        if support < min_support or best_t0 is None:
            continue

        # prefer higher support; among ties prefer shorter period
        key = (support, -P)

        if best is None or key > best["key"]:
            best = {
                "key": key,
                "period_days": float(P),
                "t0_days": float(best_t0),
                "support": int(support),
            }

    if best is None:
        return None

    return {
        "period_days": best["period_days"],
        "t0_days": best["t0_days"],
        "support": best["support"],
    }

def _get_summary(c: PlanetCandidate, var: str) -> Optional[dict]:
    d = getattr(c, "pymc_summary", None)
    if isinstance(d, dict):
        vv = d.get(var, None)
        return vv if isinstance(vv, dict) else None
    return None

def _median(c: PlanetCandidate, var: str, fallback=None):
    vv = _get_summary(c, var)
    if vv and "median" in vv:
        try:
            return float(vv["median"])
        except Exception:
            pass
    return fallback

def _hdi16(c: PlanetCandidate, var: str, fallback=None):
    vv = _get_summary(c, var)
    if vv and "hdi_16%" in vv:
        try:
            return float(vv["hdi_16%"])
        except Exception:
            pass
    return fallback

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


def _depth_med(c: PlanetCandidate) -> Optional[float]:
    d = _median(c, "depth", getattr(c, "depth", None))
    return None if d is None else float(d)


def _signal_not_zero(c: PlanetCandidate, eps: float = 1e-6) -> bool:
    """
    Your idea: reject periods that 'fit flat light curves'.
    Implemented as: rp_rs (or depth) HDI lower bound must be > 0-ish.
    """
    rp_lo = _hdi16(c, "rp_rs", None)
    if rp_lo is not None:
        return rp_lo > eps
    d_lo = _hdi16(c, "depth", None)
    if d_lo is not None:
        return d_lo > eps
    # fallback: if no HDI, use median depth
    d_med = _depth_med(c)
    return (d_med is not None) and (float(d_med) > eps)

def _snr_med(c: PlanetCandidate) -> float:
    s = _median(c, "SNR", getattr(c, "snr", 0.0))
    try:
        return float(s)
    except Exception:
        return 0.0

def _period(c: PlanetCandidate) -> Optional[float]:
    p = getattr(c, "period_days", None)
    return None if p is None else float(p)

def _t0(c: PlanetCandidate) -> float:
    return float(getattr(c, "t0_days"))



def _phase_offset_days(t: float, t0: float, P: float) -> float:
    """
    Distance (days) between t and nearest epoch implied by (t0,P)
    """
    x = (t - t0 + 0.5 * P) % P - 0.5 * P
    return abs(x)


# ---------- clustering criteria ----------
def _get_observed_transit_times(
    c
) -> Optional[np.ndarray]:
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
    arr = getattr(c, "transit_times_days", None)
    if arr:
        vv = np.asarray(arr, dtype=float)
        vv = vv[np.isfinite(vv)]
        return np.sort(vv) if vv.size else None

    return None



def _shared_observed_transit_time_overlap(a: PlanetCandidate, b: PlanetCandidate, tol_days: float) -> Tuple[int, int]:
    """
    Count overlaps between observed transit-time lists.
    Returns (n_overlap, n_minlist). If either list missing -> (0,0).
    """
    ta = _get_observed_transit_times(a)
    tb = _get_observed_transit_times(b)
    if ta is None or tb is None:
        return 0, 0

    i = j = 0
    overlap = 0
    while i < ta.size and j < tb.size:
        da = ta[i]
        db = tb[j]
        if abs(da - db) <= tol_days:
            overlap += 1
            i += 1
            j += 1
        elif da < db:
            i += 1
        else:
            j += 1

    return overlap, int(min(ta.size, tb.size))

def _depth_consistent(a: PlanetCandidate, b: PlanetCandidate, ratio_max: float = 1.75, floor: float = 5e-5) -> bool:
    da = _depth_med(a)
    db = _depth_med(b)
    if da is None or db is None or (not np.isfinite(da)) or (not np.isfinite(db)):
        return True  # don't block if depth missing
    dmax = max(da, db)
    dmin = max(min(da, db), floor)
    return (dmax / dmin) <= ratio_max


# ---------- winner selection ----------
def _winner_key(c: PlanetCandidate) -> Tuple:
    """
    Higher is better. This encodes what we discussed:
      1) signal_not_zero (your 'flat LC fit' veto)
      2) fit_is_current True
      3) lower rhat (better)  -> negative
      4) higher SNR median
      5) shorter period (tie-breaker; negative period so smaller wins)
    """
    sig = 1 if _signal_not_zero(c) else 0
    fit = 1 if getattr(c, "fit_is_current", False) else 0

    rhat = _max_rhat(c)
    # rhat_ok: 1 if acceptable / 0 if not (tune threshold)
    rhat_ok = 1
    if rhat is not None and float(rhat) > 1.1:
        rhat_ok = 0
    snr = _snr_med(c)
    P = _period(c)
    per_score = 0.0 if P is None else -float(P)  # shorter is better as tie-breaker
    return (sig, fit, rhat_ok, snr, per_score)


# ---------- main entrypoint ----------
def alias_dedup_periodic_candidates(
    periodic_candidates: List[PlanetCandidate],
    *,
    # clustering thresholds
    shared_t0_tol_days: float = 0.05,
    shared_overlap_frac: float = 0.6,
    depth_ratio_max: float = 1.75,
    # ephemeris fallback thresholds
    epoch_tol_scale: float = 0.25,
    epoch_tol_floor_days: float = 0.075,
) -> List[PlanetCandidate]:
    """
    Mutates candidates in-place:
      - sets default=False on duplicates
      - appends a note pointing to the winner candidate_id
    Returns the same list (mutated).
    """

    # only periodic with a period
    cands = [c for c in periodic_candidates if getattr(c, "ptype", None) == "Periodic" and _period(c) is not None]
    n = len(cands)
    if n < 2:
        return periodic_candidates

    # union-find clustering
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # pairwise cluster decision
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cands[i], cands[j]

            # If both candidates have observed transit-time lists, use those directly.
            a_observed_transit_times_days = _get_observed_transit_times(a)
            b_observed_transit_times_days = _get_observed_transit_times(b)

            if a_observed_transit_times_days is not None and b_observed_transit_times_days is not None:
                if not _depth_consistent(a, b, ratio_max=depth_ratio_max):
                    continue

                overlap_count, overlap_denom = _shared_observed_transit_time_overlap(
                    a, b, tol_days=shared_t0_tol_days
                )

                if overlap_denom > 0 and (overlap_count / overlap_denom) >= shared_overlap_frac:
                    union(i, j)
                    continue

                fraction_a_explained_by_b, fraction_b_explained_by_a = _mutual_ephemeris_explainability(
                    a, b, tol_days=shared_t0_tol_days
                )

                if (
                    fraction_a_explained_by_b >= 0.85
                    and fraction_b_explained_by_a >= 0.85
                ):
                    union(i, j)

                continue
            # elif off
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    # apply winner/loser marking
    for idx_list in groups.values():
        if len(idx_list) < 2:
            continue

        members = [cands[i] for i in idx_list]

        missing_base = _infer_missing_base_period_from_cluster(
            members,
            min_support=3,
            perc_tol=0.02,
            max_ratio=5,
            P_min=0.25,
            merge_tol=0.02,
            support_tol_days=shared_t0_tol_days,
        )

        winner = max(members, key=_winner_key)
        winner_id = winner.candidate_id()

        if missing_base is not None:
            note = (
                f"alias_dedup: cluster implies missing base period "
                f"~{missing_base['period_days']:.6f} d "
                f"(support={missing_base['support']})"
            )
            winner.notes = (winner.notes + "; " + note) if getattr(winner, "notes", "") else note


        for m in members:
            if m is winner:
                continue
            m.default = False
            note = f"alias_dedup: duplicate/alias of {winner_id}"
            m.notes = (m.notes + "; " + note) if getattr(m, "notes", "") else note

    _flag_missing_base_period_families(
            periodic_candidates,
            same_period_frac_tol=0.02,
            min_group_size=3,
            min_support=3,
            min_supporting_members=3,
            shorter_period_fraction_max=0.8,
            support_tol_days=shared_t0_tol_days,
        )

    return periodic_candidates