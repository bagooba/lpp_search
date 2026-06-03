# utils/alias_dedup.py
from __future__ import annotations
import math
import copy

import numpy as np
from typing import List, Tuple, Optional

from core.transit_event import TransitEvent
from core.periodic_event import PeriodicEvent

from utils.singles_periodicity import potential_periods_from_single_dt_events, best_t0_and_snr_for_period
import utils.periodic_event_dedup as ped

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
        and ped._period(c) is not None
        and getattr(c, "default", True)
    ]

    if len(survivors) < 2:
        return []

    survivors = sorted(survivors, key=lambda c: ped._period(c))

    groups: List[List[PlanetCandidate]] = []
    current_group: List[PlanetCandidate] = [survivors[0]]

    for c in survivors[1:]:
        current_period_days = ped._period(c)
        reference_period_days = np.median([ped._period(x) for x in current_group])

        frac_diff = abs(current_period_days - reference_period_days) / max(reference_period_days, 1e-12)

        if frac_diff <= same_period_frac_tol:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]

    groups.append(current_group)
    return groups

def _get_all_t0_values(members):
    all_t0s = []
    existing_periods = []
    for c in members:
        if isinstance(c, PlanetCandidate):
            tt = ped._observed_transit_times_days(c)
            if tt is not None and tt.size > 0:
                all_t0s.extend(tt.tolist())

            try:
                P = ped._period(c)
                if P is not None and np.isfinite(P) and P > 0:
                    existing_periods.append(float(P))
            except Exception:
                pass

        else:  # TransitEvent
            t0 = getattr(c, "t0_days", None)
            if t0 is not None:
                all_t0s.append(float(t0))
    return all_t0s, existing_periods



def _count_family_members_supporting_period(
    members: List[PlanetCandidate],
    proposed_period_days: float,
    proposed_t0_days: float,
    tol_days: float,
) -> Tuple[int, List[PlanetCandidate]]:
    """
    Count how many family members contribute at least one observed transit time
    that lands on the proposed shorter-period grid.
    """
    supporting_members = 0
    inc_members = []

    for c in members:

        # --- per-member times ---
        if isinstance(c, PlanetCandidate):
            obs_t0s = ped._observed_transit_times_days(c)
        else:
            t0 = getattr(c, "t0_days", None)
            if t0 is None:
                continue
            obs_t0s = np.array([float(t0)])

        if obs_t0s is None or len(obs_t0s) == 0:
            continue
    
        phase_residual_days = (
            (obs_t0s - proposed_t0_days + 0.5 * proposed_period_days)
            % proposed_period_days
        ) - 0.5 * proposed_period_days

        if np.any(np.abs(phase_residual_days) <= tol_days):
            supporting_members += 1
            inc_members.append(c)

    return supporting_members, inc_members


def _flag_missing_base_period_families(
    periodic_candidates: List[PlanetCandidate],
    *,
    same_period_frac_tol: float = 0.02,
    min_group_size: int = 3,
    min_support: int = 3,
    min_supporting_members: int = 3,
    shorter_period_fraction_max: float = 0.8,
    support_tol_days: float = 0.05,
) -> List[Tuple[dict, List[PlanetCandidate], PlanetCandidate]]:
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

    try_props = []


    for members in period_families:
        if len(members) < min_group_size:
            continue

        family_reported_period_days = np.array([ped._period(c) for c in members], dtype=float)
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

        supporting_members_count, sup_members = _count_family_members_supporting_period(
            members=members,
            proposed_period_days=proposed_period_days,
            proposed_t0_days=proposed_t0_days,
            tol_days=support_tol_days,
        )

        if supporting_members_count < min_supporting_members:
            continue

        add_str = ''
        if min_supporting_members<3:
            add_str = ' weakly? '
        note = (
            "alias_dedup: near-equal-period family"+add_str+"implies missing base period "
            f"~{proposed_period_days:.6f} d "
            f"(family_period~{family_median_period_days:.6f} d, "
            f"support={proposed_support}, "
            f"supporting_members={supporting_members_count})"
        )

        winner = sup_members[0]  # placeholder if you want something simple for now
        try_props.append((proposal, sup_members, winner))

        for c in members:
            c.notes = (c.notes + "; " + note) if getattr(c, "notes", "") else note
    return try_props



def _best_support_for_period(
    t0s: np.ndarray,
    P: float,
    tol_days: float,
) -> Tuple[int, Optional[float], np.ndarray]:
    """
    For a trial period P, try each observed t0 as the anchor epoch and return:
      - maximum number of observed times supported
      - best anchor t0
      - matched observed transit times for that anchor
    """
    t0s = np.asarray(t0s, dtype=float)

    if t0s.size == 0 or not np.isfinite(P) or P <= 0:
        return 0, None, np.array([], dtype=float)

    best_support = 0
    best_t0 = None
    best_transit = np.array([], dtype=float)

    for t0 in t0s:
        resid = ((t0s - t0 + 0.5 * P) % P) - 0.5 * P
        matched = np.abs(resid) <= tol_days
        support = int(np.sum(matched))

        if support > best_support:
            best_support = support
            best_t0 = float(t0)
            best_transit = t0s[matched]

    return best_support, best_t0, best_transit


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
    all_t0s, existing_periods = _get_all_t0_values(members)
     
    if len(all_t0s) < min_support:
        return None

    t0s = np.unique(np.asarray(all_t0s, dtype=float))
    if t0s.size < min_support:
        return None

    trial_periods = potential_periods_from_single_dt_events(
        t0s,
        min_support=min_support,
        perc_tol=perc_tol,
        max_divisor=max_ratio,
        P_min=P_min,
    )

    if trial_periods.size == 0:
        print('no trial periods - stop')
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
        print('all trial periods already found - stop')
        return None

    best = None
    for P in truly_missing:
        support, best_t0, mid_transit_times = _best_support_for_period(
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
                "matched_times_days": list(mid_transit_times),
                "support": int(support),
            }

    if best is None:
        return None

    return {
        "period_days": best["period_days"],
        "t0_days": best["t0_days"],
        "matched_times_days": best["matched_times_days"],
        "support": best["support"],
    }

# ---------- considering singles ----------

def _get_event_like_duration(x):
    if hasattr(x, "duration_days"):
        return x.duration_days
    return None

def _get_event_like_depth(x):
    if hasattr(x, "depth"):
        return x.depth
    return None


def _add_matching_singles_to_group(
    group: list[PlanetCandidate],
    extra_planet_candidates,
) -> list:

    if extra_planet_candidates is None or len(extra_planet_candidates) == 0:
        return list(group)

    augmented = list(group)

    for extra in extra_planet_candidates:

        extra_duration = _get_event_like_duration(extra)
        extra_depth = _get_event_like_depth(extra)

        if extra_duration is None or extra_depth is None:
            continue

        for member in group:
            member_duration = ped._duration_days(member)
            member_depth = ped._depth_med(member)

            if member_duration is None or member_depth is None:
                continue

            dur_ratio = max(extra_duration, member_duration) / max(min(extra_duration, member_duration), 1e-12)
            depth_ratio = max(extra_depth, member_depth) / max(min(extra_depth, member_depth), 1e-12)

            if dur_ratio <= 2.0 or depth_ratio <= 2.0:
                augmented.append(extra)
                break

    return augmented


def _make_missing_base_refit_candidate(
    winner: PlanetCandidate,
    cluster: List[PlanetCandidate],
    missing_base: dict,
    ) -> PlanetCandidate:
    """
    Build a new PlanetCandidate to seed a refit at an inferred missing base period.

    Strategy:
      - clone the winner so the returned object matches the type/shape expected
      - replace period_days with the inferred base period
      - set t0_days from an observed transit time
      - keep morphology (duration/depth) from robust cluster summaries if possible,
        otherwise fall back to winner values
    """
    refit_cand = copy.deepcopy(winner)

    new_period = float(missing_base["period_days"])
    refit_cand.period_days = new_period

    # collect all observed transit times from the augmented members
    all_times = missing_base['matched_times_days']

    all_times = sorted(set(all_times))
    

    # t0 choice:
    # for a periodic candidate, any observed transit center on the new ephemeris is a valid epoch anchor.
    # simplest/safest seed is the earliest observed transit time.
    if len(all_times) > 0:
        refit_cand.t0_days = all_times[0]
        refit_cand.transit_times_days = all_times
        refit_cand.n_transits_obs = len(all_times)
    else:
        # fallback if somehow no observed times are available
        refit_cand.t0_days = getattr(winner, "t0_days", None)

    # robust morphology from the periodic cluster only
    cluster_durations = np.array([ped._duration_days(c) for c in cluster])
    cluster_depths = np.array([ped._depth_med(c) for c in cluster])


    finite_durations = cluster_durations[np.isfinite(cluster_durations)]
    finite_depths = cluster_depths[np.isfinite(cluster_depths)]

    if len(finite_durations) > 0:
        refit_cand.duration_days = float(np.nanmedian(finite_durations))

    if len(finite_depths) > 0:
        refit_cand.depth = float(np.nanmedian(finite_depths))   

    # metadata / bookkeeping
    refit_cand.default = False
    refit_cand.fit_is_current = False

    note = (
        f"alias_dedup: refit candidate seeded from {winner.candidate_id()} "
        f"at inferred missing base period ~{new_period:.6f} d "
        f"(support={missing_base['support']})"
    )
    refit_cand.notes = (
        refit_cand.notes + "; " + note
        if getattr(refit_cand, "notes", "")
        else note
    )

    return refit_cand



# ---------- main entrypoint ----------
def alias_dedup_periodic_candidates(
    periodic_candidates: List[PlanetCandidate],
    extra_planet_candidates: List[PlanetCandidate],
    *,
    shared_t0_tol_days: float = 0.05,
    containment_threshold: float = 0.5,
    min_shared_events: int = 2,
    depth_ratio_max: float = 1.75,
    duration_ratio_max: float = 1.5,
    frac_of_duration_tol: float = 0.25,
) -> Tuple[List[PlanetCandidate], List[dict]]:
    """
    Mutates candidates in-place:
      - sets default=False on duplicates
      - appends a note pointing to the winner candidate_id
    Returns the same list (mutated).

    Main duplicate criterion:
      two periodic candidates are treated as the same planet family if
      one candidate's observed transit-event set is mostly contained in
      the other's.
    """

    cands = [
        c for c in periodic_candidates
        if getattr(c, "ptype", None) == "Periodic" and ped._period(c) is not None
    ]
    n = len(cands)


    refit_requests = []
    seen_refits = set()


    # if n < 2:
    #     return periodic_candidates

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    # pairwise cluster decision
    for i in range(n):
        for j in range(i + 1, n):
            a = cands[i]
            b = cands[j]

            transit_times_a_days = ped._observed_transit_times_days(a)
            transit_times_b_days = ped._observed_transit_times_days(b)

            if len(transit_times_a_days) == 0 or len(transit_times_b_days) == 0:
                continue

            match_tolerance_days = ped._event_match_tolerance_days(
                a,
                b,
                floor_days=shared_t0_tol_days,
                frac_of_duration=frac_of_duration_tol,
            )


            overlap_count, na, nb = ped._shared_event_counts(
                a,
                b,
                tol_days=match_tolerance_days,
            )


            frac_a_in_b, frac_b_in_a, _ = ped._containment_fractions(
                a,
                b,
                tol_days=match_tolerance_days,
            )

            unique_a_count = na - overlap_count
            unique_b_count = nb - overlap_count

            if overlap_count < min_shared_events:
                continue

            if max(frac_a_in_b, frac_b_in_a) < containment_threshold:
                continue

            if not (ped._depth_consistent(a, b, ratio_max=depth_ratio_max) or ped._duration_consistent(a, b, ratio_max=duration_ratio_max)):
                continue


            # if both still have meaningful unique evidence beyond the overlap,
            # do not merge immediately. Let missing-base-period logic handle them.
            if unique_a_count > 1 and unique_b_count > 1:
                continue

            union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)



    # apply winner/loser marking
    for root, idxs in groups.items():
        cluster = [cands[i] for i in idxs]

        members = _add_matching_singles_to_group(
            cluster,
            extra_planet_candidates,
        )

        missing_base = _infer_missing_base_period_from_cluster(
            members,
            min_support=3,
            perc_tol=0.02,
            max_ratio=5,
            P_min=0.25,
            merge_tol=0.02,
            support_tol_days=shared_t0_tol_days,
        )

        # hard preference: if only one member passes signal_not_zero, it wins
        current_members = [m for m in cluster if getattr(m, "fit_is_current", False)]

        if len(current_members) > 0:
            winner_pool = current_members
        else:
            winner_pool = cluster

        # passing_signal = [m for m in winner_pool if ped._signal_not_zero(m)]

        # if len(passing_signal) == 1:
        #     winner = passing_signal[0]
        # elif len(passing_signal) > 1:
        #     winner = max(passing_signal, key=ped._winner_key)
        # else:
        winner = max(winner_pool, key=ped._winner_key)
        winner_id = winner.candidate_id()

        if missing_base is not None:
            key = (winner.candidate_id(), round(float(missing_base["period_days"]), 6))

            if key not in seen_refits:
                refit_requests.append({
                    "candidate": _make_missing_base_refit_candidate(
                        winner,
                        cluster,
                        missing_base,
                    ),
                    "family_ids": [m.candidate_id() for m in cluster],
                    "proposed_period_days": float(missing_base["period_days"]),
                })
                seen_refits.add(key)

                note = (
                    f"alias_dedup: cluster implies missing base period "
                    f"~{missing_base['period_days']:.6f} d "
                    f"(support={missing_base['support']})"
                )
                winner.notes = (winner.notes + "; " + note) if getattr(winner, "notes", "") else note


        for m in cluster:
            if m is winner:
                continue
            m.default = False
            note = f"alias_dedup: duplicate/alias of {winner_id}"
            m.notes = (m.notes + "; " + note) if getattr(m, "notes", "") else note

    propose_targets = _flag_missing_base_period_families(
        periodic_candidates,
        same_period_frac_tol=0.02,
        min_group_size=2,
        min_support=3,
        min_supporting_members=2,
        shorter_period_fraction_max=0.8,
        support_tol_days=shared_t0_tol_days,
    )

    for tup in propose_targets:
        missing_base, cluster, winner = tup
        key = (winner.candidate_id(), round(float(missing_base["period_days"]), 6))

        if missing_base is not None and key not in seen_refits:
            refit_requests.append({
                "candidate": _make_missing_base_refit_candidate(
                    winner,
                    cluster,
                    missing_base,
                ),
                "family_ids": [m.candidate_id() for m in cluster],
                "proposed_period_days": float(missing_base["period_days"]),
            })            
            seen_refits.add(key)

    return periodic_candidates, refit_requests




# ---------- main entrypoint ----------
def alias_dedup_periodic_events(
    periodic_events: List[PeriodicEvent],
    extra_planet_events: List[TransitEvent],
    *,
    shared_t0_tol_days: float = 0.3,
    containment_threshold: float = 0.65,
    min_shared_events: int = 2,
    depth_ratio_max: float = 1.75,
    duration_ratio_max: float = 2,
    frac_of_duration_tol: float = 2,
) -> Tuple[List[PlanetCandidate], List[dict]]:
    """
    Mutates candidates in-place:
      - sets default=False on duplicates
      - appends a note pointing to the winner candidate_id
    Returns the same list (mutated).

    Main duplicate criterion:
      two periodic candidates are treated as the same planet family if
      one candidate's observed transit-event set is mostly contained in
      the other's.
    """


    cands = [
        c for c in periodic_events
        if getattr(c, "ptype", None) == "Periodic" and ped._period(c) is not None
    ]
    n = len(cands)

    refit_requests = []
    seen_refits = set()


    # if n < 2:
    #     return periodic_events

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    # pairwise cluster decision
    for i in range(n):
        for j in range(i + 1, n):
            a = cands[i]
            b = cands[j]

            transit_times_a_days = ped._observed_transit_times_days(a)
            transit_times_b_days = ped._observed_transit_times_days(b)

            if len(transit_times_a_days) == 0 or len(transit_times_b_days) == 0:
                continue

            match_tolerance_days = ped._event_match_tolerance_days(
                a,
                b,
                floor_days=shared_t0_tol_days,
                frac_of_duration=frac_of_duration_tol,
            )


            overlap_count, na, nb = ped._shared_event_counts(
                a,
                b,
                tol_days=match_tolerance_days,
            )


            frac_a_in_b, frac_b_in_a, _ = ped._containment_fractions(
                a,
                b,
                tol_days=match_tolerance_days,
            )

            unique_a_count = na - overlap_count
            unique_b_count = nb - overlap_count

            if overlap_count < min_shared_events:
                continue

            if max(frac_a_in_b, frac_b_in_a) < containment_threshold:
                continue

            if not (ped._depth_consistent(a, b, ratio_max=depth_ratio_max) or ped._duration_consistent(a, b, ratio_max=duration_ratio_max)):
                continue


            # if both still have meaningful unique evidence beyond the overlap,
            # do not merge immediately. Let missing-base-period logic handle them.
            if unique_a_count > 1 and unique_b_count > 1:
                continue

            union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)



    # apply winner/loser marking
    for root, idxs in groups.items():
        cluster = [cands[i] for i in idxs]

        members = _add_matching_singles_to_group(
            cluster,
            extra_planet_events,
        )

        missing_base = _infer_missing_base_period_from_cluster(
            members,
            min_support=3,
            perc_tol=0.02,
            max_ratio=5,
            P_min=0.25,
            merge_tol=0.02,
            support_tol_days=shared_t0_tol_days,
        )

        

        # hard preference: if only one member passes signal_not_zero, it wins
        # current_members = [m for m in cluster if getattr(m, "fit_is_current", False)]


        # passing_signal = [m for m in winner_pool if ped._signal_not_zero(m)]

        # if len(passing_signal) == 1:
        #     winner = passing_signal[0]
        # elif len(passing_signal) > 1:
        #     winner = max(passing_signal, key=ped._winner_key)
        # else:
      
    return periodic_events