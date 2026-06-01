import numpy as np
from core.planet_candidate import PlanetCandidate
from stages.search_periodic import PeriodicSearchConfig, run_seed_prepass_full_lc

def keep_unique_period_indices(periods, merge_tol=0.02):
    """
    Keep only the first occurrence of periods that are within merge_tol
    in fractional difference.
    """
    kept = []

    for i, Pi in enumerate(periods):
        keep = True

        for j in kept:
            Pj = periods[j]
            if abs(Pi - Pj) / max(Pj, 1e-12) < merge_tol:
                keep = False
                break

        if keep:
            kept.append(i)

    return kept


def potential_periods_from_single_dt_events(
    t0s,
    min_support=3,
    perc_tol=0.02,
    max_divisor=3,
    P_min=1.0,
    event_tol_days=0.05,
):
    """
    Generate candidate periods from pairwise event separations, allowing
    dt = n * P for n = 1..max_divisor, and keep only periods that actually
    support >= min_support events for some anchor t0.
    """
    t0s = np.asarray(t0s, dtype=float)
    t0s = np.unique(t0s[np.isfinite(t0s)])

    if t0s.size < min_support:
        return np.array([], dtype=float)

    # all positive pairwise separations
    diffs = np.abs(t0s[:, None] - t0s[None, :])
    diffs = np.unique(diffs[(diffs > 0) & np.isfinite(diffs)])

    if diffs.size == 0:
        return np.array([], dtype=float)

    # candidate periods from dt / n, vectorized
    divisors = np.arange(1, max_divisor + 1, dtype=float)
    trial_periods = (diffs[:, None] / divisors[None, :]).ravel()
    trial_periods = trial_periods[np.isfinite(trial_periods)]
    trial_periods = trial_periods[trial_periods > P_min]

    if trial_periods.size == 0:
        return np.array([], dtype=float)

    trial_periods = np.sort(trial_periods)

    # dedup nearby periods
    keep = keep_unique_period_indices(trial_periods, merge_tol=perc_tol)
    trial_periods = trial_periods[keep]

    # keep only periods that actually support >= min_support events
    kept_periods = []
    for P in trial_periods:
        supported = False

        for t0 in t0s:
            resid = ((t0s - t0 + 0.5 * P) % P) - 0.5 * P
            support = np.sum(np.abs(resid) <= event_tol_days)

            if support >= min_support:
                supported = True
                break

        if supported:
            kept_periods.append(P)

    return np.asarray(kept_periods, dtype=float)

def group_events_by_depth(events, ratio_max=1.25, min_group_size=3):
    """
    Group events by similar depth.

    Parameters
    ----------
    events : list
        Objects with attributes:
            - t0_days
            - duration_days
            - depth
    ratio_max : float
        Allowed multiplicative depth spread within a group.
        Example: 1.25 means within ~25%.
    min_group_size : int
        Minimum group size to keep.

    Returns
    -------
    groups : list of dict
        Each dict contains:
            {
                "indices": [... original indices ...],
                "depth": median depth,
                "duration": median duration,
                "t0s": sorted unique event times
            }
    """
    groups = []

    for idx, ev in enumerate(events):
        depth = getattr(ev, "depth", None)
        if depth is None or not np.isfinite(depth):
            continue

        placed = False

        for g in groups:
            d_med = g["depth"]
            frac = abs(depth - d_med) / (abs(d_med) + 1e-6)

            if frac < (ratio_max - 1.0):
                g["indices"].append(idx)

                depths = [
                    events[i].depth
                    for i in g["indices"]
                    if getattr(events[i], "depth", None) is not None
                    and np.isfinite(events[i].depth)
                ]
                g["depth"] = float(np.nanmedian(depths)) if len(depths) else np.nan

                placed = True
                break

        if not placed:
            groups.append({"indices": [idx], "depth": float(depth)})

    out = []
    for g in groups:
        idxs = g["indices"]
        if len(idxs) < min_group_size:
            continue

        t0s = np.array(
            [
                events[i].t0_days
                for i in idxs
                if getattr(events[i], "t0_days", None) is not None
                and np.isfinite(events[i].t0_days)
            ],
            dtype=float,
        )

        if t0s.size < min_group_size:
            continue

        durs = np.array(
            [
                events[i].duration_days
                for i in idxs
                if getattr(events[i], "duration_days", None) is not None
                and np.isfinite(events[i].duration_days)
            ],
            dtype=float,
        )

        out.append(
            {
                "indices": list(idxs),
                "depth": float(g["depth"]) if np.isfinite(g["depth"]) else np.nan,
                "duration": float(np.nanmedian(durs)) if durs.size else np.nan,
                "t0s": np.sort(np.unique(t0s)),
            }
        )

    return out


def seed_periods_from_dt_events(
    events,
    min_support=3,
    depth_ratio_max=1.25,
    perc_tol=0.02,
    max_divisor=3,
    P_min=1.0,
    event_tol_scale=0.25,
    event_tol_floor=0.05,
):
    groups = group_events_by_depth(
        events,
        ratio_max=depth_ratio_max,
        min_group_size=min_support,
    )

    seeds = []

    for g in groups:
        t0s = np.asarray(g["t0s"], dtype=float)
        if t0s.size < min_support:
            continue

        periods = potential_periods_from_single_dt_events(
            t0s,
            min_support=min_support,
            perc_tol=perc_tol,
            max_divisor=max_divisor,
            P_min=P_min,
        )

        if np.isfinite(g["duration"]) and g["duration"] > 0:
            event_tol_days = max(event_tol_floor, event_tol_scale * float(g["duration"]))
        else:
            event_tol_days = event_tol_floor

        # membership signature -> best (shortest-period) representative
        best_for_members = {}

        for P in periods:
            for t0 in t0s:
                resid = ((t0s - t0 + 0.5 * P) % P) - 0.5 * P
                member_mask = np.abs(resid) <= event_tol_days
                member_local_idx = np.where(member_mask)[0]
                support = int(member_local_idx.size)

                if support < min_support:
                    continue

                # convert local group indices -> original event indices
                member_global_idx = tuple(int(g["indices"][k]) for k in member_local_idx)

                row = {
                    "P": float(P),
                    "t0_seed": float(t0),
                    "support": support,
                    "member_indices": list(member_global_idx),
                    "group_indices": list(g["indices"]),
                    "depth": float(g["depth"]) if np.isfinite(g["depth"]) else np.nan,
                    "duration": float(g["duration"]) if np.isfinite(g["duration"]) else np.nan,
                    "t0s": [float(x) for x in g["t0s"]],
                }

            
                # keep shortest P for this exact member set
                if member_global_idx not in best_for_members:
                    best_for_members[member_global_idx] = row
                else:
                    if row["P"] < best_for_members[member_global_idx]["P"]:
                        best_for_members[member_global_idx] = row

        seeds.extend(best_for_members.values())

    # optional final sort: strongest support first, then shorter period
    seeds.sort(key=lambda x: (-x["support"], x["P"]))
    return seeds


def folded_transit_snr(
    time,
    flux,
    period_days,
    t0_days,
    duration_days,
    expected_depth,
    oot_window_days,
):
    """
    Compute folded SNR for a trial (P, T0), using a supplied expected depth.

    SNR = expected_depth / local_scatter * sqrt(N_in_transit)
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if time.size == 0 or flux.size == 0:
        return np.nan
    if not np.isfinite(period_days) or period_days <= 0:
        return np.nan
    if not np.isfinite(t0_days):
        return np.nan
    if not np.isfinite(duration_days) or duration_days <= 0:
        return np.nan
    if not np.isfinite(expected_depth) or expected_depth <= 0:
        return np.nan
    if not np.isfinite(oot_window_days) or oot_window_days <= 0:
        return np.nan

    phase = ((time - t0_days + 0.5 * period_days) % period_days) - 0.5 * period_days

    in_transit = np.abs(phase) <= 0.5 * duration_days
    local_oot = (np.abs(phase) <= oot_window_days) & (~in_transit)

    n_in = int(np.sum(in_transit))
    n_oot = int(np.sum(local_oot))

    if n_in == 0 or n_oot < 5:
        return np.nan

    oot_flux = flux[local_oot]
    oot_med = np.median(oot_flux)
    scatter = 1.4826 * np.median(np.abs(oot_flux - oot_med))

    if not np.isfinite(scatter) or scatter <= 0:
        return np.nan

    return float(expected_depth / scatter * np.sqrt(n_in))


def best_t0_and_snr_for_period(
    time,
    flux,
    period_days,
    trial_t0s,
    duration_days,
    expected_depth,
    oot_window_days=None,
):
    """
    For one trial period, test each event t0 as a possible anchor and keep
    the one with the best folded SNR.

    Returns
    -------
    best_t0, best_snr
    """
    trial_t0s = np.asarray(trial_t0s, dtype=float)
    trial_t0s = trial_t0s[np.isfinite(trial_t0s)]

    if trial_t0s.size == 0:
        return np.nan, np.nan

    if oot_window_days is None:
        if np.isfinite(duration_days) and duration_days > 0:
            oot_window_days = max(0.1, 2.0 * duration_days)
        else:
            oot_window_days = 0.1

    best_t0 = np.nan
    best_snr = np.nan

    for t0 in trial_t0s:
        snr = folded_transit_snr(
            time=time,
            flux=flux,
            period_days=period_days,
            t0_days=t0,
            duration_days=duration_days,
            expected_depth=expected_depth,
            oot_window_days=oot_window_days,
        )

        if not np.isfinite(snr):
            continue

        if not np.isfinite(best_snr) or snr > best_snr:
            best_snr = float(snr)
            best_t0 = float(t0)

    return best_t0, best_snr


def matching_event_indices_for_period(
    events,
    candidate_indices,
    period_days,
    t0_days,
    event_tol_days = 0.05,
):
    """
    Return which of candidate_indices are phase-consistent with a chosen (P, T0).
    """
    candidate_indices = list(candidate_indices)
    if len(candidate_indices) == 0:
        return np.array([], dtype=int)

    valid_indices = []
    t0s = []

    for i in candidate_indices:
        t0 = getattr(events[i], "t0_days", None)
        if t0 is None or not np.isfinite(t0):
            continue
        valid_indices.append(i)
        t0s.append(t0)

    if len(t0s) == 0:
        return np.array([], dtype=int)

    t0s = np.asarray(t0s, dtype=float)
    resid = ((t0s - t0_days + 0.5 * period_days) % period_days) - 0.5 * period_days
    members = np.where(np.abs(resid) <= event_tol_days)[0]
    print('memebers - fail? ', len(members), members)
    return np.array([valid_indices[k] for k in members], dtype=int)


def candidate_from_indices(
    events,
    member_indices,
    period_days,
    t0_days,
    source,
    notes_prefix="",
):
    """
    Build a periodic PlanetCandidate from selected member event indices.

    Intended to be called only after the periodic model fit has passed.
    """
    member_indices = np.asarray(member_indices, dtype=int)

    if member_indices.size == 0:
        return None, None

    member_events = [events[i] for i in member_indices]

    dur_vals = [
        e.duration_days
        for e in member_events
        if getattr(e, "duration_days", None) is not None
        and np.isfinite(e.duration_days)
    ]
    dep_vals = [
        e.depth
        for e in member_events
        if getattr(e, "depth", None) is not None
        and np.isfinite(e.depth)
    ]
    t0_vals = [
        e.t0_days
        for e in member_events
        if getattr(e, "t0_days", None) is not None
        and np.isfinite(e.t0_days)
    ]

    if len(dur_vals) == 0 or len(dep_vals) == 0:
        return None, None

    dur_est = float(np.nanmedian(dur_vals))
    dep_est = float(np.nanmedian(dep_vals))
    support = int(member_indices.size)

    pc = PlanetCandidate(
        ptype="Periodic",
        t0_days=float(t0_days),
        period_days=float(period_days),
        duration_days=dur_est,
        depth=dep_est,
        n_transits_obs=support,
        transit_times_days=[float(x) for x in t0_vals],
        source=source,
        notes=f"{notes_prefix}support={support}",
    )
    print('candidates indices fail? ', pc, member_indices)

    return pc, member_indices


def mark_single_members_consumed(
    single_candidates,
    member_indices,
    periodic_candidate_id,
    note_prefix="promoted_into=",
):
    """
    Mutate singles in-place AFTER a successful periodic fit.
    """
    for i in member_indices:
        sc = single_candidates[i]
        sc.default = False
        cur = sc.notes or ""
        add = f"{note_prefix}{periodic_candidate_id}"
        sc.notes = (cur + "; " + add) if cur else add






####### From previous seed stuff

def same_harmonic_family(P1, P2, max_multiple=5, rel_tol=0.02):
    """
    True if P1 and P2 are approximately integer multiples of each other
    up to max_multiple.
    Examples:
        P and 2P -> same family
        P and 3P -> same family
    """
    P1 = float(P1)
    P2 = float(P2)

    if not np.isfinite(P1) or not np.isfinite(P2) or P1 <= 0 or P2 <= 0:
        return False

    ratio = max(P1, P2) / min(P1, P2)
    k = int(round(ratio))

    if k < 1 or k > max_multiple:
        return False

    return abs(ratio - k) / max(k, 1) <= rel_tol


def choose_best_seed_row_per_depth_group(scored_seed_rows, max_multiple=3, rel_tol=0.02):
    """
    Within each depth group (same group_indices), cluster harmonically related
    periods together and keep the highest-BLS-SNR representative of each family.

    Ranking:
      1) must pass seeded BLS
      2) higher seeded-BLS SNR
      3) shorter period as tie-breaker
    """
    # outer grouping: same depth group
    grouped = defaultdict(list)
    for row in scored_seed_rows:
        depth_key = tuple(sorted(row["group_indices"]))
        grouped[depth_key].append(row)

    winner_rows = []

    for depth_key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: float(r["P"]))

        # cluster harmonic relatives inside this depth group
        families = []

        for row in rows:
            placed = False


            for fam in families:
                if any(
                    same_harmonic_family(row["P"], other["P"], max_multiple=max_multiple, rel_tol=rel_tol)
                    for other in fam
                ):
                    fam.append(row)
                    placed = True
                    break


            if not placed:
                families.append([row])
            print(depth_key, )

        print(f"depth group {depth_key}: n_rows={len(rows)}, n_families={len(families)}, which are {[set([row['P'] for row in fam]) for fam in families]}")

        # keep one winner per harmonic family
        for fam in families:
            fam_passed = [r for r in fam if r["bls_passed"]]
            if len(fam_passed) == 0:
                continue

            winner = max(
                fam_passed,
                key=lambda r: (float(r["bls_snr"]), -float(r["P"]))
            )
            winner_rows.append(winner)

    return winner_rows


def load_lightcurve_for_periodic_search(target, flavour):
    total_csv = find_total_csv(target.root_dir, flavour)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])

    time = df["TIME"].to_numpy(float)
    flux = df["FLUX"].to_numpy(float)

    if "FLUX_ERR" in df.columns:
        flux_err = df["FLUX_ERR"].to_numpy(float)
    else:
        flux_err = np.full_like(flux, np.nanstd(flux))

    return total_csv, time, flux, flux_err


def score_unique_seed_periods(time, flux, flux_err, periods, flavour):
    """
    Score each distinct period once with seeded BLS.
    Returns:
        period_score_cache[P] = {
            "passed": bool,
            "snr": float,
            "period": float,
        }
    """
    seed_cfg = PeriodicSearchConfig(flavour=flavour)
    seed_cfg.use_seed_periods = True
    seed_cfg.max_iters = 1

    period_score_cache = {}

    for P in periods:
        P = float(P)

        accepted_events, _ = run_seed_prepass_full_lc(
            time,
            flux,
            flux_err,
            cfg=seed_cfg,
            seed_periods=[P],
            accepted_events=[],
            intransit=np.zeros_like(time, dtype=bool),
        )

        if len(accepted_events) > 0:
            ev = accepted_events[0]
            period_score_cache[P] = {
                "passed": True,
                "snr": float(getattr(ev, "snr", -np.inf)),
                "period": P,
            }
        else:
            period_score_cache[P] = {
                "passed": False,
                "snr": -np.inf,
                "period": P,
            }

    return period_score_cache


def attach_scores_to_seed_rows(seed_rows, period_score_cache):
    """
    Add seeded-BLS score info back onto every seed row.
    """
    scored_seed_rows = []

    for row in seed_rows:
        P = float(row["P"])
        score_info = period_score_cache.get(
            P,
            {
                "passed": False,
                "snr": -np.inf,
                "period": P,
            },
        )

        row2 = dict(row)
        row2["bls_passed"] = bool(score_info["passed"])
        row2["bls_snr"] = float(score_info["snr"])
        scored_seed_rows.append(row2)

    return scored_seed_rows

def choose_best_seed_row_per_depth_group(scored_seed_rows, max_multiple=3, rel_tol=0.02):
    """
    Within each depth group (same group_indices), cluster harmonically related
    periods together and keep the highest-BLS-SNR representative of each family.
    """
    grouped = defaultdict(list)
    for row in scored_seed_rows:
        depth_key = tuple(sorted(row["group_indices"]))
        grouped[depth_key].append(row)

    winner_rows = []

    for depth_key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: float(r["P"]))

        families = []
        for row in rows:
            placed = False

            for fam in families:
                if any(
                    same_harmonic_family(
                        row["P"], other["P"],
                        max_multiple=max_multiple,
                        rel_tol=rel_tol,
                    )
                    for other in fam
                ):
                    fam.append(row)
                    placed = True
                    break

            if not placed:
                families.append([row])

        fam_summary = [
            {round(float(r["P"]), 12) for r in fam}
            for fam in families
        ]

        print(
            f"depth group {depth_key}: "
            f"n_rows={len(rows)}, n_families={len(families)}, "
            f"which are {fam_summary}"
        )

        for fam in families:
            fam_passed = [r for r in fam if r["bls_passed"]]
            if len(fam_passed) == 0:
                continue

            winner = max(
                fam_passed,
                key=lambda r: (float(r["bls_snr"]), -float(r["P"]))
            )
            winner_rows.append(winner)

    return winner_rows

def choose_best_seed_row_per_group(scored_seed_rows):
    """
    Group by the exact member set explained by the seed row:

        key = tuple(sorted(row["member_indices"]))

    and keep the best-scoring period per group.
    Ranking:
      1) passed seeded BLS
      2) higher seeded-BLS SNR
      3) shorter period as tie-breaker
    """
    grouped = defaultdict(list)
    for row in scored_seed_rows:
        key = tuple(sorted(row["member_indices"]))
        grouped[key].append(row)

    best_by_key = {}

    for key, rows in grouped.items():
        best_row = None
        best_score = None

        for row in rows:
            score = (
                int(bool(row["bls_passed"])),
                float(row["bls_snr"]),
                -float(row["P"]),
            )

            if best_row is None or score > best_score:
                best_row = row
                best_score = score

        if best_row is not None and best_row["bls_passed"]:
            best_by_key[key] = best_row

    return list(best_by_key.values()), grouped


def summarize_groups(grouped):
    print("n groups:", len(grouped))
    for i, (key, rows) in enumerate(grouped.items()):
        periods = sorted(set(round(float(r["P"]), 6) for r in rows))
        print(
            f"group {i}: members={key}, "
            f"n_rows={len(rows)}, "
            f"n_unique_periods={len(periods)}"
        )