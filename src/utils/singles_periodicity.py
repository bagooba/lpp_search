# utils/singles_periodicity.py
import numpy as np
from core.planet_candidate import PlanetCandidate


# --- paste your existing functions here ---
def prepping_singles_for_periodic_check(
    t0_singles,
    durations=None, depths=None,
    max_missed_transits=7, P_min=0.25, P_max=None,
    phase_win=None, rel_merge=0.01, min_support=3
):
    t0_all = np.asarray(t0_singles, dtype=float)
    t0, idx = np.unique(t0_all, return_index=True)
    if t0.size < 2:
        return t0, None, None, [], (phase_win if phase_win is not None else 0.05)

    dur = (np.asarray(durations, dtype=float)[idx]
           if (durations is not None and len(durations) == len(t0_singles)) else None)
    dep = (np.asarray(depths, dtype=float)[idx]
           if (depths is not None and len(depths) == len(t0_singles)) else None)

    baseline = float(t0[-1] - t0[0])
    P_max_eff = float(P_max) if (P_max is not None) else max(1.0, baseline if (min_support <= 2) else (baseline / 2.0))

    if phase_win is None:
        if (dur is not None) and np.isfinite(dur).any():
            phase_win_eff = max(0.02, 0.4 * float(np.nanmedian(dur)))
        else:
            phase_win_eff = 0.04
    else:
        phase_win_eff = float(phase_win)

    diffs = np.array([t0[j] - t0[i] for i in range(t0.size) for j in range(i + 1, t0.size)], dtype=float)
    diffs = diffs[diffs > 0]

    cand = []
    for delta in diffs:
        for m in range(1, max_missed_transits + 1):
            P = delta / m
            if (P_min <= P <= P_max_eff):
                cand.append(P)
    if not cand:
        return t0, dur, dep, [], phase_win_eff

    cand = np.sort(np.array(cand, dtype=float))
    groups, cur = [], [cand[0]]
    for v in cand[1:]:
        if abs(v - np.median(cur)) / max(v, 1e-6) <= rel_merge:
            cur.append(v)
        else:
            groups.append(np.array(cur, dtype=float)); cur = [v]
    groups.append(np.array(cur, dtype=float))
    return t0, dur, dep, groups, phase_win_eff


def score_once_modes(
    t0, dur, dep, groups, phase_win,
    min_support=3,
    use_depth=True, depth_zmax=2.5, depth_ratio_max=1.75, depth_floor=5e-5,
    local_span=0.02, local_n=41,
    windows=None,
    allow_missed=2,              # <-- NEW: set 0 for your "hard" principle
    prefer_smaller_P_on_tie=True # <-- NEW: your stated preference
):
    """
    Score candidate period groups by folding event times and counting aligned members.

    Key logic change (coverage-aware, hard-veto):
      - If windows is provided, compute 'expected' epochs in-window for (T0_mode, P).
      - Require expected - support <= allow_missed.
        For your cadence-based principle, use allow_missed=0.

    Parameters
    ----------
    t0 : array-like
        Event times (days) for *all* singles.
    dur, dep : arrays or None
        Durations/depths aligned with t0 (optional).
    groups : list of arrays
        Candidate period clusters (from deltaT/m generation + grouping).
    phase_win : float
        Half-width in days for membership around the phase center.
    windows : array-like shape (K,2) or None
        Observed coverage windows [(start,end), ...] in days.
    allow_missed : int
        Allowed number of in-window expected epochs that are not present as events.
        Set to 0 to enforce: "if it should have been observed, it doesn't exist".
    prefer_smaller_P_on_tie : bool
        If True, break exact ties by preferring smaller P.

    Returns
    -------
    out : list of dict
        Each dict has keys: P, T0, support, members, phase_rms, depth_med, depth_scat
    """
    out = []
    t0 = np.asarray(t0, dtype=float)
    if t0.size < min_support or len(groups) == 0:
        return out

    # Ensure windows array shape is sane if provided
    if windows is not None:
        windows = np.asarray(windows, dtype=float)
        if windows.ndim != 2 or windows.shape[1] != 2:
            raise ValueError("windows must be an array of shape (K,2) with (start,end) rows")

    for g in groups:
        P0 = float(np.median(g))
        grid = P0 * np.linspace(1.0 - local_span, 1.0 + local_span, local_n)

        best = None
        best_members = None

        for P in grid:
            if not (np.isfinite(P) and P > 0):
                continue

            # --- phase folding relative to an arbitrary anchor (t0[0]) ---
            phases = (t0 - t0[0]) % P
            phases = np.where(phases > P/2, phases - P, phases)

            # Choose phase-center robustly
            center = np.median(phases)

            # Residuals around center, wrapped to [-P/2, P/2]
            resid = (phases - center) % P
            resid = np.where(resid > P/2, resid - P, resid)

            members = np.where(np.abs(resid) <= phase_win)[0]
            support = int(members.size)
            if support < min_support:
                continue

            # --- coverage-aware hard veto: expected must match observed (support) ---
            if windows is not None:
                T0_mode = float(t0[0] + center)

                expected = 0
                for start, end in windows:
                    # integer k such that start <= T0 + kP <= end
                    k0 = int(np.ceil((start - T0_mode) / P))
                    k1 = int(np.floor((end   - T0_mode) / P))
                    if k1 >= k0:
                        expected += (k1 - k0 + 1)

                missed = expected - support

                # Your principle: allow_missed=0
                if missed > allow_missed:
                    continue

                # (Optional sanity: if missed < 0 something inconsistent happened)
                # if missed < 0: continue

            # --- depth consistency (unchanged) ---
            if use_depth and (dep is not None) and (members.size > 1):
                dep = np.asarray(dep, dtype=float)
                d_sup = dep[members]
                d_med = float(np.median(d_sup))
                scat  = 1.4826 * np.median(np.abs(d_sup - d_med))  # MAD

                if scat <= depth_floor:
                    dmax, dmin = float(np.max(d_sup)), float(np.min(d_sup))
                    if (dmax / max(dmin, depth_floor)) > depth_ratio_max:
                        continue
                else:
                    z = np.abs(d_sup - d_med) / scat
                    if np.any(z > depth_zmax):
                        continue

                depth_penalty = scat
            else:
                d_med, depth_penalty = (np.nan, 0.0)

            # --- score ---
            phase_rms = float(np.sqrt(np.mean(resid[members]**2)))

            # Primary: support. Then: tightness. Then: depth consistency.
            # Tie-break: your preference (smaller P).
            if prefer_smaller_P_on_tie:
                key = (support, -phase_rms, -depth_penalty, -float(P))
            else:
                key = (support, -phase_rms, -depth_penalty)

            if (best is None) or (key > best[0]):
                best = (key, float(P), float(center), support, phase_rms, d_med, depth_penalty)
                best_members = members

        if best is None:
            continue

        _, P_star, center, support, phase_rms, d_med, depth_pen = best

        out.append({
            "P": float(P_star),
            "T0": float(t0[0] + center),
            "support": int(support),
            "members": np.array(best_members, dtype=int),
            "phase_rms": float(phase_rms),
            "depth_med": float(d_med),
            "depth_scat": float(depth_pen) if np.isfinite(depth_pen) else np.nan,
        })

    # Keep your original final sort behavior
    out.sort(key=lambda d: (-d["support"], d["phase_rms"]))
    return out


def extract_all_modes_iterative(
    t0_init, dur_init, dep_init,
    min_support=3, prep_kwargs=None, scorer_kwargs=None
):
    prep_kwargs = prep_kwargs or {}
    scorer_kwargs = scorer_kwargs or {}

    accepted = []
    t0 = np.array(t0_init, dtype=float)
    dur = None if dur_init is None else np.array(dur_init, dtype=float)
    dep = None if dep_init is None else np.array(dep_init, dtype=float)

    while True:
        t0_w, dur_w, dep_w, groups, phase_win = prepping_singles_for_periodic_check(
            t0, dur, dep, min_support=min_support, **prep_kwargs
        )
        if (t0_w.size < min_support) or (len(groups) == 0):
            break


        candidates = score_once_modes(
            t0_w, dur_w, dep_w, groups, phase_win,
            min_support=min_support, **scorer_kwargs
        )
        if len(candidates) == 0:
            break

        best = candidates[0]
        accepted.append(best)

        keep = np.ones(t0_w.size, dtype=bool)
        keep[best['members']] = False
        t0 = t0_w[keep]
        dur = None if dur_w is None else dur_w[keep]
        dep = None if dep_w is None else dep_w[keep]

        if t0.size < min_support:
            break

    return accepted



def periodic_modes_from_dt_events(
    events,
    *,
    min_support=3,
    max_missed_transits=7,
    P_min=0.25,
    P_max=None,
    rel_merge=0.01,
    use_depth=True,
    local_span=0.02,
    local_n=41,
    windows=None,   # <-- add
):

    """
    Return periodic modes inferred from DT TransitEvent objects.
    Each mode is a dict with keys like P, T0, support, members, phase_rms, ...
    """
    if events is None or len(events) < min_support:
        return []

    t0 = np.array([float(e.t0_days) for e in events], dtype=float)
    dur = np.array([float(e.duration_days) for e in events], dtype=float)
    dep = np.array([float(e.depth) for e in events], dtype=float)

    modes = extract_all_modes_iterative(
        t0, dur, dep,
        min_support=min_support,
        prep_kwargs={
            "max_missed_transits": max_missed_transits,
            "P_min": P_min,
            "P_max": P_max,
            "rel_merge": rel_merge,
        },
        scorer_kwargs={
            "use_depth": use_depth,
            "local_span": local_span,
            "local_n": local_n,
            "windows": windows,   # <-- pass through
        }
    )
    return modes


def seed_periods_from_dt_events(
    events,
    *,
    top_k=10,
    min_support=3,
    windows=None,   # <-- add
    **mode_kwargs
):
    modes = periodic_modes_from_dt_events(
        events,
        min_support=min_support,
        windows=windows,    # <-- pass through
        **mode_kwargs
    )
    return [float(m["P"]) for m in modes[:top_k]]


def candidate_from_mode(mode, events, *, source, notes_prefix=""):
    """
    Convert one periodic mode (P/T0/members) + DT events into a Periodic PlanetCandidate.
    Returns (PlanetCandidate, member_indices) or (None, None).
    """
    members = mode.get("members", [])
    if members is None or len(members) == 0:
        return None, None

    member_events = [events[i] for i in members]
    dur_vals = [e.duration_days for e in member_events if e.duration_days is not None]
    dep_vals = [e.depth for e in member_events if e.depth is not None]
    if len(dur_vals) == 0 or len(dep_vals) == 0:
        return None, None

    dur_est = float(np.nanmedian(dur_vals))
    dep_est = float(np.nanmedian(dep_vals))
    if not (np.isfinite(dur_est) and np.isfinite(dep_est)):
        return None, None

    P = float(mode["P"])
    T0 = float(mode["T0"])
    support = int(mode.get("support", len(member_events)))
    transit_times_days = [event.t0_days for event in member_events]
    pc = PlanetCandidate(
        ptype="Periodic",
        t0_days=T0,
        period_days=P,
        duration_days=dur_est,
        depth=dep_est,
        n_transits_obs=support,
        source=source,
        notes=f"{notes_prefix}support={support}",
        
    )
    return pc, np.array(members, dtype=int)

def periodic_candidates_from_modes(modes, events, *, source, min_support=3, notes_prefix=""):
    out = []
    for m in modes:
        if int(m.get("support", 0)) < min_support:
            continue
        pc, members = candidate_from_mode(m, events, source=source, notes_prefix=notes_prefix)
        if pc is not None:
            out.append((pc, members))
    return out


def mark_single_members_consumed(single_candidates, member_indices, periodic_candidate_id, note_prefix="promoted_into="):
    """
    Mutate singles in-place AFTER a successful periodic fit:
      - default=False
      - notes append promoted_into=<candidate_id>
    """
    for i in member_indices:
        sc = single_candidates[i]
        sc.default = False
        cur = sc.notes or ""
        add = f"{note_prefix}{periodic_candidate_id}"
        sc.notes = (cur + "; " + add) if cur else add

def single_matches_periodic(s_t0, periodic_candidates, time_min, time_max, fixed_tol_days=0.05):
    """
    Return True if a single event time matches ANY epoch of fitted periodic candidates.
    """
    s_t0 = float(s_t0)

    for p in periodic_candidates:
        if not getattr(p, "fit_is_current", False):
            continue

        P  = p.period_days
        t0 = p.t0_days
        dur = p.duration_days

        if P is None or t0 is None:
            continue

        P = float(P); t0 = float(t0)

        if not np.isfinite(P) or P <= 0:
            continue

        # predict transit times in window
        epochs = _epochs_in_window(time_min, time_max, t0, P)

        # tolerance: max(fixed, 0.25 * duration)
        if dur is not None and np.isfinite(dur):
            tol = max(fixed_tol_days, 0.25 * float(dur))
        else:
            tol = fixed_tol_days

        # check match
        if np.min(np.abs(epochs - s_t0)) <= tol:
            return True

    return False