#!/usr/bin/env python
import glob
import os
from runpy import run_path
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.target import Target, PipelineStage
from core.planet_candidate import PlanetCandidate
from core.transit_event import TransitEvent
from core.periodic_event import PeriodicEvent

from utils.find_total_csv import find_total_csv
from utils.run_json import upsert_run_json, append_run_json_list
from utils.singles_periodicity import (
    seed_periods_from_dt_events,
    best_t0_and_snr_for_period,
    matching_event_indices_for_period,
    candidate_from_indices,
    mark_single_members_consumed,
    keep_unique_period_indices,
    score_unique_seed_periods,
    attach_scores_to_seed_rows,
    choose_best_seed_row_per_depth_group,
)

import utils.periodic_event_dedup as ped

from utils.alias_dedup import alias_dedup_periodic_candidates
from utils.queue import enqueue

from stages.search_singles import singles_search, SinglesSearchConfig
from stages.search_periodic import run_seed_prepass_full_lc, PeriodicSearchConfig

from utils.handling_data import normalize_depth_to_fractional
from engines.pyMC_core import pymc_fit_candidate, write_converged_fit_csv
from utils.check_singles_recovery import check_singles_against_periodic_candidate

TARGET_GLOB = "../../toi_data/target_*"   # adjust


# ---------------------------
# I/O helpers
# ---------------------------


def overwrite_run_json_keys(run_path: Path, updates: dict) -> None:
    d = json.loads(run_path.read_text())
    d.update(updates)
    run_path.write_text(json.dumps(d, indent=2, sort_keys=True))

def load_run_json(run_path: Path) -> dict:
    return json.loads(run_path.read_text())

def write_final_candidates_csv(target: Target, candidates: list[PlanetCandidate]) -> Path:
    out_path = target.root_dir / "final_candidates.csv"

    rows = []
    for c in candidates:
        rows.append({
            "ticid": target.ticid,
            "gaia_id": target.gaia_id,
            "candidate_id": c.candidate_id(),
            "ptype": c.ptype,
            "t0_days": c.t0_days,
            "period_days": c.period_days,
            "duration_days": c.duration_days,
            "depth": c.depth,

            "rp_rs": getattr(c, "rp_rs", None),
            "cosi": getattr(c, "cosi", None),
            "a_smaj": getattr(c, "a_smaj", None),
            "n_transits_obs": c.n_transits_obs,
            "fit_is_current": c.fit_is_current,
            "source": c.source,
            "default": c.default,
            "notes": c.notes,
            # store the full summary stats as JSON text so you keep *everything*
            "pymc_summary_json": json.dumps(c.pymc_summary) if c.pymc_summary else "",
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def _epochs_in_window(tmin, tmax, t0, P):
    # predict epochs in [tmin, tmax]
    k0 = int(np.floor((tmin - t0) / P)) - 1
    k1 = int(np.ceil((tmax - t0) / P)) + 1
    ks = np.arange(k0, k1 + 1)
    return t0 + ks * P



def single_matches_periodic(s_t0, periodic_candidates, time_min, time_max, fixed_tol_days=0.1):
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


def _pick_family_winner(members: list[PlanetCandidate]) -> PlanetCandidate:
    current_members = [m for m in members if getattr(m, "fit_is_current", False)]

    if len(current_members) > 0:
        winner_pool = current_members
    else:
        winner_pool = members

    passing_signal = [m for m in winner_pool if ped._signal_not_zero(m)]

    if len(passing_signal) == 1:
        return passing_signal[0]
    elif len(passing_signal) > 1:
        return max(passing_signal, key=ped._winner_key)
    else:
        return max(winner_pool, key=ped._winner_key)


def _observed_support_count(c: PlanetCandidate) -> int:
    tt = getattr(c, "transit_times_days", None)
    if tt is None:
        return 0
    return int(len(tt))


def _should_promote_missing_base_refit(
    refit_candidate: PlanetCandidate,
    old_family_members: list[PlanetCandidate],
) -> tuple[bool, PlanetCandidate, str]:
    old_winner = _pick_family_winner(old_family_members)

    if not getattr(refit_candidate, "fit_is_current", False):
        return False, old_winner, "refit is not current"

    if not ped._signal_not_zero(refit_candidate):
        return False, old_winner, "refit signal is zero"

    # if not (
    #     ped._duration_consistent(refit_candidate, old_winner)
    #     or ped._depth_consistent(refit_candidate, old_winner)
    # ):
    #     return False, old_winner, "refit morphology not consistent with previous family winner"

    new_support = _observed_support_count(refit_candidate)
    old_support = _observed_support_count(old_winner)

    if new_support < old_support:
        return False, old_winner, "refit explains fewer observed transits"

    if new_support > old_support:
        return True, old_winner, "refit explains more observed transits"

    if ped._winner_key(refit_candidate) > ped._winner_key(old_winner):
        return True, old_winner, "same support, refit wins on winner key"

    return False, old_winner, "previous family winner preferred"




# def consume_singles_under_periodics(final_candidates, time_min, time_max, fixed_tol_days=0.05):
#     periodics = [c for c in final_candidates if c.ptype == "Periodic" and c.fit_is_current]
#     singles   = [c for c in final_candidates if c.ptype == "Single"]

#     for s in singles:
#         if s.t0_days is None:
#             continue
#         s_t0 = float(s.t0_days)
#         best = None

#         for p in periodics:
#             if p.period_days is None or p.t0_days is None:
#                 continue
#             P = float(p.period_days); t0 = float(p.t0_days)
#             if not np.isfinite(P) or P <= 0:
#                 continue

#             epochs = _epochs_in_window(time_min, time_max, t0, P)
#             # tolerance = max(0.25 * dur, fixed)
#             dur = float(p.duration_days) if p.duration_days is not None else np.nan
#             tol = max(fixed_tol_days, 0.25 * dur) if np.isfinite(dur) else fixed_tol_days

#             if np.min(np.abs(epochs - s_t0)) <= tol:
#                 best = p
#                 break

#         if best is not None:
#             # mark consumed (don’t delete)
#             s.default = False
#             s.notes = (s.notes + f" consumed_by={best.candidate_id()};").strip()
#             # optionally add attribute for CSV
#             setattr(s, "consumed_by", best.candidate_id())

#     return final_candidates

def append_global_candidates_csv(candidates: list[PlanetCandidate], target: Target, global_path: Path) -> None:
    # NOTE: if you run this in parallel across many targets, concurrent appends can collide.
    # In that case, write per-target CSVs and merge later in one job.
    rows = []
    for c in candidates:
        rows.append({
            "ticid": target.ticid,
            "gaia_id": target.gaia_id,
            "candidate_id": c.candidate_id(),
            "ptype": c.ptype,
            "t0_days": c.t0_days,
            "period_days": c.period_days,
            "duration_days": c.duration_days,
            "depth": c.depth,
            "rp_rs": getattr(c, "rp_rs", None),
            "cosi": getattr(c, "cosi", None),
            "a_smaj": getattr(c, "a_smaj", None),
            "n_transits_obs": c.n_transits_obs,
            "fit_is_current": c.fit_is_current,
            "source": c.source,
            "default": c.default,
            "notes": c.notes,
            "pymc_summary_json": json.dumps(c.pymc_summary) if c.pymc_summary else "",
        })

    df = pd.DataFrame(rows)
    write_header = not global_path.exists()
    df.to_csv(global_path, mode="a", header=write_header, index=False)

def resolve_periodic_conflicts(candidates, tol=0.35):

    candidates = sorted(
        candidates,
        key=lambda c: getattr(c, "snr", -np.inf),
        reverse=True,
    )

    final = []
    used_times = []

    for c in candidates:
        t0s = np.array(c.transit_times_days or [])

        conflict = False
        for ut in used_times:
            if np.any(np.abs(t0s - ut) < tol):
                conflict = True
                break

        if conflict:
            continue

        final.append(c)
        
        used_times.extend([float(x) for x in t0s])


    return final
# ---------------------------
# Fit helpers
# ---------------------------
def fit_and_attach(target: Target, cand: PlanetCandidate, time, flux, unc, run_path: Path, verbose: bool=False) -> bool:
    """
    Runs PyMC fit and attaches *full summary stats* to the candidate.
    """
    print('trying fit')
    attempt_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    upsert_run_json(run_path, {"status": {"stage": "pymc_fit", "state": "running", "attempt_id": attempt_id}})

    summary_df, ok, fit_info = pymc_fit_candidate(target, cand, time, flux, unc, verbose=verbose)
    
    print('ok? ', ok)
    
    if ok and summary_df is not None:
        cand.pymc_summary = summary_df.to_dict(orient="index")
        print('candidate pymc summary', cand.pymc_summary)
        cand.mark_fitted()
        print('summary')
        print(summary_df)

        stats_csv = write_converged_fit_csv(target, cand, fit_info)
        print(f"wrote converged fit stats to {stats_csv}")


        # Update working hypothesis from PyMC medians
        cand.t0_days = _summary_median(cand, "t0", fallback=cand.t0_days)
        if cand.ptype == "Periodic":
            cand.period_days = _summary_median(cand, "Per", fallback=cand.period_days)
        cand.duration_days = _summary_median(cand, "dur", fallback=cand.duration_days)

        raw_depth = _summary_median(cand, "depth", fallback=cand.depth)
        cand.depth = normalize_depth_to_fractional(raw_depth)
        cand.rp_rs = _summary_median(cand, "rp_rs", fallback=np.sqrt(cand.depth))    
        cand.cosi = _summary_median(cand, "cosi", fallback=None)    
        cand.a_smaj = _summary_median(cand, "a_rs", fallback=None)    

    else:
        cand.fit_is_current = False

    append_run_json_list(run_path, "fit_attempts", {
        "attempt_id": attempt_id,
        "candidate_id": cand.candidate_id(),
        "ptype": cand.ptype,
        "ok": bool(ok),
        "finished_at": datetime.now().isoformat()
    })
    snr = _summary_median(cand, "SNR", fallback=0)
    passed = bool(ok and snr > 10.0)

    if not passed:
        cand.fit_is_current = False

    upsert_run_json(run_path, {"status": {"stage": "pymc_fit", "state": "done", "attempt_id": attempt_id}})
    return passed

def finalize_pass1_singles_only(target, run_path, run_json):
    # print('only pass1 singles')
    raw_pass1 = run_json.get("dt_events_raw_pass1", [])
    pass1_events = [TransitEvent.from_dict(d) for d in raw_pass1] if isinstance(raw_pass1, list) else []

    flavour = target.data_source.value
    total_csv = find_total_csv(target.root_dir, flavour)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])
    time = df["TIME"].to_numpy(float)
    flux = df["FLUX"].to_numpy(float)
    unc = df["FLUX_ERR"].to_numpy(float) if "FLUX_ERR" in df.columns else np.full_like(flux, np.nanstd(flux))

    single_candidates = []
    print(len(pass1_events))
    for ev in pass1_events:
        sc = PlanetCandidate(
            ptype="Single",
            t0_days=float(ev.t0_days),
            period_days=None,
            duration_days=float(ev.duration_days),
            depth=normalize_depth_to_fractional(ev.depth),
            source="DT_pass1",
        )
        fit_and_attach(target, sc, time, flux, unc, run_path, verbose=False)
        single_candidates.append(sc)

    return pass1_events, single_candidates

def _summary_median(cand, varname, fallback=None):
    """
    cand.pymc_summary is a dict-of-dicts from summary_df.to_dict().
    Expected structure: cand.pymc_summary["Per"]["median"], etc.
    """
    try:
        d = cand.pymc_summary[varname]
        if isinstance(d, dict) and "median" in d.keys():
            return float(d["median"])
    except Exception:
        pass
    return fallback


def periodic_mask_from_fitted_candidate(time: np.ndarray, cand: PlanetCandidate,
                                        buffer_days: float = 0.5) -> np.ndarray:
    """
    Build a full-length in-transit mask using PyMC medians:
      - Per median
      - t0 median
      - dur median (PyMC deterministic, full duration in days)
    """

    # Pull fitted medians
    P  = _summary_median(cand, "Per", fallback=cand.period_days)
    t0 = _summary_median(cand, "t0", fallback=cand.t0_days)

    # Your PyMC model defines pm.Deterministic("dur", ...)
    dur = _summary_median(cand, "dur", fallback=cand.duration_days)

    if P is None or dur is None or t0 is None:
        return np.zeros_like(time, dtype=bool)

    P = float(P); t0 = float(t0); dur = float(dur)

    phase = np.abs(((time - t0 + 0.5 * P) % P) - 0.5 * P)
    return phase < (0.5 * dur + buffer_days)


# ---------------------------
# Main per-target pipeline
# ---------------------------
def run_fit_refine_for_target(target: Target, global_csv_path: Path) -> None:
    target.load_state()

    # Gate: must have DT pass-1 run file
    last_rel = getattr(target, "last_candidates_run", None)
    if not last_rel:
        print(f"[SKIP] {target.root_dir.name}: no last_candidates_run (run 02 first).")
        return

    run_path = (target.root_dir / last_rel).resolve()
    if not run_path.exists():
        print(f"[SKIP] {target.root_dir.name}: last_candidates_run missing on disk.")
        return

    # Load run json + periodic latest
    run_json = load_run_json(run_path)

    periodic_raw = run_json.get("periodic_events_raw_latest", None)

    
    # Load merged total for fitting arrays

    flavour = target.data_source.value
    total_csv = find_total_csv(target.root_dir, flavour)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])
    time = df["TIME"].to_numpy(float)
    flux = df["FLUX"].to_numpy(float)
    unc = (
        df["FLUX_ERR"].to_numpy(float)
        if "FLUX_ERR" in df.columns
        else np.full_like(flux, np.nanstd(flux))
    )

    primary_events = [
        PeriodicEvent.from_dict(d)
        for d in run_json.get("periodic_hypotheses_primary", [])
    ]

    alternative_events = [
        PeriodicEvent.from_dict(d)
        for d in run_json.get("periodic_hypotheses_alternative", [])
    ]


    periodic_candidates = []
    single_candidates = []

    all_periodic = primary_events#+alternative_events

    if not len(all_periodic)>0:
        if periodic_raw:
            all_periodic = [PeriodicEvent.from_dict(d) for d in periodic_raw]


    if len(all_periodic)>0:

        pass2_events = []




        upsert_run_json(run_path, {"status": {"stage": "fit_refine", "state": "running", "updated_at": datetime.now().isoformat()}})

        seen_periods = []
        unique_events = []

        for ev in all_periodic:
            if any(abs(ev.period_days - p) / max(p, 1e-12) < 0.02 for p in seen_periods):
                continue
            seen_periods.append(ev.period_days)
            unique_events.append(ev)

        periodic_events = unique_events



        # 1) Convert periodic events -> periodic candidates and fit them
        for ev in periodic_events:
            if ev.duration_days is None or ev.depth is None:

                print(f"[WARN] Missing duration/depth for period {ev.period_days}")

                upsert_run_json(run_path, {
                    "warnings": [f"Skipped periodic event with missing duration/depth: {ev.to_dict()}"]
                })
                continue
            pc = PlanetCandidate(
                ptype="Periodic",
                t0_days=float(ev.t0_days),
                period_days=float(ev.period_days),
                duration_days=float(ev.duration_days) if ev.duration_days is not None else None,
                depth=normalize_depth_to_fractional(ev.depth) if ev.depth is not None else None,
                n_transits_obs=ev.n_transits_obs,
                transit_times_days = ev.transit_times_days,
                source="BLS",
            )
            print('Checking period: ', ev.period_days)
            fit_and_attach(target, pc, time, flux, unc, run_path, verbose=False)
            periodic_candidates.append(pc)


        # 2) Mask using fitted periodic candidates ONLY
        intransit = np.zeros_like(time, dtype=bool)
        for pc in periodic_candidates:
            if pc.fit_is_current:
                intransit |= periodic_mask_from_fitted_candidate(time, pc, buffer_days=0.2)

        have_mask = bool(intransit.any())

        print(f"[DEBUG] masked points: {intransit.sum()} / {len(intransit)}")
        print(f"[DEBUG] periodic fitted: {[pc.fit_is_current for pc in periodic_candidates]}")


        # 2.5) CHECK BEFORE DT pass-2 - were all the DT pass-1 things found? 
        # 3) DT pass-2 (residual) ONLY if we actually masked something

        pass2_events = []


        if have_mask:
            # Prevent pass2 accumulation across reruns
            overwrite_run_json_keys(run_path, {
                "dt_events_raw_pass2": [],
                # optional: also clear any derived products you might store
                # "dt_events_pass2_summary": [],
})
            singles_cfg = SinglesSearchConfig(flavour=flavour, confidence=0.7, plot_events=False, verbose=False)
            singles_search(
                target,
                cfg=singles_cfg,
                exclude_mask=intransit,
                pass_label="pass2",
                run_id=run_path.stem.replace("run_", ""),
                run_path=run_path
            )
            # Reload run json to get pass2 events

            run_json = load_run_json(run_path)
            raw_pass2 = run_json.get("dt_events_raw_pass2", [])
            pass2_events = [TransitEvent.from_dict(d) for d in raw_pass2] if isinstance(raw_pass2, list) else []
        print(f"[DEBUG] pass2_events: {len(pass2_events)}")


        print('time', len(time))

        time_min = float(min(time))
        time_max = float(max(time))

        # 4) Fit the pass2 events as singles (then later: promote periodic if periodicity emerges)
        for ev in pass2_events:
            if single_matches_periodic(ev.t0_days, periodic_candidates, time_min, time_max):
                continue

            sc = PlanetCandidate(
                ptype="Single",
                t0_days=float(ev.t0_days),
                period_days=None,
                duration_days=float(ev.duration_days),
                depth=normalize_depth_to_fractional(ev.depth),
                snr=None if ev.snr is None else float(ev.snr),
                source="DT_pass2",
                transit_times_days=[float(ev.t0_days)]
            )
            fit_and_attach(target, sc, time, flux, unc, run_path, verbose=False)
            single_candidates.append(sc)

           # 5) Optional promotion from fitted singles

    else: 
        pass1_events, single_candidates = finalize_pass1_singles_only(target, run_path, run_json)    

    # 6) Write outputs (no PDFs)
    candidates = periodic_candidates + single_candidates 

    candidates = [c for c in candidates if ped._snr_med(c) > 10]

    final_candidates = resolve_periodic_conflicts(candidates)


    # time_min = float(time.min())
    # time_max = float(time.max())
    # final_candidates = consume_singles_under_periodics(final_candidates, time_min, time_max)

    # final_defaults = [c for c in final_candidates if c.default == True and c.fit_is_current]
 
    print('final candidates', final_candidates)
    per_target_csv = None
    if len(final_candidates)>0:
        per_target_csv = write_final_candidates_csv(target, final_candidates)
        append_global_candidates_csv(final_candidates, target, global_csv_path)

        # Update stage
        target.set_stage(PipelineStage.FITTED)
    enqueue("DONE_FOUND", target.ticid)
    upsert_run_json(run_path, {"status": {"stage": "fit_refine", "state": "done", "finished_at": datetime.now().isoformat()}})
    print(f"[DONE] {target.root_dir.name}: wrote {per_target_csv} and appended to {global_csv_path}")


def main(idx: int) -> None:
    dirs = sorted(glob.glob(TARGET_GLOB))
    if not (0 <= idx < len(dirs)):
        print(f"[FATAL] idx={idx} out of range for {len(dirs)} targets.")
        sys.exit(2)

    root = Path(dirs[idx])
    target = Target.from_dir(root)
    # print('Target:', target)

    if not target.stage_at_least(PipelineStage.SEARCHED1):
        print(f"[FATAL] {root.name}: need DT pass‑1 first (stage < SEARCHED1). Run script 02.")
        sys.exit(3)


    global_csv = Path.cwd() / "all_final_candidates.csv"
    run_fit_refine_for_target(target, global_csv)


if __name__ == "__main__":
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    print('indx string', idx_str)
    if idx_str is None:
        print("Usage: python scripts/04_run_fit_refine.py <index>  # or SLURM_ARRAY_TASK_ID")
        sys.exit(1)
    main(int(idx_str))