#!/usr/bin/env python
import glob
import os
import sys
import json
from pathlib import Path
from datetime import datetime

import gc
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
    periodic_modes_from_dt_events,
    periodic_candidates_from_modes,
    mark_single_members_consumed,
    keep_unique_period_indices,
    group_events_by_depth,
)

from utils.alias_dedup import alias_dedup_periodic_candidates
from utils.queue import enqueue

from stages.search_singles import singles_search, SinglesSearchConfig
from stages.search_periodic import PeriodicSearchConfig, run_seed_prepass_full_lc
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
            "pymc_summary_json": json.dumps(c.pymc_summary) if c.pymc_summary else "",
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def append_global_candidates_csv(candidates: list[PlanetCandidate], target: Target, global_path: Path) -> None:
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


# ---------------------------
# Small helpers
# ---------------------------

def _epochs_in_window(tmin, tmax, t0, P):
    k0 = int(np.floor((tmin - t0) / P)) - 1
    k1 = int(np.ceil((tmax - t0) / P)) + 1
    ks = np.arange(k0, k1 + 1)
    return t0 + ks * P

def single_matches_periodic(s_t0, periodic_candidates, time_min, time_max, fixed_tol_days=0.25):
    s_t0 = float(s_t0)

    for p in periodic_candidates:
        if not getattr(p, "fit_is_current", False):
            continue

        P = p.period_days
        t0 = p.t0_days
        dur = p.duration_days

        if P is None or t0 is None:
            continue

        P = float(P)
        t0 = float(t0)

        if not np.isfinite(P) or P <= 0:
            continue

        epochs = _epochs_in_window(time_min, time_max, t0, P)

        if dur is not None and np.isfinite(dur):
            tol = max(fixed_tol_days, 0.25 * float(dur))
        else:
            tol = fixed_tol_days

        if np.min(np.abs(epochs - s_t0)) <= tol:
            return True

    return False

def _summary_median(cand, varname, fallback=None):
    try:
        d = cand.pymc_summary[varname]
        if isinstance(d, dict) and "median" in d:
            return float(d["median"])
    except Exception:
        pass
    return fallback

def periodic_mask_from_fitted_candidate(time: np.ndarray, cand: PlanetCandidate, buffer_days: float = 0.5) -> np.ndarray:
    P = _summary_median(cand, "Per", fallback=cand.period_days)
    t0 = _summary_median(cand, "t0", fallback=cand.t0_days)
    dur = _summary_median(cand, "dur", fallback=cand.duration_days)

    if P is None or dur is None or t0 is None:
        return np.zeros_like(time, dtype=bool)

    P = float(P)
    t0 = float(t0)
    dur = float(dur)

    phase = np.abs(((time - t0 + 0.5 * P) % P) - 0.5 * P)
    return phase < (0.5 * dur + buffer_days)


# ---------------------------
# Fit helpers
# ---------------------------

def fit_and_attach(
    target: Target,
    cand: PlanetCandidate,
    time,
    flux,
    unc,
    run_path: Path,
    verbose: bool = False,
    max_runs: int = 3
) -> bool:
    print("trying fit")
    attempt_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    upsert_run_json(run_path, {"status": {"stage": "pymc_fit", "state": "running", "attempt_id": attempt_id}})

    summary_df, ok, fit_info = pymc_fit_candidate(
        target, cand, time, flux, unc, verbose=verbose, max_runs=max_runs
    )

    print("ok? ", ok)

    if ok and summary_df is not None:
        cand.pymc_summary = summary_df.to_dict(orient="index")
        cand.mark_fitted()

        print("candidate pymc summary", cand.pymc_summary)

        stats_csv = write_converged_fit_csv(target, cand, fit_info)
        print(f"wrote converged fit stats to {stats_csv}")

        cand.t0_days = _summary_median(cand, "t0", fallback=cand.t0_days)
        if cand.ptype == "Periodic":
            cand.period_days = _summary_median(cand, "Per", fallback=cand.period_days)
        cand.duration_days = _summary_median(cand, "dur", fallback=cand.duration_days)

        raw_depth = _summary_median(cand, "depth", fallback=cand.depth)
        cand.depth = normalize_depth_to_fractional(raw_depth)
        cand.rp_rs = _summary_median(cand, "rp_rs", fallback=None)
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

    upsert_run_json(run_path, {"status": {"stage": "pymc_fit", "state": "done", "attempt_id": attempt_id}})
    gc.collect()

    return bool(ok)

def finalize_pass1_singles_only(target, run_path, run_json, time, flux, unc):
    raw_pass1 = run_json.get("dt_events_raw_pass1", [])
    pass1_events = [TransitEvent.from_dict(d) for d in raw_pass1] if isinstance(raw_pass1, list) else []

    single_candidates = []
    for ev in pass1_events:
        sc = PlanetCandidate(
            ptype="Single",
            t0_days=float(ev.t0_days),
            period_days=None,
            duration_days=float(ev.duration_days),
            depth=normalize_depth_to_fractional(ev.depth),
            source="DT_pass1",
        )
        ok = fit_and_attach(target, sc, time, flux, unc, run_path, verbose=False)
        if ok:
            single_candidates.append(sc)

    return pass1_events, single_candidates


# ---------------------------
# Main per-target pipeline
# ---------------------------

def run_fit_refine_for_target(target: Target, global_csv_path: Path) -> None:
    target.load_state()

    last_rel = getattr(target, "last_candidates_run", None)
    if not last_rel:
        print(f"[SKIP] {target.root_dir.name}: no last_candidates_run (run 02 first).")
        return

    run_path = (target.root_dir / last_rel).resolve()
    if not run_path.exists():
        print(f"[SKIP] {target.root_dir.name}: last_candidates_run missing on disk.")
        return

    run_json = load_run_json(run_path)
    periodic_raw = run_json.get("periodic_events_raw_latest", None)

    if periodic_raw is None:
        attempts = run_json.get("periodic_attempts", [])
        if attempts:
            periodic_raw = attempts[-1].get("periodic_events_raw", [])

    # >>> CHANGED: load light curve ONCE here so it's always available
    flavour = target.data_source.value
    total_csv = find_total_csv(target.root_dir, flavour)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])

    time = df["TIME"].to_numpy(float)
    flux = df["FLUX"].to_numpy(float)
    unc = df["FLUX_ERR"].to_numpy(float) if "FLUX_ERR" in df.columns else np.full_like(flux, np.nanstd(flux))

    time_min = float(np.min(time))
    time_max = float(np.max(time))

    periodic_candidates = []
    single_candidates = []

    if (periodic_raw is None) or (isinstance(periodic_raw, list) and len(periodic_raw) == 0):
        print("no periodic")

        if not bool(getattr(target, "dt_prelim_found", False)):
            print("no initial singles")
            print(f"[SKIP] {target.root_dir.name}: dt_prelim_found=False, dt_periodic_found=False, nothing to fit.")
        else:
            pass1_events, single_candidates = finalize_pass1_singles_only(
                target, run_path, run_json, time, flux, unc
            )
    else:
        pass1_events = [TransitEvent.from_dict(d) for d in run_json.get("dt_events_raw_pass1", [])]

    raw_pass1 = run_json.get("dt_events_raw_pass1", [])
    pass1_events = [TransitEvent.from_dict(d) for d in raw_pass1] if isinstance(raw_pass1, list) else []

    if periodic_raw:
        periodic_events = [PeriodicEvent.from_dict(d) for d in periodic_raw]

        upsert_run_json(run_path, {"status": {"stage": "fit_refine", "state": "running", "updated_at": datetime.now().isoformat()}})

        # 1) Fit the existing periodic events
        for ev in periodic_events:
            if ev.duration_days is None or ev.depth is None:
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
                transit_times_days=ev.transit_times_days,
                source="BLS",
            )

            print("Checking period: ", ev.period_days)
            ok = fit_and_attach(target, pc, time, flux, unc, run_path, verbose=False)
            if ok:
                periodic_candidates.append(pc)

        # 2) Mask using fitted periodic candidates only
        intransit = np.zeros_like(time, dtype=bool)
        for pc in periodic_candidates:
            if pc.fit_is_current:
                intransit |= periodic_mask_from_fitted_candidate(time, pc, buffer_days=0.2)

        have_mask = bool(intransit.any())

        print(f"[DEBUG] masked points: {intransit.sum()} / {len(intransit)}")
        print(f"[DEBUG] periodic fitted: {[pc.fit_is_current for pc in periodic_candidates]}")

        # 2.5) Check recovery of pass1 singles against these periodic candidates
        for pc in periodic_candidates:
            result = check_singles_against_periodic_candidate(
                periodic=pc,
                singles=pass1_events
            )

            print("Matched:", result["n_matched"])
            print("Unmatched:", result["n_unmatched"])

            for c in result["unmatched_candidates"]:
                print("Unmatched single:", c.t0_days)

        # 3) DT pass2 on residuals
        pass2_events = []

        if have_mask:
            overwrite_run_json_keys(run_path, {"dt_events_raw_pass2": []})

            singles_cfg = SinglesSearchConfig(
                flavour=flavour,
                confidence=0.75,
                plot_events=False,
                verbose=False
            )

            singles_search(
                target,
                cfg=singles_cfg,
                exclude_mask=intransit,
                pass_label="pass2",
                run_id=run_path.stem.replace("run_", ""),
                run_path=run_path
            )

            run_json = load_run_json(run_path)
            raw_pass2 = run_json.get("dt_events_raw_pass2", [])
            pass2_events = [TransitEvent.from_dict(d) for d in raw_pass2] if isinstance(raw_pass2, list) else []

        print(f"[DEBUG] pass2_events: {len(pass2_events)}")
        print("time", len(time))

        # 4) Fit pass2 singles
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

            ok = fit_and_attach(target, sc, time, flux, unc, run_path, verbose=False, max_runs=2)
            if ok:
                single_candidates.append(sc)

    print("number of singles remaining ", len(single_candidates), " vs originally", len(pass2_events))

    # 5) Promote periodic modes from the current singles
    event_groups = group_events_by_depth(single_candidates)

    promotions = []

    # >>> CHANGED: build promotions PER GROUP so local indices stay valid
    for g in event_groups:
        if len(g) < 3:
            continue

        group_modes = periodic_modes_from_dt_events(
            g,
            min_support=3,
            use_depth=True,
            P_min=1
        )

        group_promotions = periodic_candidates_from_modes(
            group_modes,
            g,
            source="DT_PASS_PROMOTED",
            min_support=3,
            notes_prefix="promoted_from_pass; "
        )

        for pc, member_idx in group_promotions:
            promotions.append((pc, member_idx, g))

    promoted_periodic_candidates = []

    # keep strongest close-enough period first
    promotions.sort(key=lambda x: (-x[0].n_transits_obs, x[0].period_days))

    pers = [pc.period_days for pc, _, _ in promotions]
    per_sort_idx = keep_unique_period_indices(pers)

    for idx in per_sort_idx:
        pc, member_idx, member_pool = promotions[idx]

        # >>> CHANGED: indices are local to this depth group
        member_events = [member_pool[i] for i in member_idx]
        t0s = np.array([e.t0_days for e in member_events], dtype=float)

        sp_cand = []
        for i in range(len(t0s)):
            for j in range(i + 1, len(t0s)):
                dt = t0s[j] - t0s[i]
                if dt <= 0:
                    continue
                sp_cand.append(dt)

        sp_cand = np.array(sp_cand, dtype=float)
        if len(sp_cand) == 0:
            continue

        kept_idx = keep_unique_period_indices(sp_cand)
        seed_pers = sp_cand[kept_idx]
        seed_pers = seed_pers[seed_pers > 1]
        print('seeded periods', seed_pers)
        if len(seed_pers) == 0:
            continue
        if len(seed_pers) == 1:
            ppp = seed_pers[0]
            if ppp<0.5:
                continue
            elif ppp>2:
                seed_pers = np.array([0.5, 1, 2])* ppp
            else:
                seed_pers = np.array([1, 2, 3])* ppp
                
                

        accepted_events, _ = run_seed_prepass_full_lc(
            time,
            flux,
            unc,
            cfg=PeriodicSearchConfig(use_seed_periods=True),
            seed_periods=list(seed_pers),
            accepted_events=[],
            intransit=np.zeros_like(time, dtype=bool)
        )

        for ev in accepted_events:
            if ev.duration_days is None or ev.depth is None:
                upsert_run_json(run_path, {
                    "warnings": [f"Skipped periodic event with missing duration/depth: {ev.to_dict()}"]
                })
                continue

            pc_p = PlanetCandidate(
                ptype="Periodic",
                t0_days=float(ev.t0_days),
                period_days=float(ev.period_days),
                duration_days=float(ev.duration_days) if ev.duration_days is not None else None,
                depth=normalize_depth_to_fractional(ev.depth) if ev.depth is not None else None,
                n_transits_obs=ev.n_transits_obs,
                transit_times_days=ev.transit_times_days,
                source="BLS",
            )

            print("Checking period: ", ev.period_days)
            ok = fit_and_attach(target, pc_p, time, flux, unc, run_path, verbose=False, max_runs=2)

            if ok:
                promoted_periodic_candidates.append(pc_p)

                # >>> CHANGED: consume the actual singles corresponding to this member set
                consumed_idx = [
                    i for i, sc in enumerate(single_candidates)
                    if any(sc is me for me in member_events)
                ]
                mark_single_members_consumed(single_candidates, consumed_idx, pc_p.candidate_id())

    deduped = alias_dedup_periodic_candidates(periodic_candidates + promoted_periodic_candidates)

    for c in deduped:
        if getattr(c, "ptype", None) == "Periodic":
            print(c.candidate_id(), c.period_days, getattr(c, "default", True), c.notes)

    # 6) Remove singles now explained by final periodic set
    # >>> CHANGED: use deduped final periodic list, not only periodic_candidates
    single_candidates = [
        sc for sc in single_candidates
        if not single_matches_periodic(sc.t0_days, deduped, time_min, time_max)
    ]

    final_candidates = deduped + single_candidates

    per_target_csv = None
    if len(final_candidates) > 0:
        per_target_csv = write_final_candidates_csv(target, final_candidates)
        append_global_candidates_csv(final_candidates, target, global_csv_path)
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

    global_csv = Path.cwd() / "all_final_candidates.csv"
    run_fit_refine_for_target(target, global_csv)


if __name__ == "__main__":
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    print("indx string", idx_str)
    if idx_str is None:
        print("Usage: python scripts/04_run_fit_refine.py <index>  # or SLURM_ARRAY_TASK_ID")
        sys.exit(1)
    main(int(idx_str))