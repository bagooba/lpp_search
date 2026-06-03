#!/usr/bin/env python

import glob
import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.target import Target
from core.transit_event import TransitEvent
from core.periodic_event import PeriodicEvent

from utils.find_total_csv import find_total_csv
from utils.run_json import upsert_run_json
from utils.queue import enqueue

from utils.singles_periodicity import seed_periods_from_dt_events
from stages.search_periodic import run_seed_prepass_full_lc, PeriodicSearchConfig

TARGET_GLOB = "../../toi_data/target_*"


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def period_is_close(P1, P2, tol=0.02):
    if P1 is None or P2 is None:
        return False
    return abs(P1 - P2) / max(P2, 1e-12) < tol


def dedupe_periods(events, tol=0.02):
    unique = []
    for ev in events:
        keep = True
        for other in unique:
            if period_is_close(ev.period_days, other.period_days, tol):
                keep = False
                break
        if keep:
            unique.append(ev)
    return unique


def evaluate_with_bls(ev, time, flux, flux_err, flavour):
    cfg = PeriodicSearchConfig(flavour=flavour)
    cfg.use_seed_periods = True
    cfg.max_iters = 1

    accepted_events, _ = run_seed_prepass_full_lc(
        time,
        flux,
        flux_err,
        cfg=cfg,
        seed_periods=[ev.period_days],
        accepted_events=[],
        intransit=np.zeros_like(time, dtype=bool),
    )

    if len(accepted_events) == 0:
        return None

    return accepted_events[0]  # return new evaluated event


# -------------------------------------------------
# Core
# -------------------------------------------------

def alias_dedup_periodic_events(
    periodic_events,
    extra_events,
    time,
    flux,
    flux_err,
    flavour,
    *,
    shared_t0_tol_days=1/3,
    min_shared_events=2,
    max_keep_per_cluster=2,
):

    if len(periodic_events) == 0:
        return [], []

    # ----------------------------------------
    # clustering (shared events)
    # ----------------------------------------
    groups = []

    for ev in periodic_events:
        placed = False

        for g in groups:
            for other in g:
                t1 = np.array(ev.transit_times_days or [])
                t2 = np.array(other.transit_times_days or [])

                if len(t1) == 0 or len(t2) == 0:
                    continue

                overlap = np.sum([
                    np.any(np.abs(t2 - x) <= shared_t0_tol_days)
                    for x in t1
                ])

                if overlap >= min_shared_events:
                    g.append(ev)
                    placed = True
                    break

            if placed:
                break

        if not placed:
            groups.append([ev])

    primary = []
    alternative = []

    # ----------------------------------------
    # process clusters
    # ----------------------------------------
    for cluster in groups:
        sub_alt = []

        members = cluster + extra_events

        hypotheses = []

        # ---- existing periodic events
        hypotheses.extend(cluster)

        # ---- seed-generated periods (YOUR function)
        seed_rows = seed_periods_from_dt_events(
            members,
            min_support=3,
            depth_ratio_max=1.25,
            perc_tol=0.02,
            max_divisor=3,
            P_min=0.5,
        )

        for row in seed_rows:
            ev = PeriodicEvent(
                t0_days=row["t0_seed"],
                period_days=row["P"],
                duration_days=row["duration"],
                depth=row["depth"],
                transit_times_days=row["t0s"],
                n_transits_obs=row["support"],
            )
            hypotheses.append(ev)

        # ---- spacing-based periods (offset detection)
        t0s_all = sorted([
            t for ev in members
            for t in (getattr(ev, "transit_times_days", None) or [])
        ])

        if len(t0s_all) >= 2:
            diffs = np.diff(np.unique(t0s_all))

            for d in diffs:
                if d > shared_t0_tol_days:
                    ev = PeriodicEvent(
                        t0_days=t0s_all[0],
                        period_days=float(d),
                        duration_days=None,
                        depth=None,
                        transit_times_days=t0s_all,
                        n_transits_obs=len(t0s_all),
                    )
                    hypotheses.append(ev)

        # ---- dedupe (fractional)
        hypotheses = dedupe_periods(hypotheses)

        # cap for stability
        if len(hypotheses) > 30:
            hypotheses = hypotheses[:30]

        # ----------------------------------------
        # evaluate ALL hypotheses with BLS
        # ----------------------------------------
        evaluated = []

        for ev in hypotheses:
            if ev.period_days is None or ev.period_days <= 0:
                continue

            new_ev = evaluate_with_bls(ev, time, flux, flux_err, flavour)
            if new_ev is not None:
                evaluated.append(new_ev)

        # ----------------------------------------
        # sort by SNR
        # ----------------------------------------
        evaluated.sort(
            key=lambda ev: (
                getattr(ev, "snr", -np.inf),
                -ev.period_days if ev.period_days else 0,
            ),
            reverse=True,
        )

        # ----------------------------------------
        # enforce NO SHARED EVENTS
        # ----------------------------------------
        selected = []
        used = []

        for ev in evaluated:
            t0s = np.array(ev.transit_times_days or [])

            conflict = False
            for ut in used:
                if np.any(np.abs(t0s - ut) < shared_t0_tol_days):
                    conflict = True
                    break

            if conflict:
                alternative.append(ev)
                sub_alt.append(ev)
                continue

            selected.append(ev)
            used.extend(t0s)

            if len(selected) >= max_keep_per_cluster:
                alternative.append(ev)
                sub_alt.append(ev)
                continue

        print('prime periods: ', [ev.period_days for ev in selected])
        print('alternate periods: ', [ev.period_days for ev in sub_alt])

        primary.extend(selected)

    return primary, alternative


# -------------------------------------------------
# Main
# -------------------------------------------------

def main(idx):
    dirs = sorted(glob.glob(TARGET_GLOB))
    root = Path(dirs[idx])
    target = Target.from_dir(root)

    target.load_state()
    run_path = target.root_dir / target.last_candidates_run
    run_json = json.loads(run_path.read_text())

    periodic_raw = run_json.get("periodic_events_raw_latest", [])
    periodic_events = [PeriodicEvent.from_dict(d) for d in periodic_raw]

    raw_pass1 = run_json.get("dt_events_raw_pass1", [])
    pass1_events = [TransitEvent.from_dict(d) for d in raw_pass1]

    flavour = target.data_source.value
    total_csv = find_total_csv(target.root_dir, flavour)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])

    time = df["TIME"].to_numpy(float)
    flux = df["FLUX"].to_numpy(float)
    flux_err = (
        df["FLUX_ERR"].to_numpy(float)
        if "FLUX_ERR" in df.columns
        else np.full_like(flux, np.nanstd(flux))
    )

    primary, alternative = alias_dedup_periodic_events(
        periodic_events,
        pass1_events,
        time,
        flux,
        flux_err,
        flavour,
    )

    upsert_run_json(
        run_path,
        {
            "periodic_hypotheses_primary": [e.to_dict() for e in primary],
            "periodic_hypotheses_alternative": [e.to_dict() for e in alternative],
            "status": {
                "stage": "periodic_prefit",
                "state": "done",
                "updated_at": datetime.now().isoformat(),
            },
        },
    )

    enqueue("04", target.ticid)

    print(f"[DONE] {root.name}: primary={len(primary)}, alt={len(alternative)}")


if __name__ == "__main__":
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    main(idx)