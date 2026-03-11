# scripts/02_run_quick_singles.py
#!/usr/bin/env python
import glob
import os
import sys
from pathlib import Path
import pandas as pd

from core.target import Target, PipelineStage
from stages.search_singles import singles_search, SinglesSearchConfig

TARGET_GLOB = "../toi_data/target_*"
FLAVOUR = "TGLC"

def _has_merged_data(target: Target, flavour: str) -> bool:
    try:
        target.load_state()
    except Exception:
        pass
    if getattr(target, "pipeline_stage", None) == PipelineStage.MERGED:
        return True
    rd = target.root_dir
    return any(rd.glob(f"*{flavour}*_*total.csv")) or any(rd.glob("*total.csv"))

def main(idx: int) -> None:
    dirs = sorted(glob.glob(TARGET_GLOB))
    if not (0 <= idx < len(dirs)):
        print(f"[FATAL] idx={idx} out of range for {len(dirs)} targets.")
        sys.exit(2)

    root = Path(dirs[idx])
    ticid, gaia_id = Target.discover_ids_from_dirname(root)
    t = Target(ticid=int(ticid), gaia_id=gaia_id, root_dir=root)
    t.load_state()

    if not _has_merged_data(t, FLAVOUR):
        print(f"[{root.name}] Not ready (PipelineStage < MERGED and no *total.csv). Skipping.")
        return

    cfg = SinglesSearchConfig(flavour=FLAVOUR, confidence=0.55, plot_events=False, verbose=False)
    planet_df, _ = singles_search(t, cfg=cfg, run_1=True)

    found = len(planet_df) > 0
    t.dt_prelim_found = found
    t.set_stage(PipelineStage.SEARCHED)
    t.save_state()

    print(f"[TIC {t.ticid}] {'FOUND' if found else 'no'} transit-like signal")

if __name__ == "__main__":
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if idx_str is None:
        print("Usage: python scripts/02_run_quick_singles.py <index>  # or SLURM_ARRAY_TASK_ID")
        sys.exit(1)
    main(int(idx_str))