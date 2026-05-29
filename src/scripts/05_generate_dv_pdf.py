#!/usr/bin/env python
import glob
import os
import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.target import Target, PipelineStage
from utils.find_total_csv import find_total_csv

# Your DV plotting lives here (we'll add it next)
from utils.dv_gen_page1 import creating_first_DV_report_page,build_planet_df_from_final_csv, build_catalog_df_for_target

TARGET_GLOB = "../../toi_data/target_*"   # match your other scripts’ pattern

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def write_figures_to_pdf(figures, data_path, *, close=True, metadata=None):
    """
    Write an iterable of matplotlib Figure objects to a single PDF.

    Parameters
    ----------
    figures : iterable
        Iterable (list or generator) yielding matplotlib.figure.Figure objects.
    out_path : str | Path
        Output PDF filename.
    close : bool
        If True, plt.close(fig) after saving each page (recommended).
    metadata : dict | None
        Optional PDF metadata, e.g. {"Title": "...", "Author": "..."}.
    """

    target = str(data_path).split('/')[-1]


    out_path = '../../DV_reports/'+target+'.pdf'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        if metadata:
            info = pdf.infodict()
            for k, v in metadata.items():
                info[str(k)] = str(v)

        for fig in figures:
            if fig is None:
                continue
            pdf.savefig(fig)   # writes current figure as a page
            if close:
                plt.close(fig)

    return out_path


def run_for_target(target: Target) -> Path:
    target.load_state()

    # 1) locate final candidates
    final_csv = target.root_dir / "final_candidates.csv"
    if not final_csv.exists():
        raise FileNotFoundError(f"Missing {final_csv}")

    # 2) build planet_df in the shape your DV function expects
    planet_df = build_planet_df_from_final_csv(final_csv)

    # 3) locate the primary pipeline “total” CSV (local, no MAST yet)
    flavour = target.data_source.value
    data_filename = find_total_csv(target.root_dir, flavour)

    # 4) build catalog_df (thin 1-row object with RA/DEC/Rad/Mass/Teff/etc)
    # catalog_df = build_catalog_df_for_target(target)

    # 5) build a mask if you have one; otherwise empty for now
    intransit = []  # later: load from run json or recompute from fitted periodic candidates

    # 6) generate the PDF
    fig1 = creating_first_DV_report_page(
        target,
        planet_df=planet_df,
        intransit=intransit,
    )
    out = write_figures_to_pdf([fig1], target.root_dir)
    return out


def main(idx: int) -> None:
    dirs = sorted(glob.glob(TARGET_GLOB))
    if not (0 <= idx < len(dirs)):
        print(f"[FATAL] idx={idx} out of range for {len(dirs)} targets.")
        sys.exit(2)

    root = Path(dirs[idx])
    target = Target.from_dir(root)

    if not target.stage_at_least(PipelineStage.FITTED):
        print(f"[FATAL] {root.name}: need to fit data first (stage < FITTED). Run script 04.")
        sys.exit(3)

    out = run_for_target(target)

    target.set_stage(PipelineStage.REPORTED)

    print(f"[DONE] wrote {out}")


if __name__ == "__main__":
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if idx_str is None:
        print("Usage: python scripts/05_generate_dv_pdf.py <index>  # or SLURM_ARRAY_TASK_ID")
        sys.exit(1)
    main(int(idx_str))