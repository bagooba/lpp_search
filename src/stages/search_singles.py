# stages/search_singles.py
# Quick single-transit detection (DT) that ONLY sets Target.dt_prelim_found and quick_singles_t0.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import deep_transit as dt

from core.target import Target
from utils.segments import breaking_up_data
import config as con

def make_LightKurveObject(time, flux, flux_err):
    lc = lk.TessLightCurve()
    lc.time = time; lc.flux = flux; lc.flux_err = flux_err
    return lc  # [1](https://unmm-my.sharepoint.com/personal/malharris19_unm_edu/Documents/Microsoft%20Copilot%20Chat%20Files/Functions_all.py)

def calc_rudimentary_snr(depth, Tdur, Ntran=1):
    sigma_1hr_15_Tmag = 6283.6147036936645 * 1e-6
    return (Ntran**0.5 / sigma_1hr_15_Tmag) * depth * np.sqrt(Tdur * 24)  # [1](https://unmm-my.sharepoint.com/personal/malharris19_unm_edu/Documents/Microsoft%20Copilot%20Chat%20Files/Functions_all.py)

def plot_lc_with_bboxes(lc_object, bboxes, ax=None, epoch=0, **kwargs):
    with plt.style.context('grayscale'):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(12, 6), constrained_layout=False)
            ax.plot(lc_object.time.value, lc_object.flux.value, color='k', zorder=1e5, **kwargs)
            ax.set_xlabel('Time - T0 (hours)'); ax.set_ylabel('Normalized Flux')
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        recs = []
        for real_mask in bboxes:
            new_start = real_mask[1] - epoch
            rec = Rectangle(
                (new_start - real_mask[3]/2, real_mask[2] - real_mask[4]/2),
                real_mask[3], real_mask[4],
                facecolor='indianred', edgecolor='indianred', linewidth=1.0, zorder=5
            )
            recs.append(rec)
            SNR = calc_rudimentary_snr(real_mask[3], real_mask[4])
            ax.text(new_start + abs(real_mask[3]), real_mask[2] + 0.5*abs(real_mask[4]), s=f"SNR: {SNR:.2f}", color='r')
        ax.add_collection(PatchCollection(recs, lw=0.2, match_original=True, zorder=5))
        return ax  # [1](https://unmm-my.sharepoint.com/personal/malharris19_unm_edu/Documents/Microsoft%20Copilot%20Chat%20Files/Functions_all.py)

def DT_analysis(time, flux, flux_err, confidence, DT_Quite=True, is_flat=True):
    if DT_Quite:
        save_stdout, save_stderr = sys.stdout, sys.stderr
        sys.stdout = open('.trash.txt', 'w'); sys.stderr = open('.trash.txt', 'w')
    model = dt.DeepTransit(make_LightKurveObject(time, flux, flux_err), is_flat=is_flat)
    bboxes = model.transit_detection(str(con.MODEL_PATH), confidence_threshold=confidence)
    if DT_Quite:
        sys.stdout.close(); sys.stderr.close()
        sys.stdout, sys.stderr = save_stdout, save_stderr
    return bboxes  # [1](https://unmm-my.sharepoint.com/personal/malharris19_unm_edu/Documents/Microsoft%20Copilot%20Chat%20Files/Functions_all.py)

def _find_total_csv(root_dir: Path, flavour: str) -> Path:
    patt = f"*{flavour}*_*total.csv"
    m = sorted(root_dir.glob(patt))
    if m: return max(m, key=lambda p: p.stat().st_mtime)
    m = sorted(root_dir.glob("*total.csv"))
    if m: return max(m, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No merged total CSV found in {root_dir}.")

@dataclass
class SinglesSearchConfig:
    flavour: str = "TGLC"
    confidence: float = 0.55
    plot_events: bool = False
    verbose: bool = False

def singles_search(target: Target, *, cfg: SinglesSearchConfig = SinglesSearchConfig(), run_1: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ticid = int(target.ticid)
    total_csv = _find_total_csv(target.root_dir, cfg.flavour)
    df = pd.read_csv(total_csv).dropna(subset=['FLUX'])
    total_time = df['TIME'].to_numpy(dtype=float)
    total_flux = df['FLUX'].to_numpy(dtype=float)
    total_flux_err = (df['FLUX_ERR'].to_numpy(dtype=float)
                      if 'FLUX_ERR' in df.columns else np.full_like(total_flux, np.nanstd(total_flux)))

    # segment and drop too-short fragments (replicates your logic)
    idx_blocks: List[np.ndarray] = breaking_up_data(total_time, break_val=0.5, min_size=1.0)
    if len(idx_blocks) > 1:
        good = np.concatenate(idx_blocks)
        total_time = total_time[good]; total_flux = total_flux[good]; total_flux_err = total_flux_err[good]

    # DT across whole baseline
    bboxes = DT_analysis(total_time, total_flux, total_flux_err, cfg.confidence, DT_Quite=True, is_flat=True)

    planet_df = pd.DataFrame(columns=['TICID','planet_name','period','T0','Tdur','depth'])
    params_df = pd.DataFrame()

    if bboxes is not None and len(bboxes) > 0:
        for k, boxes in enumerate(bboxes, start=1):
            T0, Tdur, depth = float(boxes[1]), float(boxes[3]), float(1 - boxes[4])
            planet_df.loc[len(planet_df)] = [ticid, k, np.inf, T0, Tdur, depth]
            if cfg.plot_events:
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                ax.set_xlim(T0 - 2*Tdur, T0 + 2*Tdur)
                ax.scatter(total_time, total_flux, color='k', s=6, zorder=10)
                shim = type("Obj", (), {})()
                shim.time = type("Arr", (), {"value": total_time})()
                shim.flux = type("Arr", (), {"value": total_flux})()
                plot_lc_with_bboxes(shim, bboxes, ax=ax)
                plt.show()

    # Persist to target.state.json
    t0_list = planet_df["T0"].astype(float).dropna().tolist()
    target.quick_singles_t0 = sorted(set(t0_list))
    target.dt_prelim_found = len(target.quick_singles_t0) > 0
    target.save_state()

    return planet_df.reset_index(drop=True), params_df  # [1](https://unmm-my.sharepoint.com/personal/malharris19_unm_edu/Documents/Microsoft%20Copilot%20Chat%20Files/Functions_all.py)