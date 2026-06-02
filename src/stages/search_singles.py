# stages/search_singles.py
# Quick single-transit detection (DT) that ONLY sets Target.dt_prelim_found and quick_singles_t0.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import deep_transit as dt

from core.transit_event import TransitEvent
from utils.segments import breaking_up_data
from utils.find_total_csv import find_total_csv
from utils.run_json import upsert_run_json
import utils.config as con

def make_LightKurveObject(time, flux, flux_err):
    lc = lk.TessLightCurve()
    lc.time = time; lc.flux = flux; lc.flux_err = flux_err
    return lc

def calc_rudimentary_snr(depth, Tdur, Ntran=1):
    sigma_1hr_15_Tmag = 6283.6147036936645 * 1e-6


    return (Ntran**0.5 / sigma_1hr_15_Tmag) * depth * np.sqrt(Tdur * 24)

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
            dur = float(real_mask[3])
            depth = float(1.0 - real_mask[4])
            SNR = calc_rudimentary_snr(depth, dur)
            ax.text(new_start + abs(real_mask[3]), real_mask[2] + 0.5*abs(real_mask[4]), s=f"SNR: {SNR:.2f}", color='r')
        ax.add_collection(PatchCollection(recs, lw=0.2, match_original=True, zorder=5))
        return ax

def DT_analysis(time, flux, flux_err, confidence,is_flat=True): 
    # DT_Quite=True, 
    # print('not even here?')
    # if DT_Quite:
    #     save_stdout, save_stderr = sys.stdout, sys.stderr
    #     sys.stdout = open('.trash.txt', 'w'); sys.stderr = open('.trash.txt', 'w')
    model = dt.DeepTransit(make_LightKurveObject(time, flux, flux_err), is_flat=is_flat)
    bboxes = model.transit_detection(str(con.MODEL_PATH), confidence_threshold=confidence)
    # if DT_Quite:
    #     sys.stdout.close(); sys.stderr.close()
    #     sys.stdout, sys.stderr = save_stdout, save_stderr
    return bboxes

# def DT_analysis(time, flux, flux_err, confidence, DT_Quite=True, is_flat = True):
#     """ 
#     @author: D. Dragomir, P.Steimle


#     Function for the Deep Transit analysis (Cui et al. 2021) that returns only single transit events

#     """

#     # create a new dataset for the while loop where transits can be masked out
    
    
# #     confidence = 1-np.exp(-0.15*snr)
#     print('time len', len(time))
#     if DT_Quite == True:
#         # save_stdout = sys.stdout
#         # save_stderr = sys.stderr
#         # sys.stdout = open('.trash.txt', 'w')
#         # sys.stderr = open('.trash.txt', 'w')

#         # do check for transits with DT
#         DT_model = dt.DeepTransit(make_LightKurveObject(time, flux, flux_err),  is_flat=is_flat)
#         bboxes = DT_model.transit_detection(model_path, confidence_threshold=confidence)

#         sys.stdout = save_stdout
#         sys.stderr = save_stderr

#     else:
#         # do check for transits with DT
#         DT_model = dt.DeepTransit(make_LightKurveObject(time, flux, flux_err), is_flat=is_flat)
#         bboxes = DT_model.transit_detection(model_path, confidence_threshold=confidence)

#         # if only 1 or 0 boxes were found, break the loop and continue

#     return bboxes

@dataclass
class SinglesSearchConfig:
    flavour: str = "TGLC"
    confidence: float = 0.75
    plot_events: bool = False
    verbose: bool = False


def detect_transit_events(time, flux, flux_err, cfg):
    """
    Run DeepTransit and return:
      - events: list[TransitEvent]
      - bboxes: raw DT bboxes (useful for plotting)
    This is event-level output (NOT planet candidates yet).
    """
    bboxes = DT_analysis(time, flux, flux_err, cfg.confidence, is_flat=True)
    events = []
    
    print('bboxes', len(bboxes))
    if len(bboxes) == 0:
        return events, bboxes

    for boxes in bboxes:
        # Your established bbox convention (as in legacy):
        # t0=boxes[1], dur=boxes[3], depth=1-boxes[4]
        T0 = float(boxes[1])
        Tdur = float(boxes[3])
        depth = float(1.0 - boxes[4])

        # Optional confidence-like value (only if 0..1)
        conf = None
        try:
            conf_val = float(boxes[0])
            if 0.0 <= conf_val <=1.0:
                conf = conf_val
        except Exception:
            pass

        # Rudimentary SNR (depth, duration) using your helper
        snr = None
        try:
            snr = float(calc_rudimentary_snr(depth, Tdur))
        except Exception:
            pass

        events.append(TransitEvent(
            t0_days=T0,
            duration_days=Tdur,
            depth=depth,
            snr=snr,
            confidence=conf,
        ))

    return events, bboxes



def singles_search(target, *, cfg=SinglesSearchConfig(), run_1=True,
                   exclude_mask=None, pass_label="pass1",
                   run_id=None, run_path=None):
    """
    
    Quick DT single-transit detection.

    - Pass 1 (default): run on full merged total LC.
    - Pass 2 (later, in finalize): run on residual LC using exclude_mask
      (typically periodic in-transit points).

    This stage:
      - updates Target.dt_prelim_found and Target.quick_singles_t0
      - writes a run artifact candidates/run_<run_id>.json with dt_events_raw_<pass_label>
      - returns (event_df, params_df) for compatibility with existing code/tests
    """

    print("confidence :", cfg.confidence    )
    ticid = int(target.ticid)
    total_csv = find_total_csv(target.root_dir, cfg.flavour)  # existing helper

    print('file path', total_csv)

    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])
    total_time = df["TIME"].to_numpy(dtype=float)
    total_flux = df["FLUX"].to_numpy(dtype=float)


    events, bboxes = [], []

    if "FLUX_ERR" in df.columns:
        total_flux_err = df["FLUX_ERR"].to_numpy(dtype=float)
    else:
        # keep your pragmatic fallback (constant scatter) 
        total_flux_err = np.full_like(total_flux, np.nanstd(total_flux))

    # Optional exclusion mask (used for DT pass 2 later)
    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape == total_time.shape:
            keep = ~exclude_mask
            total_time = total_time[keep]
            total_flux = total_flux[keep]
            total_flux_err = total_flux_err[keep]

    if total_time.size > 0:

        idx_blocks = breaking_up_data(total_time, break_val=0.5, min_size=2.5)
        good_idx = []
        if len(idx_blocks) > 1:
            spans = np.array([np.ptp(total_time[idx]) for idx in idx_blocks])

            good_idx = np.concatenate(list(itertools.compress(idx_blocks, spans>1))).ravel()

            # good_blocks = [idx_blocks[i] for i in range(len(idx_blocks)) if spans[i] > 1.0]
            # if len(good_blocks) > 0:
            # good_idx = np.concatenate(good_blocks)
        if len(good_idx) == 0: 
            good_idx = list(range(len(total_time)))


            
        total_time = total_time[good_idx]
        total_flux = total_flux[good_idx]
        total_flux_err = total_flux_err[good_idx]

    # --- DT detection (this is the core) ---

        print('time length', len(total_time))
        events, bboxes = detect_transit_events(total_time, total_flux, total_flux_err, cfg)
        


    # --- Update Target quick-singles state (existing contract) ---
    target.dt_prelim_found = (len(events) > 0)
    target.quick_singles_t0 = sorted({float(e.t0_days) for e in events})
    target.save_state()  # persists dt_prelim_found + quick_singles_t0

    # choose run_id/run_path (finalize should pass these)
    if run_id is None:
        run_id = target.new_run_id()

    run_path = target.candidates_run_path(run_id)    
    
    payload_update = {
        "run_id": run_id,
        "ticid": int(target.ticid),
        "gaia_id": target.gaia_id,
        "total_csv": str(total_csv),
        "dt_config": {"flavour": cfg.flavour, "confidence": cfg.confidence},
        f"dt_events_raw_{pass_label}": [e.to_dict() for e in events],
    }

    upsert_run_json(run_path, payload_update)

    # Optionally store pointers if you added them earlier (safe if missing)
    if hasattr(target, "last_run_id"):
        target.last_run_id = run_id
    if hasattr(target, "last_candidates_run"):
        target.last_candidates_run = str(run_path.relative_to(target.root_dir))
    try:
        target.save_state()
    except Exception:
        pass

    # --- Optional plotting (kept here, not in helper) ---
    if cfg.plot_events and len(bboxes)>0:
        # minimal shim compatible with plot_lc_with_bboxes
        shim = type("Obj", (), {})()
        shim.time = type("Arr", (), {"value": total_time})()
        shim.flux = type("Arr", (), {"value": total_flux})()

        for boxes in bboxes:
            T0 = float(boxes[1])
            Tdur = float(boxes[3])

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.set_xlim(T0 - 2*Tdur, T0 + 2*Tdur)
            ax.scatter(total_time, total_flux, color="k", s=6, zorder=10)
            plot_lc_with_bboxes(shim, bboxes, ax=ax)
            plt.show()

    # --- Compatibility return frames (keep for now) ---
    # This is your old "planet_df" style output; we leave it so callers/tests won't break.
    column_names = ["TICID", "planet_name", "period", "T0", "Tdur", "depth"]
    if not run_1:
        column_names.append("SNR")

    event_df = pd.DataFrame(columns=column_names)
    for k, e in enumerate(events, start=1):
        row = [ticid, k, np.inf, float(e.t0_days), float(e.duration_days), float(e.depth)]
        if not run_1:
            row.append(np.nan if e.snr is None else float(e.snr))
        event_df.loc[len(event_df)] = row

    params_df = pd.DataFrame()  # keep empty for quick pass

    return event_df, params_df
    


# def singles_search(target,*, cfg=SinglesSearchConfig(), run_1=True,
#                    intransit=[], pass_label="pass1",
#                    run_id=None, run_path=None):


#     print("confidence level: ", cfg.confidence)

#     ticid = target.ticid
#     total_csv = find_total_csv(target.root_dir, cfg.flavour)  # existing helper

#     df = pd.read_csv(total_csv).dropna(subset=["FLUX"])
#     total_time = df["TIME"].to_numpy(dtype=float)
#     total_flux = df["FLUX"].to_numpy(dtype=float)


#     if "FLUX_ERR" in df.columns:
#         total_flux_err = df["FLUX_ERR"].to_numpy(dtype=float)
#     else:
#         # keep your pragmatic fallback (constant scatter) 
#         total_flux_err = np.full_like(total_flux, np.nanstd(total_flux))

    
#     print('SINGLES SEARCH')
#     column_names = ['TICID', 'planet_name', 'period', 'T0', 'Tdur', 'depth', 'SNR']

#     planet_df = pd.DataFrame(columns=column_names)
    
#     if len(intransit)>0:
# #         print('evil bs', intransit, len(intransit), len(np.where(intransit)[0]))
#         total_time = total_time[~intransit]
#         total_flux = total_flux[~intransit]
#         total_flux_err = total_flux_err[~intransit]

#     indexes_split_unorganize = breaking_up_data(total_time, break_val = 0.5, min_size = 1.)  
#     all_good_indxs = []

#     if len(indexes_split_unorganize)>1:
#         diff_ary = np.array([max(np.array(total_time)[x])-min(np.array(total_time)[x]) for x in indexes_split_unorganize])
#         all_good_indxs =np.concatenate(list(itertools.compress(indexes_split_unorganize, diff_ary>1))).ravel()
   
#     if len(all_good_indxs) == 0: 
#         all_good_indxs = list(range(len(total_time)))
        
# #     print('all good indexes (i.e., itertools result: ', type(all_good_indxs), len(total_time))
#     list_1 = list(all_good_indxs)
#     list_2 = []
#     for index, value in enumerate(total_time):
#         list_2.append(index)


#     total_time = total_time[all_good_indxs]
#     total_flux = total_flux[all_good_indxs]
#     total_flux_err =total_flux_err[all_good_indxs]

# #     print('val', ab)
    
#     params_df = []

#     if len(total_time)>0:
#         print('time length', len(total_time))

#         bboxes = DT_analysis(total_time, total_flux, total_flux_err, cfg.confidence)
#     #     print('ran bboxes', bboxes)

#         print('number singles found', len(bboxes))
#         # t0_singles, dur_singles, depth_singles = [],[],[]

#         events = []


#         if len(bboxes)>0:
#             target.dt_prelim_found = True
#             n_events = len(bboxes)
#             print('number of events: ', n_events)

#             for boxes in bboxes:
#                 events.append(
#                     TransitEvent(
#                         t0_days=float(boxes[1]),
#                         duration_days=float(boxes[3]),
#                         depth=float(1 - boxes[4]),
#                         snr=None
#                     )
#                 )

#     target.quick_singles_t0 = sorted({float(e.t0_days) for e in events})
#     target.save_state()  # persists dt_prelim_found + quick_singles_t0

#     # choose run_id/run_path (finalize should pass these)
#     if run_id is None:
#         run_id = target.new_run_id()

#     run_path = target.candidates_run_path(run_id)    
    
#     payload_update = {
#         "run_id": run_id,
#         "ticid": int(target.ticid),
#         "gaia_id": target.gaia_id,
#         "total_csv": str(total_csv),
#         "dt_config": {"flavour": cfg.flavour, "confidence": cfg.confidence},
#         "dt_events_raw_{pass_label}": [e.to_dict() for e in events],
#     }

#     upsert_run_json(run_path, payload_update)

#     # Optionally store pointers if you added them earlier (safe if missing)
#     if hasattr(target, "last_run_id"):
#         target.last_run_id = run_id
#     if hasattr(target, "last_candidates_run"):
#         target.last_candidates_run = str(run_path.relative_to(target.root_dir))
#     try:
#         target.save_state()
#     except Exception:
#         pass

#     # --- Optional plotting (kept here, not in helper) ---
#     if cfg.plot_events and bboxes:
#         # minimal shim compatible with plot_lc_with_bboxes
#         shim = type("Obj", (), {})()
#         shim.time = type("Arr", (), {"value": total_time})()
#         shim.flux = type("Arr", (), {"value": total_flux})()

#         for boxes in bboxes:
#             T0 = float(boxes[1])
#             Tdur = float(boxes[3])

#             fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#             ax.set_xlim(T0 - 2*Tdur, T0 + 2*Tdur)
#             ax.scatter(total_time, total_flux, color="k", s=6, zorder=10)
#             plot_lc_with_bboxes(shim, bboxes, ax=ax)
#             plt.show()

#     # --- Compatibility return frames (keep for now) ---
#     # This is your old "planet_df" style output; we leave it so callers/tests won't break.
#     column_names = ["TICID", "planet_name", "period", "T0", "Tdur", "depth"]
#     if not run_1:
#         column_names.append("SNR")

#     event_df = pd.DataFrame(columns=column_names)
#     for k, e in enumerate(events, start=1):
#         row = [ticid, k, np.inf, float(e.t0_days), float(e.duration_days), float(e.depth)]
#         if not run_1:
#             row.append(np.nan if e.snr is None else float(e.snr))
#         event_df.loc[len(event_df)] = row

#     params_df = pd.DataFrame()  # keep empty for quick pass

#     return event_df, params_df


