# utils/handling_data.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.segments import find_breaks, breaking_up_data

import batman

def normalize_depth_to_fractional(depth_val):
    """
    Force depth into (0, 0.5] as a positive fractional transit depth.
    If something arrives as ~0.997 (i.e., flux level), convert to ~0.003.
    """
    if depth_val is None:
        return None
    d = float(depth_val)
    if not np.isfinite(d):
        return None
    d = abs(d)  # guard against negative sign conventions
    if d > 0.5:
        d = 1.0 - d
    return d


def _finite(*xs):
    try:
        return all(np.isfinite(float(x)) for x in xs)
    except Exception:
        return False



def bin_by_time_many_args(time, time_size_of_bins, **params):
    time = np.asarray(time)

    interval = time_size_of_bins / 60. / 24.  # days

    # Precompute bins
    min_t, max_t = np.min(time), np.max(time)
    nbins = int(np.ceil((max_t - min_t) / interval)) + 1
    bins = min_t + np.arange(nbins + 1) * interval

    # Assign each time point to a bin
    bin_idx = np.digitize(time, bins) - 1

    # Prepare output
    new_dict = {}
    new_time = []

    for b in range(nbins):
        mask = bin_idx == b
        if not np.any(mask):
            continue

        new_time.append(np.mean(time[mask]))

        for key, arr in params.items():
            arr = np.asarray(arr)
            if key not in new_dict:
                new_dict[key] = []
            new_dict[key].append(np.mean(arr[mask]))

    return np.array(new_time), new_dict

def sort_arrays_by_time(total_time, *args):
    total_time = np.asarray(total_time)

    # Skip sorting if already monotonic
    if np.all(np.diff(total_time) >= 0):
        return [np.asarray(arg) for arg in args]

    idx = np.argsort(total_time)
    return [np.asarray(arg)[idx] for arg in args]

def sort_arrays_by_index(index_lst, *args):
    return [[arg[index] for index in index_lst] for arg in args]




def predict_lc(time_lc,t0,P,rp_rs,cosi,a,u1,u2,cad):
    oversample = 4
    e = 0.
    omega = np.pi/2.
    inc = np.arccos(cosi)*180./np.pi
    params = batman.TransitParams()
    params.t0  = t0
    params.per = P
    params.rp  = rp_rs
    params.a   = a
    params.inc = inc
    params.ecc = e
    params.w = omega*180./np.pi
    params.u = [u1,u2]
    params.limb_dark = "quadratic"
        
    if not cad>0:
        cad = 30.
    m = batman.TransitModel(params, time_lc ,supersample_factor = oversample, exp_time = cad/24./60.)

    flux_theo = m.light_curve(params)

    return flux_theo


