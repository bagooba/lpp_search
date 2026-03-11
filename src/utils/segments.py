# utils/segments.py
# -*- coding: utf-8 -*-
"""
Common time-series segmentation helpers.

These are factored out because they're used by *both* the singles and periodic
search steps. The logic mirrors your prior helpers in Functions_all.py:
- find_breaks(time, val)
- breaking_up_data(time, break_val=27., min_size=0.5)

Reference for original implementations: Functions_all.py  (Mallory’s code).
"""

from __future__ import annotations
from typing import List
import numpy as np


def find_breaks(time: np.ndarray, val: float = 27.0) -> np.ndarray:
    """
    Return the indices AFTER which a break larger than `val` days occurs.
    Mirrors the original: diffs of sorted times, threshold on gap length.
    """
    time = np.array(time)[np.argsort(time)]
    gaps = np.diff(time)
    inds = np.where(gaps > val)[0]
    return inds + 1


def breaking_up_data(time: np.ndarray, break_val: float = 27.0, min_size: float = 0.5) -> List[np.ndarray]:
    """
    Split an observation timeline into contiguous index blocks (segments).

    Parameters
    ----------
    time : array-like
        Timestamps (days). Will be sorted internally.
    break_val : float
        Gap (days) that defines a break between segments.
    min_size : float
        Minimum time span (days) for a segment to be kept.

    Returns
    -------
    List[np.ndarray]
        List of index arrays (into the *sorted* time array).
    """
    time = np.array(time)
    # original logic: prepend 0, append len(time), and split
    brk = np.append(np.append([0], find_breaks(time, break_val)), [len(time)])
    indexes: List[np.ndarray] = []
    for i in range(len(brk) - 1):
        r = np.arange(brk[i], brk[i + 1], 1)
        if len(r) > 1:
            if np.ptp(time[r]) > min_size:
                indexes.append(r)
    return indexes