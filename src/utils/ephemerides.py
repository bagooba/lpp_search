from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def phase_fold(time, t0, P):
    """Return phase in [-0.5, 0.5)."""
    return ((time - t0 + 0.5 * P) % P) / P - 0.5


def transit_mask(time: np.ndarray, period: float, duration: float, t0: float, buffer: float) -> np.ndarray:
    """
    Full-length transit mask (days). Mirrors your legacy modulo-based mask.
    """
    t = np.asarray(time, dtype=float)
    P = float(period)
    return np.abs(phase_fold(t, t0, P)) < (duration + buffer)




def _nearest_epoch_time(t: float, t0: float, P: float) -> float:
    """Nearest transit center predicted by (t0, P) to time t."""
    n = int(np.rint((t - t0) / P))
    return t0 + n * P

