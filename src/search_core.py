
"""
search_core.py

Shared types and per-candidate helpers used by detection and fitting:
- LightCurve, Candidate
- transit_mask (local)
- ephemeris_uncertainties_from_transit_times (OLS)
- prior_hint_from_midtimes (per-candidate Uniform windows)
- prior_hint_from_duration (no-OLS alternative, optional)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, TypedDict
import numpy as np

# ---------- Data containers ----------

@dataclass
class LightCurve:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray

    @property
    def cadence_days(self) -> float:
        if len(self.time) < 2:
            return 30.0/60.0/24.0
        return float(np.nanmin(np.diff(np.sort(self.time))))

    @property
    def baseline_days(self) -> float:
        return float(np.ptp(self.time)) if len(self.time) else 0.0


@dataclass
class Candidate:
    period: float
    t0: float
    duration: float
    depth: float
    midtimes: Optional[np.ndarray] = None
    diag: Optional[Dict[str, Any]] = None


# ---------- Types for prior hints ----------

class _ScalarWindow(TypedDict, total=False):
    mu: float
    low: float
    high: float
    half: float  # optional convenience

class _Diag(TypedDict, total=False):
    sigma_t0: float
    sigma_P: float
    n_trans: int
    baseline_days: float

class PriorHint(TypedDict):
    t0: _ScalarWindow
    Per: _ScalarWindow
    rp_rs: _ScalarWindow
    _diag: _Diag


# ---------- Utilities ----------

def transit_mask(t: np.ndarray, period: float, duration: float, t0: float, buffer: float = 0.0) -> np.ndarray:
    phase_dist = np.abs(((t - t0 + 0.5 * period) % period) - 0.5 * period)
    return phase_dist < (float(duration) + float(buffer))

def ephemeris_uncertainties_from_transit_times(transit_times: np.ndarray) -> tuple[float, float, float, float]:
    t = np.asarray(transit_times, dtype=float)
    if t.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    t = np.sort(t)
    E = np.arange(t.size, dtype=float)
    Ebar, Tbar = E.mean(), t.mean()
    SEE = np.sum((E - Ebar)**2)
    SET = np.sum((E - Ebar) * (t - Tbar))
    P_hat  = SET / SEE
    t0_hat = Tbar - P_hat * Ebar
    resid = t - (t0_hat + P_hat * E)
    sigma_eps2 = np.sum(resid**2) / max(1, t.size - 2)
    sigma_P  = np.sqrt(sigma_eps2 / SEE)
    sigma_t0 = np.sqrt(sigma_eps2 * (1.0/t.size + (Ebar**2)/SEE))
    return float(P_hat), float(t0_hat), float(sigma_P), float(sigma_t0)


# ---------- Per-candidate prior hints ----------

def prior_hint_from_midtimes(
    time: np.ndarray,
    flux: np.ndarray,
    cand: Candidate,
    beta: float = 0.75,
    min_period: float = 0.25,
) -> PriorHint:
    P_hat, t0_hat, sigma_P, sigma_t0 = ephemeris_uncertainties_from_transit_times(
        cand.midtimes if cand.midtimes is not None else np.array([], dtype=float)
    )
    cadence_floor = max((np.median(np.diff(np.sort(time))) if len(time) > 1 else 0.02), 0.002)

    if not np.isfinite(P_hat):
        P_hat, t0_hat = float(cand.period), float(cand.t0)
        sigma_P, sigma_t0 = 0.02, cadence_floor

    t0_half = max(5.0 * sigma_t0, beta * float(cand.duration), cadence_floor)
    t0_low, t0_high = t0_hat - t0_half, t0_hat + t0_half

    P_half = max(5.0 * sigma_P, 0.01)
    P_low  = max(min_period, P_hat - P_half)
    P_high = P_hat + P_half

    if cand.depth <= 0:
        rp_mu, rp_low, rp_high = 0.02, 1e-4, 0.2
    else:
        m = transit_mask(time, P_hat, cand.duration, t0_hat)
        N_in = int(np.sum(m)) or 1
        sigma_out = float(np.nanstd(flux[~m])) if np.any(~m) else float(np.nanstd(flux))
        depth_err = sigma_out / np.sqrt(N_in)
        rp_mu = float(np.sqrt(cand.depth))
        half  = max(0.01, 0.5 * max(depth_err, 1e-6) / max(rp_mu, 1e-6))
        rp_low, rp_high = max(1e-4, rp_mu - half), min(0.99, rp_mu + half)

    return {
        "t0":   {"mu": t0_hat, "low": t0_low, "high": t0_high, "half": t0_half},
        "Per":  {"mu": P_hat,  "low": P_low,  "high": P_high,  "half": P_half},
        "rp_rs":{"mu": rp_mu,  "low": rp_low, "high": rp_high, "half": (rp_high - rp_low)/2.0},
        "_diag":{"sigma_t0": sigma_t0, "sigma_P": sigma_P},
    }

def prior_hint_from_duration(
    time: np.ndarray,
    flux: np.ndarray,
    cand: Candidate,
    beta: float = 0.75,
    min_period: float = 0.25,
) -> PriorHint:
    cadence_floor = max((np.median(np.diff(np.sort(time))) if len(time) > 1 else 0.02), 0.002)
    t0_half = max(beta * float(cand.duration), cadence_floor)
    t0_low, t0_high = float(cand.t0) - t0_half, float(cand.t0) + t0_half

    P_half = max(0.01, 0.5 * float(cand.duration))  # conservative
    P_low  = max(min_period, float(cand.period) - P_half)
    P_high = float(cand.period) + P_half

    if cand.depth <= 0:
        rp_mu, rp_low, rp_high = 0.02, 1e-4, 0.2
    else:
        m = transit_mask(time, cand.period, cand.duration, cand.t0)
        N_in = int(np.sum(m)) or 1
        sigma_out = float(np.nanstd(flux[~m])) if np.any(~m) else float(np.nanstd(flux))
        depth_err = sigma_out / np.sqrt(N_in)
        rp_mu = float(np.sqrt(cand.depth))
        half  = max(0.01, 0.5 * max(depth_err, 1e-6) / max(rp_mu, 1e-6))
        rp_low, rp_high = max(1e-4, rp_mu - half), min(0.99, rp_mu + half)

    return {
        "t0":   {"mu": float(cand.t0), "low": t0_low, "high": t0_high, "half": t0_half},
        "Per":  {"mu": float(cand.period), "low": P_low, "high": P_high, "half": P_half},
        "rp_rs":{"mu": rp_mu, "low": rp_low, "high": rp_high, "half": (rp_high - rp_low)/2.0},
    }


__all__ = [
    "LightCurve",
    "Candidate",
    "PriorHint",
    "transit_mask",
    "ephemeris_uncertainties_from_transit_times",
    "prior_hint_from_midtimes",
    "prior_hint_from_duration"
]
