# core/periodic_event.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from utils.handling_data import normalize_depth_to_fractional


@dataclass
class PeriodicEvent:
    """
    One periodic detection from BLS (event-level record).
    This is NOT yet a final PlanetCandidate; it becomes one only after fit/selection.
    """
    period_days: float
    t0_days: float

    duration_days: Optional[float] = None
    depth: Optional[float] = None

    snr: Optional[float] = None
    sde: Optional[float] = None
    n_transits_obs: Optional[int] = None  # can be filled later
    transit_times_days: Optional[list[float]] = None


    # Optional provenance/debug info
    source: str = "BLS"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_days": float(self.period_days),
            "t0_days": float(self.t0_days),
            "duration_days": None if self.duration_days is None else float(self.duration_days),
            "depth": None if self.depth is None else float(self.depth),
            "snr": None if self.snr is None else float(self.snr),
            "sde": None if self.sde is None else float(self.sde),
            "n_transits_obs": None if self.n_transits_obs is None else int(self.n_transits_obs),
            "transit_times_days": None if self.transit_times_days is None else [float(t) for t in self.transit_times_days],
        
            "source": self.source,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PeriodicEvent":
        return cls(
            period_days=float(d["period_days"]),
            t0_days=float(d["t0_days"]),
            duration_days=None if d.get("duration_days") is None else float(d["duration_days"]),
            depth=None if d.get("depth") is None else normalize_depth_to_fractional(d["depth"]),
            snr=None if d.get("snr") is None else float(d["snr"]),
            sde=None if d.get("sde") is None else float(d["sde"]),
            n_transits_obs=None if d.get("n_transits_obs") is None else int(d["n_transits_obs"]),
            transit_times_days=None if d.get("transit_times_days") is None else [float(t) for t in d["transit_times_days"]],
            source=str(d.get("source", "BLS")),
            notes=str(d.get("notes", "")),
        )