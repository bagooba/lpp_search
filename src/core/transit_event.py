
# core/transit_event.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from utils.handling_data import normalize_depth_to_fractional


@dataclass
class TransitEvent:
    """
    One observed transit-like dip (DT output / event-level record).
    This is NOT a planet candidate yet (no Periodic vs Single decision here).
    """
    t0_days: float
    duration_days: float
    depth: float

    # Optional “quality” fields (useful later)
    snr: Optional[float] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t0_days": float(self.t0_days),
            "duration_days": float(self.duration_days),
            "depth": float(self.depth),
            "snr": None if self.snr is None else float(self.snr),
            "confidence": None if self.confidence is None else float(self.confidence),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransitEvent":
        return cls(
            t0_days=float(d["t0_days"]),
            duration_days=float(d["duration_days"]),
            depth=normalize_depth_to_fractional(d["depth"]),
            snr=None if d.get("snr") is None else float(d["snr"]),
            confidence=None if d.get("confidence") is None else float(d["confidence"]),
        )