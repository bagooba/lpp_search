
# target_params.py
"""
TargetParams: star/target metadata container

Stores identifiers and stellar parameters that need to be propagated across
search and fit code. Provides simple validation and JSON serialization.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple
import json
import math


@dataclass
class TargetParams:
    # Identifiers
    ticid: Optional[int] = None
    gaia_id: Optional[str] = None
    name: Optional[str] = None           # common name or nickname

    # Sky position and photometry (optional, use if convenient)
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    distance_pc: Optional[float] = None
    teff: Optional[float] = None         # K
    logg: Optional[float] = None         # cgs
    feh: Optional[float] = None          # dex
    mag_gaiag: Optional[float] = None
    mag_tmag: Optional[float] = None

    # Stellar parameters used in modeling
    mstar: Optional[float] = None        # Msun
    rstar: Optional[float] = None        # Rsun
    rho_star: Optional[float] = None     # g/cm^3 or solar units, document your convention

    # Limb darkening
    u1: Optional[float] = None
    u2: Optional[float] = None

    # Freeform metadata
    notes: Dict[str, Any] = field(default_factory=dict)

    # Versioning or provenance (optional)
    source: Optional[str] = None         # where values came from (e.g., "catalog_x v1.2")
    version: Optional[str] = None        # your internal version tag

    # ---- Convenience methods ----

    def as_ab(self) -> Tuple[float, float]:
        """
        Return limb-darkening tuple (u1, u2). Raises if missing.
        """
        if self.u1 is None or self.u2 is None:
            raise ValueError("Limb darkening (u1, u2) is not set in TargetParams.")
        return float(self.u1), float(self.u2)

    def is_complete_for_periodic_fit(self) -> bool:
        """
        Minimal check for values you typically require before periodic fitting.
        Adjust this to your needs.
        """
        checks = [
            self.u1 is not None,
            self.u2 is not None,
            self.mstar is not None or self.rstar is not None or self.rho_star is not None,
        ]
        return all(checks)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetParams":
        """Construct from a dict."""
        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save parameters to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TargetParams":
        """Load parameters from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def update(self, **kwargs: Any) -> None:
        """
        Update any subset of fields safely (e.g., after a catalog query).
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                # Keep unknowns in notes to avoid losing information
                self.notes[k] = v

    # Optional derived helper if you want a fallback for rho_star
    def ensure_rho_star(self) -> Optional[float]:
        """
        Compute a simple rho_* fallback when mstar and rstar exist.
        You can replace with your preferred units/convention later.
        """
        if self.rho_star is not None:
            return self.rho_star
        if self.mstar is None or self.rstar is None:
            return None
        # Very rough solar-units estimate; keep this only if useful to you
        try:
            rho = float(self.mstar) / (float(self.rstar) ** 3)
            self.rho_star = rho
            return rho
        except Exception:
            return None
``
