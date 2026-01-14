
# singles_search.py
"""
SinglesSearch orchestrator

Thin wrapper around Mallory's existing single-event detector (e.g., DT_analysis).
Converts detected boxes into Candidate objects with t0, duration, and depth.
The period is set to a safe large value to stabilize a_rs priors in downstream fits.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from search_core import LightCurve, Candidate

# Canonical detector; replace import path if it lives elsewhere in your project
from Functions_all import DT_analysis  # DeepTransit analysis that returns bounding boxes


@dataclass
class SinglesSearchConfig:
    confidence: float = 0.55     # DT confidence threshold
    verbose: bool = True
    set_period_to: Optional[float] = None  # override period guess; None → lc.baseline_days or 27.8 d fallback


class SinglesSearch:
    """
    Detect single-transit events and convert them into Candidate objects.
    """

    def __init__(self, lc: LightCurve, cfg: Optional[SinglesSearchConfig] = None):
        self.lc = lc
        self.cfg = cfg or SinglesSearchConfig()

    def detect(self) -> List[Candidate]:
        """
        Run DT_analysis (or your single-event detector) and return Candidate list.
        """
        # DT_analysis signature may differ in your codebase; adjust args as needed
        bboxes = DT_analysis(
            self.lc.time,
            self.lc.flux,
            self.lc.flux_err,
            confidence=self.cfg.confidence,
            DT_Quite=not self.cfg.verbose,
            is_flat=True,
        )

        # Period guess: use LC baseline or a safe default if not provided
        if self.cfg.set_period_to is not None:
            per_guess = float(self.cfg.set_period_to)
        else:
            per_guess = self.lc.baseline_days if self.lc.baseline_days > 0 else 27.8

        events: List[Candidate] = []
        for boxes in bboxes:
            # Adjust indices if your DT_analysis returns a different layout
            # Here we assume: boxes[1] = t0, boxes[3] = duration, boxes[4] = transit depth complement
            t0 = float(boxes[1])
            dur = float(boxes[3])
            depth = float(1.0 - boxes[4])  # convert DT "complement" to fractional depth

            events.append(Candidate(period=per_guess, t0=t0, duration=dur, depth=depth))

        return events

    def run(self) -> List[Candidate]:
        return self.detect()
