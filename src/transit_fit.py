
# transit_fit.py
"""
Transit fitting wrapper for periodic candidates.

This module provides a small class that:
- accepts LightCurve + limb-darkening (ab)
- converts List[Candidate] to arrays for existing fitter
- builds or passes prior hints (optional)
- delegates to canonical Functions_all implementations under the hood

No algorithms are re-written here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from search_core import LightCurve, Candidate, PriorHint, prior_hint_from_midtimes, prior_hint_from_duration, compute_intransit_masks, compute_single_masks

# Canonical implementations you already have
from Functions_all import fitting_periodic_planets as _fitting_periodic_planets
from Functions_all import pymc_new_general_function as _pymc_new_general_function  # used only if you want direct calls



from target_params import TargetParams



# -----------------------------
# Configuration for fitting
# -----------------------------
@dataclass
class FittingConfig:
    verbose: bool = True
    save_phasefold: bool = False
    use_ols_hints: bool = True     # True → use mid-times + OLS; False → duration+cadence only
    t0_beta: float = 0.75          # duration-based component for t0 Uniform window
    buffer_days: float = 1.0/6.0   # buffer used when building in-transit masks


# -----------------------------
# Fitter wrapper
# -----------------------------
class PeriodicFitter:
    """
    Class-friendly wrapper for Mallory's existing PyMC fitting code.

    Responsibilities:
    - hold LC + limb darkening (ab)
    - build per-candidate prior hints when requested
    - recompute in-transit masks on-demand from candidates
    - delegate to canonical fitter in Functions_all
    """


    def __init__(self, lc: LightCurve, target: TargetParams, cfg: Optional[FittingConfig] = None):
        self.lc = lc
        self.target = target
        self.ab = target.as_ab()
        self.cfg = cfg or FittingConfig()

    # ---------- Prior hints (optional) ----------
    def build_prior_hints(self, candidates: List[Candidate]) -> List[PriorHint]:
        """
        Build one hint per candidate using either OLS mid-times or duration+cadence.
        """
        hints: List[PriorHint] = []
        for c in candidates:
            if self.cfg.use_ols_hints and c.midtimes is not None and len(c.midtimes) >= 2:
                hints.append(prior_hint_from_midtimes(self.lc.time, self.lc.flux, c, beta=self.cfg.t0_beta))
            else:
                hints.append(prior_hint_from_duration(self.lc.time, self.lc.flux, c, beta=self.cfg.t0_beta))
        return hints

    # ---------- In-transit masks ----------
    def build_masks(self, candidates: List[Candidate]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute per-candidate and union in-transit masks from candidates.
        """
        return compute_intransit_masks(self.lc.time, candidates, buffer=self.cfg.buffer_days)

    # ---------- Fit entry point (delegates to your canonical fitter) ----------
    def fit(self, candidates: List[Candidate], prior_hints: Optional[List[PriorHint]] = None):
        """
        Convert Candidate list to arrays, compute union mask, and call your canonical fitter.
        Prior hints are optional. If omitted, your current fitter runs as-is.
        """
        periods   = [c.period for c in candidates]
        t0s       = [c.t0     for c in candidates]
        depths    = [c.depth  for c in candidates]

        # Build masks from candidates
        _, union_mask = self.build_masks(candidates)

        # If your canonical fitter signature doesn’t accept prior_hints/durations yet,
        # we call it unchanged. When you add those params, pass them through here.
        return _fitting_periodic_planets(
            time=self.lc.time,
            flux=self.lc.flux,
            flux_err=self.lc.flux_err,
            pers=periods,
            t0s=t0s,
            depths=depths,
            ab=self.ab,
            intransit=union_mask,
            verbose=self.cfg.verbose,
            save_phaseFold=self.cfg.save_phasefold,
            data_file='.',
        )

    # ---------- Optional: direct model call for one candidate ----------
    def build_and_call_model_for_one(self, candidate: Candidate, prior_hint: Optional[PriorHint] = None):
        """
        Optional convenience when experimenting with one candidate:
        call pymc_new_general_function directly for a single fit.
        """
        # This uses your canonical builder unchanged, with a single prior_hint if desired.
        return _pymc_new_general_function(
            time=self.lc.time,
            flux=self.lc.flux,
            unc=self.lc.flux_err,
            T0=candidate.t0,
            other_pars=[candidate.period, self.ab, candidate.depth],
            type_fn='Periodic',
            verbose=self.cfg.verbose,
            keep_ld_fixed=True,
            phase_fold=False,
            prior_hint=prior_hint,   # only works if your canonical builder supports it
        )


# transit_fit.py (add below PeriodicFitter)
"""
Transit fitting wrappers for periodic and single events.
"""


@dataclass
class SinglesFittingConfig:
    verbose: bool = True
    t0_beta: float = 1.0          # singles often use a slightly wider t0 window
    buffer_days: float = 1.0/6.0  # mask buffer for single events
    keep_ld_fixed: bool = True    # pass-through to your PyMC builder
    save_phasefold: bool = False  # not used here; kept for parity


class SinglesFitter:
    """
    Class-friendly wrapper for fitting single-transit candidates.

    Responsibilities:
    - hold LC
    - build duration+cadence prior hints for t0 and rp_rs
    - compute single-event masks on demand (if you add masking to your likelihood later)
    - call your canonical PyMC builder for each candidate with type_fn='Single'
    """


    def __init__(self, lc: LightCurve, target: TargetParams, cfg: Optional[SinglesFittingConfig] = None):
        self.lc = lc
        self.target = target
        self.ab = target.as_ab()
        self.cfg = cfg or SinglesFittingConfig()

    def build_prior_hints(self, candidates: List[Candidate]) -> List[PriorHint]:
        """
        Use duration+cadence-only hints for singles. OLS mid-times are not defined for single events.
        """
        hints: List[PriorHint] = []
        for c in candidates:
            hints.append(prior_hint_from_duration(self.lc.time, self.lc.flux, c, beta=self.cfg.t0_beta))
        return hints

    def build_masks(self, candidates: List[Candidate]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute single-event masks per candidate and their union (optional for plotting/use later).
        """
        return compute_single_masks(self.lc.time, candidates, buffer=self.cfg.buffer_days)

    def fit(self, candidates: List[Candidate], prior_hints: Optional[List[PriorHint]] = None):
        """
        Loop over candidates and call your canonical PyMC builder for singles.
        Returns a list of per-candidate results (params_df, conv, conv_attempt).
        """
        if prior_hints is None:
            prior_hints = self.build_prior_hints(candidates)

        results = []
        for c, hint in zip(candidates, prior_hints):
            # Build one single fit using your canonical builder
            out = _pymc_new_general_function(
                time=self.lc.time,
                flux=self.lc.flux,
                unc=self.lc.flux_err,
                T0=c.t0,
                other_pars=[c.period, self.ab, c.depth],
                type_fn='Single',
                verbose=self.cfg.verbose,
                keep_ld_fixed=self.cfg.keep_ld_fixed,
                phase_fold=False,
                prior_hint=hint,  # Uniform t0 and rp_rs windows; your builder handles singles branch
            )
            results.append(out)

        return results

__all__ = ["FittingConfig", "PeriodicFitter"]
