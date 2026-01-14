
# periodic_search.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from search_core import LightCurve, Candidate, compute_intransit_masks  # <- helper we just added
from Functions_all import breaking_up_data  # generic utility you already use


from target_params import TargetParams

# ---------- Periodic-exclusive adapter (class-compatible signature) ----------
def using_BLS_recursive(
    lc: LightCurve,
    verbose: bool = True,
    plot: bool = True,
    max_planets: int = 10,
    min_SDE: int = 10,
    seeds: Optional[List[Candidate]] = None,
    first: bool = False,
) -> Tuple[List[Candidate], np.ndarray]:
    """
    Adapter around your existing using_BLS_recursive that:
      - takes LightCurve (arrays live here)
      - optionally continues recursion from seed Candidates
      - returns List[Candidate] AND the intransit mask produced by your original code

    No LC state mutation; the caller decides whether to use the returned mask
    or to recompute a union from the Candidate list (preferred).
    """
    # Prepare seed arrays if given
    periods_seed = [c.period for c in seeds] if seeds else None
    t0_seed      = [c.t0     for c in seeds] if seeds else None
    tdur_seed    = [c.duration for c in seeds] if seeds else None
    depth_seed   = [c.depth  for c in seeds] if seeds else None

    # ---- PASTE REGION START ----------------------------------------------------------
    # For now, call through to your current implementation in Functions_all.
    # When ready, paste your full function body here and delete the import+call.
    from Functions_all import using_BLS_recursive as _using_BLS_recursive  # noqa
    periods, T0s, Tdur, depths, intransit = _using_BLS_recursive(
        time=lc.time,
        flux=lc.flux,
        flux_err=lc.flux_err,
        intransit=np.zeros(len(lc.time), dtype=bool),
        verbose=verbose,
        plot=plot,
        max_planets=max_planets,
        min_SDE=min_SDE,
        periods=periods_seed,
        T0=t0_seed,
        Tdur=tdur_seed,
        depths=depth_seed,
        first=first,
    )
    # ---- PASTE REGION END ------------------------------------------------------------

    # Wrap into Candidate objects
    candidates: List[Candidate] = [
        Candidate(period=float(p), t0=float(t0), duration=float(dur), depth=float(dpt))
        for p, t0, dur, dpt in zip(periods, T0s, Tdur, depths)
    ]

    # Return candidates and the original intransit mask (caller may recompute a union)
    return candidates, intransit


# ---------- Orchestrator (BLS-only; TLS removed) ----------
@dataclass
class PeriodicSearchConfig:
    verbose: bool = True
    save_phasefold: bool = False
    run_chunked: bool = False
    break_val_days: float = 27.8
    # You can add your SDE/BIC thresholds here later and thread them into the adapter


class PeriodicSearch:

    def __init__(self, lc: LightCurve, target: TargetParams, cfg: Optional[PeriodicSearchConfig] = None):
        self.lc = lc
        self.target = target
        self.ab = target.as_ab()  # limb-darkening tuple
        self.cfg = cfg or PeriodicSearchConfig()

    def detect_full(self, seeds: Optional[List[Candidate]] = None) -> Tuple[List[Candidate], np.ndarray]:
        """
        Run the BLS recursion on the full LC and return candidates and the original mask.
        """
        candidates, mask_from_bls = using_BLS_recursive(
            lc=self.lc,
            verbose=self.cfg.verbose,
            plot=False,
            seeds=seeds,
            first=True,
        )
        return candidates, mask_from_bls

    def build_intransit_masks(self, candidates: List[Candidate], buffer: float = 1.0/6.0) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Preferred path: recompute per-candidate masks and their union from the Candidate list.
        """
        return compute_intransit_masks(self.lc.time, candidates, buffer=buffer)

    def fit_full(self, candidates: List[Candidate], intransit_union: np.ndarray):
        """
        Fitting function expects arrays; derive from Candidate list and pass union mask.
        """
        periods   = [c.period for c in candidates]
        t0s       = [c.t0     for c in candidates]
        depths    = [c.depth  for c in candidates]

        # Call your existing fitter exactly as-is
        from Functions_all import fitting_periodic_planets as _fitting_periodic_planets  # noqa
        return _fitting_periodic_planets(
            time=self.lc.time,
            flux=self.lc.flux,
            flux_err=self.lc.flux_err,
            pers=periods,
            t0s=t0s,
            depths=depths,
            ab=self.ab,
            intransit=intransit_union,
            verbose=self.cfg.verbose,
            save_phaseFold=self.cfg.save_phasefold,
            data_file='.',
        )

    def run_chunked(self, buffer: float = 1.0/6.0):
        """
        Segment the LC, run detection on each segment, and fit locally.
        Masks are recomputed from each segment’s candidates via the same helper.
        """
        idx_splits = breaking_up_data(self.lc.time, break_val=self.cfg.break_val_days)
        results_per_segment = []
        intransit_union = np.full(len(self.lc.time), False)

        for idx in idx_splits:
            if len(idx) < 2:
                continue

            t_seg = self.lc.time[idx]
            f_seg = self.lc.flux[idx]
            e_seg = self.lc.flux_err[idx]

            seg_lc = LightCurve(time=t_seg, flux=f_seg, flux_err=e_seg)
            seg_cands, _mask_from_bls = using_BLS_recursive(seg_lc, verbose=self.cfg.verbose, plot=False, first=False)
            if len(seg_cands) == 0:
                continue

            # Recompute segment masks from candidates
            _, seg_union = compute_intransit_masks(t_seg, seg_cands, buffer=buffer)

            # Fit on the segment (unchanged fitter)
            from Functions_all import fitting_periodic_planets as _fitting_periodic_planets  # noqa
            fit_out = _fitting_periodic_planets(
                time=t_seg,
                flux=f_seg,
                flux_err=e_seg,
                pers=[c.period for c in seg_cands],
                t0s=[c.t0 for c in seg_cands],
                depths=[c.depth for c in seg_cands],
                ab=self.ab,
                intransit=seg_union,
                verbose=self.cfg.verbose,
                save_phaseFold=False,
                total_time=self.lc.time,
                data_file='.',
            )
            results_per_segment.append(fit_out)

            # If you later expose per-candidate masks from the fitter, you can OR them here.

        return results_per_segment, intransit_union

    def run(self, seeds: Optional[List[Candidate]] = None, buffer: float = 1.0/6.0):
        """
        Full-LC detect → recompute masks from candidates → fit.
        If run_chunked=True, also execute the per-segment path.
        """
        candidates, mask_from_bls = self.detect_full(seeds=seeds)

        if len(candidates) == 0:
            empty = ([], [], [], [], np.zeros(len(self.lc.time), dtype=bool), [], None)
            if self.cfg.run_chunked:
                seg_results, intransit_union = self.run_chunked(buffer=buffer)
                return empty, (seg_results, intransit_union)
            return empty

        # Preferred: recompute union from candidates (more transparent & reproducible)
        masks, union = self.build_intransit_masks(candidates, buffer=buffer)

        full_fit = self.fit_full(candidates, intransit_union=union)

        if self.cfg.run_chunked:
            seg_results, intransit_union = self.run_chunked(buffer=buffer)
            return full_fit, (seg_results, intransit_union)

        return full_fit
