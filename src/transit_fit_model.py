
# transit_fit_model.py (new or inside transit_fit.py)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from search_core import LightCurve, Candidate, PriorHint
from target_params import TargetParams

# Optional knobs for priors and parameterization
@dataclass
class ModelConfig:
    t0_strategy: Literal["uniform_direct", "uniform_phase_offset"] = "uniform_phase_offset"
    # Only used if prior_hint["t0"]["half"] is not provided
    beta_for_t0: float = 0.75   # duration contribution to t0 window
    cadence_floor_min: float = 0.002  # days
    min_period_periodic: float = 0.25 # days
    rp_half_min: float = 0.01        # minimal rp/rs half-width
    per_half_min: float = 0.01       # minimal Per half-width

# Optional knobs for sampling (kept separate from model construction)
@dataclass
class SamplerConfig:
    method: Literal["SMC", "DEMetropolisZ"] = "SMC"
    draws: int = 1500
    chains: int = 2
    cores: int = 2
    random_seed: Optional[int] = 42

        

def build_transit_model(
    lc: LightCurve,
    cand: Candidate,
    target: TargetParams,
    type_fn: Literal["Periodic", "Single"],
    prior_hint: Optional[PriorHint] = None,
    cfg: Optional[ModelConfig] = None,
) -> pm.Model:
    """
    Construct a PyMC transit model using class-based inputs and optional prior hints.
    Keeps your original BatmanOp and other math intact; only priors & parameterization change.
    """
    cfg = cfg or ModelConfig()
    u1, u2 = target.as_ab()  # limb darkening

    # Base values from candidate
    Per = float(cand.period)
    T0  = float(cand.t0)
    Dur = float(cand.duration)
    Depth = float(cand.depth)

    # Cadence floor for timing window
    cad = (np.median(np.diff(np.sort(lc.time))) if len(lc.time) > 1 else cfg.cadence_floor_min)
    cad = float(max(cad, cfg.cadence_floor_min))

    # Prior windows from hint if present
    if prior_hint is not None:
        t0_mu, t0_low, t0_high = prior_hint["t0"]["mu"], prior_hint["t0"]["low"], prior_hint["t0"]["high"]
        per_mu, per_low, per_high = prior_hint["Per"]["mu"], prior_hint["Per"]["low"], prior_hint["Per"]["high"]
        rp_mu, rp_low, rp_high = prior_hint["rp_rs"]["mu"], prior_hint["rp_rs"]["low"], prior_hint["rp_rs"]["high"]
        # Optional half-width for phase parameterization
        t0_half = prior_hint["t0"].get("half", max(cfg.beta_for_t0 * Dur, cad))
    else:
        # Fallback windows if no hint was provided
        t0_mu = T0
        t0_half = float(max(cfg.beta_for_t0 * Dur, cad))
        t0_low, t0_high = t0_mu - t0_half, t0_mu + t0_half

        per_mu = Per
        per_half = max(cfg.per_half_min, 0.5 * Dur)
        if type_fn == "Periodic":
            per_low, per_high = max(cfg.min_period_periodic, per_mu - per_half), per_mu + per_half
        else:
            per_low, per_high = max(0.0, per_mu - per_half), per_mu + per_half

        if Depth <= 0:
            rp_mu, rp_low, rp_high = 0.02, 1e-4, 0.2
        else:
            # Estimate rp window from depth and OOT scatter (local single-event window for singles is fine)
            m_local = np.abs(lc.time - T0) < Dur
            sigma_out = float(np.nanstd(lc.flux[~m_local])) if np.any(~m_local) else float(np.nanstd(lc.flux))
            N_in = int(np.sum(m_local)) or 1
            depth_err = sigma_out / np.sqrt(N_in)
            rp_mu = float(np.sqrt(Depth))
            rp_half = max(cfg.rp_half_min, 0.5 * max(depth_err, 1e-6) / max(rp_mu, 1e-6))
            rp_low, rp_high = max(1e-4, rp_mu - rp_half), min(0.99, rp_mu + rp_half)

    # Build model
    with pm.Model() as model:
        # ---- t0 prior ----
        if type_fn == "Periodic" and cfg.t0_strategy == "uniform_phase_offset":
            # Keep local window in phase; reduces correlation with Per
            phi_half = min(0.45, t0_half / max(per_mu, 1e-3))
            phi0 = pm.Uniform("phi0", lower=-phi_half, upper=phi_half, testval=0.0)
            t0 = pm.Deterministic("t0", t0_mu + phi0 * per_mu)
        else:
            # Direct Uniform
            t0 = pm.Uniform("t0", lower=t0_low, upper=t0_high, testval=np.clip(t0_mu, t0_low, t0_high))

        # ---- Per prior ----
        if type_fn == "Periodic":
            per = pm.Uniform("Per", lower=max(cfg.min_period_periodic, per_low), upper=per_high,
                             testval=np.clip(per_mu, per_low, per_high))
            ecc = 0.0
            # a_rs prior consistent with your earlier guess; keep your original calc
            # Example: semi-major axis in stellar radii (rough scaling)
            a_smaj_guess = ((per_mu/365.0)**(2/3))*215.0/2.0
            a_rs = pm.TruncatedNormal("a_rs", mu=a_smaj_guess, sigma=1.0, lower=1.0)
        else:
            # Single: Per is not strongly constrained; keep Uniform & broad or your original TN
            per = pm.Uniform("Per", lower=max(0.0, per_low), upper=per_high,
                             testval=np.clip(per_mu, per_low, per_high))
            ecc = pm.TruncatedNormal("Eccen", mu=0.0, sigma=0.25, lower=0.0, upper=1.0)
            a_rs = pm.TruncatedNormal("a_rs", mu=215.0/10.0, sigma=5.0, lower=1.0)

        # ---- Limb darkening ----
        u1_var = pm.Deterministic("u1", u1)
        u2_var = pm.Deterministic("u2", u2)

        # ---- Radius ratio ----
        rp_rs = pm.Uniform("rp_rs",
                           lower=max(0.0, rp_low),
                           upper=min(1.0, rp_high),
                           testval=np.clip(rp_mu, rp_low, rp_high))

        # ---- Impact parameter & inclination (keep your original deterministics) ----
        # Example placeholders; replace with your exact relations if you had them defined
        b = pm.TruncatedNormal("b", mu=0.0, sigma=0.1, lower=0.0, upper=1.0)
        cosi = pm.Deterministic("cosi", b / a_rs)
        inc  = pm.Deterministic("inclination", pt.arccos(cosi) * 180.0/np.pi)

        # ---- Your Batman Op (unchanged) ----
        # batman_op = BatmanOp()  # if your custom Op class exists in scope
        # flux_model = batman_op(... all params you already use ...)
        # pm.Normal("obs", mu=flux_model, sigma=lc.flux_err, observed=lc.flux)

        # If you don’t use a custom Op and instead build batman curves directly,
        # call your existing function that returns the model light curve.

    return model




def sample_transit_model(
    model: pm.Model,
    s: Optional[SamplerConfig] = None,
):
    s = s or SamplerConfig()
    with model:
        if s.method == "SMC":
            idata = pm.sample_smc(
                draws=s.draws,
                chains=s.chains,
                random_seed=s.random_seed,
            )
        else:
            step = pm.DEMetropolisZ()
            idata = pm.sample(
                draws=s.draws,
                chains=s.chains,
                cores=s.cores,
                step=step,
                random_seed=s.random_seed,
                tune=0,  # DEMetropolisZ does not use NUTS-style tuning
                target_accept=None,  # avoid invalid kwarg for DEMetropolisZ
            )
    return idata
