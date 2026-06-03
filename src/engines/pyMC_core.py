# engines/pymc_core.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import copy

import batman
import multiprocessing as mp

import pymc as pm
import pytensor.tensor as pt
import arviz as az

from pytensor.graph import Op, Apply
from pytensor import config as pt_config
from core.target import Target

from pathlib import Path

import utils.config as con  # keep con.G only



def extract_summary_dataframe(trace, hdi_prob=0.68):
    summary = az.summary(trace, hdi_prob=hdi_prob)

    median_dataset = trace.posterior.median(dim=["chain", "draw"])
    medians = {var: float(median_dataset[var]) for var in median_dataset.data_vars}
    summary["median"] = medians

    selected_columns = ["mean", "median", "sd", "hdi_16%", "hdi_84%", "r_hat"]
    return summary[selected_columns]


def transit_mask_tensors(t, period, duration, T0, cad_minutes=None):
    phase = pt.abs(((t - T0 + 0.5 * period) % period) - (0.5 * period))
    buffer = 0.0
    if cad_minutes is not None:
        buffer = cad_minutes / (24.0 * 60.0)
    return phase < (0.5 * duration + buffer)

def _last_initvals_from_trace(trace):
    post = trace.posterior
    initvals = {}
    for var in post.data_vars:
        initvals[var] = np.array(post[var].values[0, -1])
    return initvals


def write_converged_fit_csv(target: Target, cand, fit_info):
    out_dir = '/users/malharris/lpp_search/fit_stats/'

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    safe_id = cand.candidate_id().replace("/", "_").replace(" ", "_")
    out_path = Path(out_dir+f"{safe_id}_attempt{int(fit_info['attempt']):02d}.csv")

    row = {
        "ticid": target.ticid,
        "gaia_id": target.gaia_id,
        "candidate_id": cand.candidate_id(),
        "ptype": cand.ptype,
        "attempt": fit_info["attempt"],
        "draws_per_chain": fit_info["draws_per_chain"],
        "tune_per_chain": fit_info["tune_per_chain"],
        "chains": fit_info["chains"],
        "cores": fit_info["cores"],
        "rhat_max": fit_info["rhat_max"],
    }

    pd.DataFrame([row]).to_csv(out_path, index=False)
    return out_path

def sample_until_converged(model, max_attempts=3, rhat_threshold=1.1, chains=4,cores=None, mp_context="forkserver"):
    # Get all free random variables in the model
    
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    print('SLURM CPUs', slurm_cpus)
    if cores is None:
        cores = min(chains, slurm_cpus)

    # cores = min(chains, os.cpu_count() or 1) if cores is None else cores

    free_vars = model.free_RVs
    if not free_vars:
        raise ValueError("No free random variables found for sampling.")
#     print('free vars', free_vars)
    # Use Metropolis for all free RVs
    # step = pm.Metropolis(vars=free_vars)



    prev_trace = None

    for attempt in range(1, 1+max_attempts):
        step = pm.DEMetropolisZ(vars=free_vars)#, target_accept=0.8) 

        run = attempt-1
        draws = 3000
        tune = 4000

        sample_kwargs = dict(
            step=step,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=12345 + attempt,
            return_inferencedata=True,
        )

        if cores > 1 and mp_context is not None:
            sample_kwargs["mp_ctx"] = mp.get_context(mp_context)        
        if prev_trace is not None:
            sample_kwargs["initvals"] = _last_initvals_from_trace(prev_trace)
            sample_kwargs['draws'] = int(draws/2)
            sample_kwargs['tune'] = int(tune/2)

        try:
            
            trace = pm.sample(**sample_kwargs)

        except Exception as e:
            print(f"Sampling crashed: {e}", flush=True)
            raise 


        summary = az.summary(trace)
        if (summary['r_hat'] < rhat_threshold).all():
            print(f"Converged on attempt {attempt}")

            fit_info = {
                "attempt": attempt,
                "draws_per_chain": draws,
                "tune_per_chain": tune,
                "chains": chains,
                "cores": cores,
                "rhat_max": float(summary["r_hat"].max()),
            }

            return trace, fit_info
        prev_trace = trace
#         print('checking nans trace', trace.posterior['SNR'])
        print('checking nans summary', az.summary(trace))
        print(f"Attempt {attempt} failed to converge.")

    raise RuntimeError("Model did not converge after multiple attempts.")

class BatmanOp(Op):

    itypes = [pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def __init__(self):
        pass  # nothing stored → picklable

    # def __init__(self, params):
    #     self.params = params

    def make_node(self, *inputs):
        # Convert all inputs to tensors if they aren't already
        converted_inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        return Apply(self, converted_inputs, [pt.dvector()])



    def perform(self, node, inputs, outputs):
        time, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, cad = inputs

        # params = copy.copy(self.params)

        params = batman.TransitParams()

        params.t0  = float(t0)
        params.per = float(per)
        params.rp  = float(rp_rs)
        params.a   = float(a_rs)
        params.inc = float(inc)
        params.ecc = float(ecc)
        params.w   = 90.0
        params.u   = [float(u1), float(u2)]
        params.limb_dark = "quadratic"

        model = batman.TransitModel(
            params,
            np.asarray(time, dtype=np.float64),
            supersample_factor=4,
            exp_time=cad / 24.0 / 60.0,
        )

                
        flux = model.light_curve(params)

        if not np.all(np.isfinite(flux)):
            outputs[0][0] = np.zeros_like(flux) + 1e6  # force huge misfit
        else:
            outputs[0][0] = flux


    def grad(self, inputs, g_outputs):
        # For now, return zeros (no gradient)
        return [pt.zeros(inp.shape, dtype=pt_config.floatX) for inp in inputs]
    

def prepare_fit_data(time, flux, unc, candidate):
    mask = np.isnan(time) | np.isnan(flux) | np.isnan(unc)
    time, flux, unc = time[~mask], flux[~mask], unc[~mask]

    cad = np.nanpercentile(np.clip(np.diff(np.unique(time))*60.*24., 200/60, 30), 95)  # minutes

    if candidate.ptype == "Single":
        # window around t0 using candidate.duration_days as your scale
        t0 = candidate.t0_days
        dur = candidate.duration_days
        idx = np.where(np.abs(time - t0) < (1.0 + dur))
        return time[idx], flux[idx], unc[idx], cad

    # periodic uses full series
    return time, flux, unc, cad

def median_pytensor(x):
    sorted_x = pt.sort(x)
    n = x.shape[0]
    mid = n // 2
    return pt.switch(
        pt.eq(n % 2, 0),
        (sorted_x[mid - 1] + sorted_x[mid]) / 2.0,
        sorted_x[mid]
    )



def make_windows_from_time_stamps(t, gap_threshold=0.5):
    """
    Convert sorted time stamps (days) into contiguous [start, end] windows.
    gap_threshold is the minimum gap that splits windows (days).
    Pick a value larger than cadence and smaller than any real gap.
    """
    t = np.asarray(t)
    t = t[np.isfinite(t)]
    t = np.sort(t)
    if t.size == 0:
        return np.empty((0, 2))
    gaps = np.diff(t)
    breaks = np.where(gaps > gap_threshold)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends   = np.concatenate((breaks, [t.size - 1]))
    return np.column_stack((t[starts], t[ends]))


def initialize_batman_params(params, t0, per, depth, ld_q):

    params.t0 = t0
    params.per = per
    params.rp = np.sqrt(depth)
    params.a = 10.0  # placeholder unless you have a better estimate
    params.inc = 90.0
    params.ecc = 0.0
    params.w = 90.0
    params.u = ld_q
    params.limb_dark = "quadratic"

    return params



def pymc_fit_candidate(target, candidate, time, flux, unc, verbose=False, keep_ld_fixed=True, max_runs =3):
    # --- star facts from Target ---
    target.load_state()
    if target.rho_star is None:
        print(target._catalog)
        # target.rho_star = target._compute_r
        # raise ValueError("target.rho_star is None. Ensure catalog Mass/Rad exist and rho_star was computed.")
        print("target.rho_star is None. Ensure catalog Mass/Rad exist and rho_star was computed.")

        rho_star = (0.5 / ((0.5)**3)) * (3.0 / (4.0 * np.pi))
    else:
        rho_star = float(target.rho_star)

    u1, u2 = target.ld_u1_u2

    # --- hypothesis from Candidate ---
    type_fn = candidate.ptype
    T0 = float(candidate.t0_days)
    Depth = float(candidate.depth)
    fold_this = False  # default for Single; may be updated in model setup

    # periodic requires a period
    Per_in = getattr(candidate, "period_days", None)
    if type_fn == "Periodic":
        if Per_in is None:
            raise ValueError("Periodic candidate missing period_days.")
        Per_in = float(Per_in)

        
    # Always do data prep first so cad exists
    time, flux, unc, cad = prepare_fit_data(time, flux, unc, candidate)

    # Then validate
    if candidate.depth is None or candidate.duration_days is None:
        raise ValueError("Candidate missing depth or duration_days.")

    # pTdur is your “window scale”. Keep your old convention unless you add a dedicated field later.
    pTdur = 1.5 * float(candidate.duration_days)


    # count observed transits. Override from candidate if available.
    nobs_est = 1
    if type_fn == "Periodic":
        nobs_est = getattr(candidate, "n_transits_obs", None)
        print('nobs_est from candidate:', nobs_est, 'period: ', Per_in)
        if nobs_est == 1:
            windows = make_windows_from_time_stamps(np.array(time), gap_threshold=0.5)
            tmp = 0
            for s, e in windows:
                k_low  = np.ceil((s - T0) / Per_in)
                k_high = np.floor((e - T0) / Per_in)
                tmp += int(max(0, k_high - k_low + 1))
            nobs_est = tmp
        nobs_est = int(nobs_est)
    if nobs_est >= 3:
        fold_this = True
    ecc = 0.0

    # batman_params = initialize_batman_params(batman.TransitParams(), T0, Per_in if type_fn == "Periodic" else np.ptp(time), Depth, [float(u1), float(u2)])

    if fold_this:
        folded_phase = ((time - T0 + 0.5 * Per_in) % Per_in) - (0.5 * Per_in)
        sort_indx = np.argsort(folded_phase)

        phase = folded_phase[sort_indx]
        
        # window = min(0.5, 3*pTdur + 0.1*Per_in)
        window = min(0.5, 3*pTdur + 0.05*Per_in)

        use_index = np.abs(phase) < window

        dt_minutes_min = np.nanpercentile(np.diff(np.unique(np.sort(time))), 5) * 24.0 * 60.0
        cad = float(np.clip(dt_minutes_min, 0.2, 60.0))



        t_fit = phase[use_index] + T0
        f_fit = flux[sort_indx][use_index]
        u_fit = unc[sort_indx][use_index]
    else:
        t_fit = time
        f_fit = flux
        u_fit = unc


        # p_flux_model = batman_op(t_fit, t0, per, rp_rs, a_rs, inc, ...)


    # batman_model = batman.TransitModel(
    #     batman_params,
    #     t_fit,
    #     supersample_factor=4,
    #     exp_time=cad / 24.0 / 60.0,)
    if len(t_fit)<2:
        return None, False, None
    # batman_op = BatmanOp(batman_params)
    batman_op = BatmanOp()
    with pm.Model() as model:
        t0 = pm.Uniform("t0", lower=T0 - pTdur, upper=T0 + pTdur)

        if type_fn == "Single":
            # initial a/R* guess uses a period guess
            k = np.sqrt(Depth)
            per1 = float(np.nanmax(t_fit) - np.nanmin(t_fit))
            per2 = float(((3*np.pi / con.G / rho_star)**0.5) * (pTdur/np.pi/(1+k))**1.5)
            Per_guess = max(per1, per2)
            if Per_guess < 10:
                Per_guess = 27.8
            # max_runs = 1

            a_rs_init = float(((con.G * rho_star * (Per_guess ** 2)) / (3.0 * np.pi)) ** (1.0 / 3.0))
            a_rs = pm.TruncatedNormal("a_rs", mu=a_rs_init, sigma=5.0, lower=1.0, initval=a_rs_init)

            per = pm.Deterministic("Per", pt.sqrt((3.0 * np.pi) / (con.G * rho_star)) * a_rs ** 1.5)

        else:  # Periodic
            Per = Per_in
            if fold_this:
                per = pm.Uniform("Per", lower=max(0.25, Per * 0.99), upper=Per * 1.01)
                a_rs = pm.Uniform("a_rs", lower=1.0, upper=300.0)
                # fold_this = True
            else:
                per = pm.TruncatedNormal("Per", mu=Per, sigma=max(0.1, 0.05 * Per),
                                         lower=max(0.25, Per * 0.80), upper=Per * 1.20)
                a_rs_mu = pm.Deterministic("a_rs_mu", (con.G * rho_star * (per ** 2) / (3*np.pi)) ** (1/3))
                a_rs = pm.TruncatedNormal("a_rs", mu=a_rs_mu, sigma=3.0, lower=1.0)

        # geometry and LD from Target
        eps = 1e-12

        rp_rs = pm.TruncatedNormal("rp_rs", mu=pt.sqrt(Depth),
                                   sigma=pt.maximum(0.02, 0.5 * pt.sqrt(Depth)),
                                   lower=0, upper=1)
        # b = pm.TruncatedNormal("b", mu=0, sigma=0.01, lower=0, upper=1)
        b = pm.Uniform("b", 0, 1)
        depth = pm.Deterministic("depth", rp_rs**2)

        cosi = pm.Deterministic("cosi", pt.clip(b / a_rs, -1.0 + eps, 1.0 - eps))
        inc  = pm.Deterministic("inclination", pt.arccos(cosi) * 180.0 / np.pi)
        inc_safe = pt.clip(inc, 0.1, 89.99)
        root = pt.sqrt(pt.clip(1.0 - b**2, eps, 1.0))
        T_dur0 = per / ((a_rs + eps) * np.pi)
        tau = pm.Deterministic("tau", rp_rs * T_dur0 / root)
        dur = pm.Deterministic("dur", root * T_dur0 + tau)

        # masks
        # if type_fn == "Periodic":
        #     intran_mask = transit_mask_tensors(t_fit, per, dur, t0, cad)
        # else:
        #     intran_mask = pt.abs(t_fit - t0) < (dur / 2.0)
        cad_days = cad / (24.0 * 60.0)

        if type_fn == 'Single':


            cad_days = cad / (24.0 * 60.0)
            width = 0.5 * dur + cad_days
            soft_mask = pt.exp(-0.5 * ((t_fit - t0) / width)**2)
        else:
            cad_days = cad / (24.0 * 60.0)
            width = 0.5 * dur + cad_days
            phase_dist = pt.abs(((t_fit - t0 + 0.5 * per) % per) - 0.5 * per)
            soft_mask = pt.exp(-0.5 * (phase_dist / width)**2)

        # outran_mask = pt.invert(intran_mask)
        out_weight = 1 - soft_mask

        out_flux = f_fit * out_weight
        count = pt.maximum(pt.sum(out_weight), 1)
        mean_out = pt.sum(out_flux) / count
        std_out = pt.sqrt(pt.sum(out_weight * (f_fit - mean_out)**2) / count)

        N_tran = pt.sum(soft_mask)
        uq = pt.ones_like(f_fit) * std_out
        sigs = pt.switch(N_tran > 0, pt.mean(pt.where(soft_mask, uq, 0)), 1e6)

        SNR_val = pt.switch(pt.gt(N_tran, 0), pt.sqrt(N_tran) * depth / sigs, 0.01)
        SNR_clipped = pt.clip(SNR_val, 0.01, 1e4)
        SNR_final = pt.where(pt.eq(SNR_clipped, 1e4), 1, SNR_clipped)
        # if not fold_this:
        pm.Deterministic("SNR", SNR_final)

        # norm = pm.Deterministic("norm", median_pytensor(out_flux))
        norm = pm.Normal("norm", mu=1.0, sigma=0.01)
        if fold_this:
            print('folded')
            p_flux_model = batman_op(t_fit, t0, per, rp_rs, a_rs, inc_safe, u1, u2, ecc, cad)
            pm.Normal("obs", mu=p_flux_model * norm, sigma=u_fit,observed=f_fit)
        else:
            flux_model = batman_op(t_fit, t0, per, rp_rs, a_rs, inc_safe, u1, u2, ecc, cad)
            pm.Normal("obs", mu=flux_model * norm, sigma=u_fit, observed=f_fit)

    with model:
        try:
            trace, fit_info = sample_until_converged(model, mp_context="fork", max_attempts=max_runs)
            summary = extract_summary_dataframe(trace)

            del trace

        except Exception as e:
            print(f"pymc_fit_candidate failed: {e}", flush=True)
            return pd.DataFrame(columns=["mean","median","sd","hdi_16%","hdi_84%","r_hat"]), False, None
    # Keep plots off in core. The caller decides.
    return summary, True, fit_info