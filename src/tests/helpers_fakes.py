
# tests/helpers_fakes.py
import types
import numpy as np

def make_fake_functions_all():
    fake = types.ModuleType("Functions_all")

    def fake_breaking_up_data(time, break_val):
        # One split index covering the full LC
        return [np.arange(len(time))]

    def fake_using_BLS_recursive(
        time, flux, flux_err=None, intransit=None, verbose=True, plot=True,
        max_planets=10, min_SDE=10, min_delta_BIC=75, use_AIC=False,
        periods=None, T0=None, Tdur=None, depths=None, first=False,
    ):
        # Return one synthetic candidate consistent with our lc_small fixture
        periods = np.array([2.0])
        T0s     = np.array([1.0])
        Tdur    = np.array([0.08])
        depths  = np.array([0.005])
        intransit = np.zeros(len(time), dtype=bool)
        return periods, T0s, Tdur, depths, intransit

    def fake_fitting_periodic_planets(
        time, flux, flux_err, pers, t0s, depths, ab, intransit,
        verbose=True, save_phaseFold=False, total_time=True, data_file='.', chain_diff=0,
    ):
        # Return the canonical tuple shape without heavy sampling
        T0_vals = t0s
        periods_fit = pers
        depth_fit   = depths
        Tdur_fit    = np.array([0.08] * len(t0s))
        SNR_vals    = np.array([10.0] * len(t0s))
        params_df   = None
        return T0_vals, periods_fit, depth_fit, Tdur_fit, SNR_vals, intransit, params_df

    fake.breaking_up_data = fake_breaking_up_data
    fake.using_BLS_recursive = fake_using_BLS_recursive
    fake.fitting_periodic_planets = fake_fitting_periodic_planets
    return fake
