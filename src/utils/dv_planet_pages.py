# utils/dv_planet_pages.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.handling_data import predict_lc, _finite
from stages.search_periodic import transit_mask
from utils.ephemerides import phase_fold
from core.planet_candidate import PlanetCandidate
from utils.handling_data import bin_by_time_many_args
from core.target import Target
from collections import OrderedDict

pipeline_sort = {"TGLC":0, "SPOC_2min":1, "SPOC_30min":2, "QLP":3, "ELEANOR":4} 




def choose_epochs(time, cand, max_epochs=3):
    """
    Returns up to max_epochs epoch centers (days), preferring cand.transit_times_days.
    """
    tmin, tmax = float(np.nanmin(time)), float(np.nanmax(time))

    tt = getattr(cand, "transit_times_days", None)
    if tt:
        arr = np.array(sorted({float(x) for x in tt}), dtype=float)
        arr = arr[(arr >= tmin) & (arr <= tmax)]
        if len(arr) > 0:
            if len(arr) <= max_epochs:
                return [float(x) for x in arr]
            return [float(arr[0]), float(arr[len(arr)//2]), float(arr[-1])]

    if getattr(cand, "ptype", None) == "Periodic" and cand.period_days:
        P = float(cand.period_days); t0 = float(cand.t0_days)
        if np.isfinite(P) and P > 0:
            k0 = int(np.floor((tmin - t0) / P)) - 1
            k1 = int(np.ceil((tmax - t0) / P)) + 1
            epochs = t0 + P * np.arange(k0, k1 + 1)
            epochs = epochs[(epochs >= tmin) & (epochs <= tmax)]
            if len(epochs) == 0:
                return [t0]
            if len(epochs) <= max_epochs:
                return [float(x) for x in epochs]
            return [float(epochs[0]), float(epochs[len(epochs)//2]), float(epochs[-1])]

    return [float(cand.t0_days)]



def creating_phase_folded_figure(time, flux, err, cand: PlanetCandidate, gs, pipeline="TGLC"):
    ppl_num = pipeline_sort[pipeline]
    ax = plt.subplot(gs[0, ppl_num])

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if err is None:
        err = np.full(len(flux), np.nanstd(flux), dtype=float)
    else:
        err = np.asarray(err, dtype=float)
        if len(err) != len(flux):
            err = np.full(len(flux), np.nanstd(flux), dtype=float)

    # Guard: only makes sense for periodic candidates with finite positive P
    P = getattr(cand, "period_days", None)
    if P is None or (not np.isfinite(float(P))) or float(P) <= 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "no valid period → no phase fold", ha="center", va="center", fontsize=10)
        return ax
    P = float(P)

    folded_time = phase_fold(time, float(cand.t0_days), P)   # phase in [-0.5, 0.5) ideally

    fold_indx = (np.abs(folded_time) * P) < 0.75   # within ±0.75 days of transit center

    # Bin in "time-from-transit" (days) using x = phase * P, then convert back to phase for plotting
    bin_fold, bin_dict = bin_by_time_many_args(
        folded_time[fold_indx] * P,         # days from transit center
        10,                                 # bin size (your function’s units; assuming minutes or something consistent with your usage)
        flux=flux[fold_indx],
        err=err[fold_indx]
    )
    bin_flux, bin_err = bin_dict["flux"], bin_dict["err"]

    # errorbar doesn't accept s= ; use fmt/ms
    ax.errorbar(
        folded_time[fold_indx], flux[fold_indx], yerr=err[fold_indx],
        fmt=".", ms=2, color="lightgrey", ecolor="lightgrey", lw=0.5, zorder=0
    )
    ax.errorbar(
        bin_fold / P, bin_flux, yerr=bin_err,
        fmt="o", ms=3, color="k", ecolor="k", lw=0.8, zorder=1
    )

    ax.tick_params(labelsize=7)
    ax.set_ylabel("Relative Flux", fontsize=9)
    ax.set_xlabel("Phase", fontsize=9)
    # ax.set_xlim(-0.5, 0.5)

    return ax



        

def creating_2_Nplanets_pages(target: Target, cand: PlanetCandidate, planet_df):

    ticid = target.ticid
    ppl = 'TGLC'

    total_csv = find_total_csv(target.root_dir, target.data_source.value)
    df = pd.read_csv(total_csv).dropna(subset=["FLUX"])

    time, flux, err = [np.array(df[col]) for col in ['TIME', 'FLUX', 'FLUX_ERR']]
    
    catalog_df = build_catalog_df_for_target(target)

    try:
        u1 = float(catalog_df.aLSM)
        u2 = float(catalog_df.bLSM)
    except Exception:
        u1, u2 = np.nan, np.nan
    
    print('checking lengths', [len(x) for x in [time, flux, err]])
    
    if len(err) != len(flux):
        err = np.full(len(flux), np.std(flux))
        print('Error')
    
    ymin = np.nanmin([np.percentile(flux, 0.25)])*0.95 
    #,1.-(max(planet_df['Depth']))]) #define y-axis limits by percentages to avoid using es 
    ymax = np.percentile(flux,99.5)*1.05
    delta_y = np.abs(ymax-ymin)
    ymin = ymin-(delta_y*.05) #make sure ymin allows for all data


    fig0 = plt.figure(figsize=(8.5, 11),constrained_layout=True,dpi=100)
    gs = fig0.add_gridspec(1,2,width_ratios=[4.25, 1], wspace = 0.1) #create grid for subplots - makes it easier to assign where each plot goes
    
    gs0 = gs[0].subgridspec(7, 5, wspace=0.02)
    gs1 = gs[1].subgridspec(1, 1)   


    ymin = np.nanmin([np.percentile(flux, 0.25)])*0.95 
    #,1.-(max(planet_df['Depth']))]) #define y-axis limits by percentages to avoid using es 
    ymax = np.percentile(flux,99.5)*1.05
    delta_y = np.abs(ymax-ymin)
    ymin = ymin-(delta_y*.05) #make sure ymin allows for all data
    subplot_row = 0
    
    ax1 = creating_phase_folded_figure(
        time, flux, err, cand, gs0, ppl)
    ax1.set_ylim(ymin, ymax=)

    subplot+=1
    
    #put the other pipeline stuff here
                
    ax_fin = plt.subplot(gs1[:,-1]) #for the last subplot, print text
    txtstr = '----- Planet Parmas -----'                              +'\n'\
    
    txtstr = txtstr + '--' +'Planet Num='+ str(int(planet.Planet_Num))+'--' +'\n'\
    +'Planet Type='+str(planet.Ptype) +'\n'\
    +'R_p='  + '{:3.5}'.format(str(planet.Rad_p*float(catalog_df.Rad)*109.122))    +'[R_e]'   +'\n'\
    +'t0='   + '{:4.9}'.format(str(planet.T0))       +'[TJD]'   +'\n'\
    +'depth='+ '{:1.6}'.format(str(planet.Depth))               +'\n'\
    +'T='    + '{:2.5}'.format(str(planet.Dur))      +'[h]'     +'\n'\
    +'P_c='  + '{:5.6}'.format(str(planet.Period))      +'[d]'     +'\n'


    # if len(planet_df)==0:
    #     txtstr = txtstr + '\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'
#         plt.axis([0,1,0,1])
    ax_fin.text(0.05, 0.98, txtstr, transform=ax_fin.transAxes, 
    verticalalignment='top', horizontalalignment='left', fontsize = 10)
#         plt.text(0., 0., txtstr,fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')


