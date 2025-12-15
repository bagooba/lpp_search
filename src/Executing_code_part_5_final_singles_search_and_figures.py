#!/usr/bin/env python
# coding: utf-8

# In[1]:


import batman
import emcee
import glob
import os
import shutil
import math
import corner
import numba
import itertools
import gc

import numpy       as np
import pandas      as pd
import time        as tm 
import lightkurve  as lk
# import mr_forecast as mr

import matplotlib                      as mpl
import matplotlib.pyplot               as plt
import matplotlib.gridspec             as gridspec
from   matplotlib.backends.backend_pdf import PdfPages
import mpl_axes_aligner


import astropy.io.fits    as apf
import astropy.units      as units
from   astropy.stats      import sigma_clip
from   astropy.wcs        import WCS
from   astropy.timeseries import BoxLeastSquares
from   astroquery.mast    import Catalogs
from   astroquery         import svo_fps

from multiprocessing import Pool, Process
from wotan           import flatten
from functools       import partial
from ldtk            import LDPSetCreator, BoxcarFilter, TabulatedFilter, SVOFilter
from ldtk.filters    import tess, sdss_z
from IPython.display import display, HTML
from tqdm.auto       import tqdm


# import eleanor

import warnings
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))

from Functions_all import *



# ###LET'S DO THIS!!!!


def main(target,save_things = True):

    num = 0
    num+=1
    ticid  = int(target.split('/')[-1].split('_')[1].split('-')[1])
    print(ticid)
    gaiaID = str(target.split('/')[-1].split('_')[2].split('-')[-1])
    print('running ticid', ticid)

#     if os.path.exists(target+'/tic_star_parameters.csv'):
#         print('getting star params')

#     print('target', target)
    if os.path.exists(target+'/tic_star_parameters.csv'):
        print('getting star params')
        catalog_df = pd.read_csv(target+'/tic_star_parameters.csv')

    else:
        all_star_data = pd.read_csv('./final_mdwarf_params.csv', header = 0)#
        print('IDs', ticid, gaiaID)
        catalog_df = get_catalog_info(ticid, df = all_star_data, rtrn_df = True, gaia_id = gaiaID)
        if len(catalog_df)>0:
#         print('catalog info', catalog_df)
            catalog_df.to_csv(target+'/tic_star_parameters.csv', index = False)
        else: 
            catalog_df = []

    # catalog_df = pd.read_csv(target+'/tic_star_parameters.csv')
    print('catalog df', catalog_df)
    total_file_path = glob.glob(target+'/*TGLC*_total.csv')[0]
    intransit = []
    per_planet_df = []
    intransit, per_planet_df, pparams_df =  executing_total_periodic_search(data_file = total_file_path, ticid = ticid, catalog_df = catalog_df)
    
    
    print('intransit 2', intransit, len(np.where(intransit)[0]), len(intransit))
    if len(per_planet_df)>0:
        per_planet_df['Ptype'] = 'Period'
#         print(per_planet_df)        
    
#     print('data-file for single', target
#         for per in per_planet_df['period']:
#             intransit = np.full(len(time), False)
 
    singles_planet_df, sparams_df = singles_search(ticid, total_file_path, intransit = intransit, catalog_df = catalog_df, confidence = 0.6, run_1 = False, data_file = target)   
        
    singles_planet_df['Ptype'] = 'Single'
    
    singles_planet_df["planet_name"] = singles_planet_df["planet_name"] + max(list(per_planet_df['planet_name'].astype(int))+[0])
    singles_planet_df["planet_name"] = singles_planet_df["planet_name"].astype(int)
    all_planets_df = pd.concat([per_planet_df, singles_planet_df]).reset_index(drop=True)
    
    print('singles params df', sparams_df)
    print('periodic params df', pparams_df)
    
    print(all_planets_df)
    if len(all_planets_df)>0:
    
        total_time_flux_df = pd.read_csv(total_file_path)

        time     = np.array(total_time_flux_df['TIME'])
        flux     = np.array(total_time_flux_df['FLUX'])
        flux_err = np.array(total_time_flux_df['FLUX_ERR'])
    
        
        column_names = ['TICID','Planet_Num','Ptype','T0','Period','Depth','Dur','Rad_p','Cosi','Semi_Maj','b','u1','u2','Norm','Win','Tau','SNR']
        pnew_params_df = []
        snew_params_df = []
#         print('time', time)
        new_planet_df = pd.DataFrame(columns=column_names)  
        column_vals = []
        new_params = np.array([np.nan])
        for indx, planet in all_planets_df.iterrows():
            init_params = [ticid, planet['planet_name'], planet['Ptype']]
            if planet['Ptype'] == 'Single':
                
                print('single planet fit')
                
                snew_params_df = pymc_new_general_function(time, flux, flux_err, planet['T0'], [planet['Tdur'], catalog_df[['aLSM', 'bLSM']].values[0].astype(float), planet['depth']], 'Single', target)
            if len(snew_params_df):

                ab = catalog_df[['aLSM', 'bLSM']].values[0].astype(float)
                print('AB', ab)
                u1, u2, = ab

                T0, per, depth, tdur, rp_rs, cosi, a, b, Norm, Win, Tau, SNR = snew_params_df.loc['t0', 'mean'], snew_params_df.loc['Per', 'mean'],   snew_params_df.loc['depth', 'mean'],          snew_params_df.loc['dur', 'mean'], snew_params_df.loc['rp_rs', 'mean'], snew_params_df.loc['cosi', 'mean'],           snew_params_df.loc['a_rs', 'mean'], snew_params_df.loc['b', 'mean'],     snew_params_df.loc['norm', 'mean'],           snew_params_df.loc['win', 'mean'],   snew_params_df.loc['tau', 'mean'],   snew_params_df.loc['SNR', 'mean']
                
                
                new_params = np.array([T0, per, depth, tdur, rp_rs, cosi, a, b, u1, u2, Norm, Win, Tau, SNR])

            elif planet['Ptype'] == 'Period': 
#                 print('periodic planet fit', planet)
                

                print('pparams_df', pparams_df)
        
                if type(pparams_df.loc['Per', 'mean']<25) == np.bool_:
                    keep = pparams_df.loc['Per', 'mean']<25
                else:
                    keep = list(pparams_df.loc['Per', 'mean']<25)[-1]

                if (len(pparams_df)>0) and keep:
                    pnew_params = pparams_df
                else:

                    pnew_params_df = pymc_new_general_function(time, flux, flux_err, planet['T0'], [planet['period'], catalog_df[['aLSM', 'bLSM']].values[0].astype(float), planet['depth']], 'Periodic')
                    
                if len(pnew_params_df):
                    
                    T0, period_, depth, tdur, rp_rs, cosi, a, b, u1, u2, Norm, Win, Tau, SNR = pnew_params_df.loc['t0', 'mean'], pnew_params_df.loc['Per', 'mean'],   pnew_params_df.loc['depth', 'mean'],                  pnew_params_df.loc['dur', 'mean'], pnew_params_df.loc['rp_rs', 'mean'], pnew_params_df.loc['cosi', 'mean'],                   pnew_params_df.loc['a_rs', 'mean'], pnew_params_df.loc['b', 'mean'],     catalog_df[['aLSM', 'bLSM']].values[0].astype(float), pnew_params_df.loc['norm', 'mean'], pnew_params_df.loc['win', 'mean'],   pnew_params_df.loc['tau', 'mean'],                    pnew_params_df.loc['SNR', 'mean']
                    
                    new_params = np.array([T0, per, depth, tdur, rp_rs, cosi, a, b, u1, u2, Norm, Win, Tau, SNR])
            else:
                print('something is wrong')
                print(planet['Ptype'])
#             print('init params', init_params)
            if not np.isnan(new_params).any():
                column_vals = list(init_params)+list(new_params)
                print('list column values', column_vals)
                print('checking new_planet_df too ', new_planet_df, )

        if save_things:
            all_planets_filename = './saving_all_planets.csv'
            new_planet_df.loc[len(new_planet_df.index)] = column_vals
            new_planet_df.to_csv(target+'/tic-'+str(ticid)+'_planets.csv', index = False, mode = 'a')
    # os.path.exists():
            if os.path.exists(all_planets_filename):

                new_planet_df.to_csv(all_planets_filename, index = False, mode = 'a', header = False)
            else: 
                new_planet_df.to_csv(all_planets_filename, index = False, mode = 'a')
    #         new_planet_df = new_planet_df[new_planet_df['Q'] != np.inf]
    #         new_planet_df = new_planet_df[(new_planet_df['Q']>9. )].reset_index(drop=True)
    #         for key in new_planet_df.columns:
    #             print(key, new_planet_df[key])
            if len(new_planet_df)>0:
                creating_first_DV_report_page(ticid, total_file_path, new_planet_df, catalog_df, intransit)
    else: return

if __name__ == "__main__":
    import multiprocessing as mpl
    # from schwimmbad import JoblibPool

    try:
        file_num = int(sys.argv[1])
    except ValueError:
        
        sys.exit(1)
#     # file_num +=1000

    time1 = tm.time()
    target_files = sorted(glob.glob('./toi_data/*check'))
# 
    # pool = mpl.Pool()


    # factor_files_max = min(len(target_files), (file_factor+1)*8)
    # factor_files_min = file_factor*8
    # factor_files = target_files[factor_files_min:factor_files_max]
    # for result in pool.imap(main, [file for file in files]):
     #     nfilename = './All_indiv_stars_new/cad_'+str(round(iter_num, 2))+'/yield_vals_'+str(file_num).zfill(4)+'.csv'

    #     result.to_csv(nfilename, index = False, mode='a')
#     with Pool(30) as pool:
#         pool.map(main, target_files)

#     lst_times = []
#     for file_num in range(len(target_files)): 
#         time_start = tm.time()
    main(target_files[file_num])
#         time_end = tm.time()
#         time_spent = time_start - time_end
#         print('time it took: ', time_spent/60, ' minutes')
#         lst_times.append(time_spent)
#         pd.DataFrame({'times': time_spent}).to_csv('time_running_takes.csv', index = False, mode = 'a')
    gc.collect()


