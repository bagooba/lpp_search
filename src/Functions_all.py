#!/usr/bin/env python
# coding: utf-8

# In[29]:


import batman
# import emcee
import glob
import os
import shutil
import math
import corner
import numba
import itertools
import sys
# import gerbls

import numpy           as np
import pandas          as pd
import time            as tm 
import lightkurve      as lk
import deep_transit    as dt
import multiprocessing as mp
# import mr_forecast as mr
import scipy.stats as sst

import matplotlib                      as mpl
import matplotlib.pyplot               as plt
import matplotlib.gridspec             as gridspec
import matplotlib.ticker               as ticker

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
import gc
# from pympler.tracker import SummaryTracker

import pymc as pm
import pytensor as PyT
import pytensor.tensor as pt
import arviz as az
from pytensor.graph import Op, Apply

from pytensor import config as pt_config


# import eleanor

import warnings
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))

import config as con
# In[32]:


def mkdir_if_doesnt_exist(outdir, str_new_dir_name):
    # Written by Mallory Harris
    # Description: creates new directory to save data to if it does not already exist
    # Arguments : outdir             = existing directory in which new directory will be located
    #             str_new_dir_name   = string of the new subdirectory of outdir's name

    if os.path.exists(outdir+str_new_dir_name)==False:
        new_outdir = os.path.join(outdir, str_new_dir_name)
        os.mkdir(new_outdir)


# In[37]:
def mk_target_dir_mv_fits_file(fits_file_with_GAIAid, sector_df):
    gaia_ID = fits_file_with_GAIAid.split('-')[2]
    ticid = int(sector_df[sector_df['GAIA_ID'].astype(str)==gaia_ID]['TICID'])
    mkdir_if_doesnt_exist('../Search_target_data/', 'target_tic-'+str(ticid)+'_gaiaID-'+str(gaia_ID))
    os.rename(fits_file_with_GAIAid, '../Search_target_data/target_tic-'+str(ticid)+'_gaiaID-'+str(gaia_ID)+'/'+fits_file_with_GAIAid.split('/')[-1])
    


LDC_for_quadratic = pd.read_csv('../data/LDC_params/table15.dat', 
                                header = None, 
                                sep="\s+", index_col=None,
                               names = ['logg', 'Teff', 'z','L/HP', 'aLSM', 'bLSM',
                                       'aFCM', 'bFCM', 'SQRT(CHI2)', 'qsr', 'PC'])

LDC_PARAMS_MDWARF = LDC_for_quadratic[LDC_for_quadratic['Teff']<4300]
                  

def match_logg_and_teff_for_LDC(df):
    
    # Written by Mallory Harris
    # Description: uses logg and effective temp to calculate quadratic limb darkening parameters based on given  csv
    # Arguments : df = panda dataframe of TIC parameters, specifically Teff and logg
    # Return    : df = panda dataframe with updated quadratic limb darkening parameters

    a = []
    b = []
    bar_format = "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} targets | {elapsed}<{remaining}"
    pbar = tqdm(total=len(df), smoothing=0.3,  position=1, leave=True, bar_format=bar_format)
    for i in range(len(df)):

        Teff = np.float128(df['Teff'])[i]
        logg =  np.float128(df['logg'])[i]
        pbar.update(1)

        try:
            int(Teff)
            mdwarf_Teff =LDC_PARAMS_MDWARF[LDC_PARAMS_MDWARF['Teff'] == np.median(LDC_PARAMS_MDWARF.iloc[(LDC_PARAMS_MDWARF['Teff'].astype('float128')-Teff+0.01).abs().argsort()[:8]].reset_index(drop=True)['Teff'])]
#             if i%100 == 0:
#                 if len(set(mdwarf_Teff['Teff']))>1:
#                        print(mdwarf_Teff)
            if not abs(logg)>=0.:
                logg = np.median(mdwarf_Teff['logg'])
            aLSM, bLSM =  mdwarf_Teff.iloc[(mdwarf_Teff['logg'].astype('float128')-logg-1E-8).abs().argsort()].iloc[0][['aLSM', 'bLSM']]
            a.append(aLSM)
            b.append(bLSM) 
        except:
            a.append(np.nan)
            b.append(np.nan)
    pbar.close()

    df['aLSM'] = a
    df['bLSM'] = b
    return df

    
def get_catalog_info(ticid, df = False, rtrn_df = False, gaia_id = False):
    try: 
        new_df = df[df['TICID'].astype(int)==int(ticid)]
#         print('checking df, ', new_df)

    except Exception as err:
        print('this is error: ', err)
        
        ctlfile = '../data/final_mdwarf_params.csv'
        mdwarfs = pd.read_csv(ctlfile, iterator =True, chunksize = 100000, index_col=None, header = 0)
        new_df = pd.concat(
            [chunk[chunk['TICID'].astype(int) == int(ticid)] 
            for chunk in mdwarfs]).reset_index(drop=True)

    new_df = match_logg_and_teff_for_LDC(new_df)

#     if len(new_df) == 0:
# #         print('we have a problem')
        
#         if (type(gaia_id)!=bool) and (type(df)!=bool):
#             new_df = df[df['GAIA_ID'].astype(str) == gaia_id]

    if len(new_df) == 0:
#                 print('we have a serious problem')
        ctlfile = '../data/final_mdwarf_params.csv'
        mdwarfs = pd.read_csv(ctlfile, iterator =True, chunksize = 100000, index_col=None, header = 0)
        new_df = pd.concat(
            [chunk[chunk['TICID'].astype(int) == ticid] 
            for chunk in mdwarfs]).reset_index(drop=True)
    if len(new_df) == 0:
        print('we have a problem')
        
        
    if rtrn_df:
        return new_df
    else: 
        return new_df[['aLSM', 'bLSM']].values[0].astype(float), float(new_df['Mass']), float(new_df['eMass']), float(new_df['eMass']), float(new_df['Rad']), float(new_df['eRad']), float(new_df['eRad'])





# In[39]:


column_names = ['TIME', 'FLUX','FLUX_ERR', 'BKG_FLUX', 'BKG_FLUX_ERR', 'QUALITY', 'CENTROID_X', 
                'CENTROID_X_ERR','CENTROID_Y', 'CENTROID_Y_ERR']

def extract_data_from_fits_files(fitsFile, PL = "", sector = 0):
    hdulist=apf.open(fitsFile) #fits time series
    indxs = [i for i  in range(len(hdulist)) if ('Table' in str(hdulist[i]))] #grab the tabular information, which is the data
    tbdata = hdulist[indxs[0]]
    data = tbdata.data
    all_col_names = [jjj.name.upper() for jjj in tbdata.columns]

    params_df = pd.DataFrame(np.nan, range(len(data)), columns = ['TIME'])
    params_df['TIME'] = data['TIME']

    if len(PL)>0:
        PL = PL.upper()+'_'

    flux_cols     = np.array(sorted([name for name in all_col_names if ('FLUX' in name) & ('X_' not in name) & ('K' != name[0]) & ('CAL' != name[0:3])], key=len))
    bkg_cols      = np.array(sorted([name for name in all_col_names if ('BKG' in name) | ('BACKGROUND' in name)], key = len))
    centroid_cols = np.array([name for name in all_col_names if ('CENTR' in name) & ('MOM' not in name)])
    qual_cols     = np.array([name for name in all_col_names if ('QUAL' in name) | ('FLAG' in name)])

    useful_cols = []
    if len(flux_cols) == 0:
        return
    elif len(flux_cols) >1:
#         print(flux_cols)
        for col in flux_cols:
            new_col = col.split('_')[0][:4]
            params_df[new_col+'_FLUX'] = data[col]
            useful_cols.append((col, new_col+'_FLUX'))

    else:
        flux_col = flux_cols[-1]
#         print(flux_col)
        params_df['FLUX'] = data[flux_col]
        useful_cols.append((flux_col, 'FLUX'))

    if len(bkg_cols)>0:
        params_df['BKG_FLUX']  = data[bkg_cols[0]]
        useful_cols.append((bkg_cols[0], 'BKG_FLUX'))

    if len(centroid_cols)>0:
        x_centr = np.array(sorted([name for name in centroid_cols if ('X' in name) | ('1' in name)], key = len))
        y_centr = np.array(sorted([name for name in centroid_cols if ('Y' in name) | ('2' in name)], key = len))
        params_df['CENTROID_X'] = data[x_centr[0]]
        params_df['CENTROID_Y'] = data[y_centr[0]]
        useful_cols.extend([(x_centr[0], 'CENTROID_X'),( y_centr[0], 'CENTROID_Y')])
        
    if len(qual_cols)>0:
        params_df['QUALITY'] = np.sum([np.array(data[qcol_]) for qcol_ in qual_cols], axis=0) 

    for col_ in useful_cols:
        if col_[0]+'_ERR' in all_col_names:
            params_df[col_[1]+'_ERR'] = data[col_[0]+'_ERR']
 
    new_filename = os.path.dirname(fitsFile)+'/'+PL+fitsFile.split('/')[-2]+'_sector'+str(sector).zfill(2)+'.csv'    
    df = params_df.astype(object)

    # print('data frame', df)
    
    df.to_csv(new_filename, index = False)
    


# In[41]:


####DAX FELIZ WINDOW LENGTH FOR WOTAN CODE


def calculate_semi_major_axis(Period, M_star ,R_star): #will replace SMA_AU_from_Period_to_stellar above
    """
    Calculate the semi-major axis using Kepler's third law.

    Parameters:
    ----------
    period : astropy.Quantity
        Orbital period of the planet (in days).
    M_star : astropy.Quantity, optional
        Stellar mass (default is solar mass).

    Returns:
    -------
    semi_major_axis : astropy.Quantity
        Semi-major axis in units of AU.
    """
    Period = Period.to(units.second)  # Convert period to seconds
    M_star = M_star.to(units.kg)  # Convert stellar mass to kg
    R_star = R_star.to(units.m)  # Convert stellar mass to kg

    from astropy import constants as const
    G = const.G

    a_cubed = (G * M_star * Period**2 / (4 * np.pi**2)).to(units.m**3)
    semi_major_axis = a_cubed**(1/3)

    scaled_SMA = (semi_major_axis / R_star).decompose() #now unitless
    SMA_cm = semi_major_axis.to(units.cm)
    return scaled_SMA, SMA_cm.value



def T14(P, R_star, M_star, R_planet, b=0, i=90*units.deg): #will replace Tdur above
    #instead of using R_planet, i'm going to try the minimum planet radius 
    #where I could potentially get a minimum SNR needed
    
    #I'm also going to have to think about the periods that I am going to try for

    """
    Estimate the total transit duration (T14) for a planet using Kepler's third law and transit geometry.

    Parameters:
    ----------
    P : float
        Orbital period of the planet (in days).
    R_star : float
        Radius of the star (in solar radii).
    M_star : float
        Mass of the star (in solar masses).
    R_planet : float
        Radius of the planet (in Earth radii).
    b : float, optional
        Impact parameter (default is 0 for central transit).
    i : astropy.Quantity, optional
        Orbital inclination (default is 90 degrees for edge-on orbit).

    Returns:
    -------
    transit_duration : astropy.Quantity
        The total transit duration (T14) in days.
    """
    import astropy.units as u
    # add units to inputs
    P, R_star, M_star, R_planet = P*units.day, R_star*units.R_sun, M_star*units.M_sun, R_planet*units.R_earth

    # Convert inclination to radians
    i = i.to(units.radian).value

    # Semi-major axis in AU
    scaled_SMA, SMA_cm = calculate_semi_major_axis(Period=P,M_star=M_star,R_star=R_star)
    a = (SMA_cm*units.cm).to(units.AU)

    # Convert units to meters
    R_star = R_star.to(units.m)
    R_planet = R_planet.to(units.m)

    k = (R_planet / R_star).decompose().value  # Planet-to-star radius ratio

    # Calculate the geometric part of the transit duration (dimensionless)
    piece_A = (P.to(units.second) / np.pi).decompose()  # This will give us a time quantity
    piece_B1 = (R_star / a.to(units.m)).decompose().value  # Dimensionless
    piece_B2 = np.sqrt((1 + k)**2 - b**2)  # Dimensionless
    piece_B3 = 1 / np.sin(i)  # Dimensionless

    # Combine the parts for the full duration calculation
#     arcsin_argument = np.clip(piece_B1 * piece_B2 * piece_B3, -1, 1)  # Ensure arcsin argument is valid
    arcsin_argument = piece_B1 * piece_B2 * piece_B3
    angle_radians = np.arcsin(arcsin_argument)  # Result in radians

    # Multiply by the time factor to get the total transit duration
    T14_seconds = (piece_A * angle_radians)  # Now this is in seconds

    return T14_seconds.to(units.day).value

# window_size_in_days = 5* T14(P=maxP, R_star=R_star, M_star=M_star, R_planet=2*R_planet_RE)


# In[42]:


def remove_outliers(time, flux, sigma_lower=6.0, sigma_upper=3., **kwargs):
    outlier_mask = sigma_clip(data=flux,
#                               sigma=sigma,
                              sigma_lower=sigma_lower,
                              sigma_upper=sigma_upper,
                              **kwargs).mask
    # Second, we return the masked light curve and optionally the mask itself
    return outlier_mask

def flatten_lc(time, flux, catalog_df = pd.DataFrame({'Rad':[-1],  'Mass': [-1]}), maxP = 100, R_planet_RE=2):

    # print('catalog df', catalog_df)
    if len(catalog_df)>0:
        M_star = float(catalog_df['Mass'])
        R_star = float(catalog_df['Rad'])
    if not M_star>0:
        M_star = 0.5
    if not R_star>0:
        R_star = 0.5
        
    windw  =3* T14(P=maxP, R_star=R_star, M_star=M_star, R_planet=2*R_planet_RE)
    
#     print('params breaking', maxP, R_star, M_star, R_planet_RE)
    flat_flux, flux_trend = flatten(
    time,                 # Array of time values
    flux,                 # Array of flux values
    method='biweight',
    window_length=windw, #R_planet_RE = in Earth RE, so = 1 # The length of the filter window in units of ``time``
#     edge_cutoff=0.1,      # length (in units of time) to be cut off each edge.
#     break_tolerance=1,  # Split into segments at breaks longer than that
    return_trend=True,    # Return trend and flattened light curve
#     cval=5.0              # Tuning parameter for the robust estimators
    )
#     print('window size currently used?', windw*24, ' Tdur: ', windw/3*24, ' hours')
    return flat_flux, flux_trend


def get_data(ticid_directory, flux_type='APER_', PL = 'TGLC', verbose = False, catalog_df = False, check_PSF = False):
    
    total_time = []
    total_flux = []
    total_flux_err = []
    
    total_flat_flux = []
    total_flat_flux_err = []
    total_flux_trend = []

    trend = []

    flux_col = flux_type+'FLUX'

    files = glob.glob(ticid_directory+'/*_sector*.csv')
#     print('is there a problem with the files?', files)
    for i in files:
        sec = float(i.split('sector')[-1][:2])
        timeseries_df = pd.read_csv(i, index_col = None)
#         print('timeseries_df', timeseries_df)
        if flux_col not in timeseries_df.columns:
            flux_col = 'FLUX'

        timeseries_df = timeseries_df[timeseries_df['QUALITY']==0].reset_index(drop=True)
        timeseries_df_new = timeseries_df[~np.isnan(timeseries_df[flux_col])]
        
        if flux_col+'_ERR' in timeseries_df_new.columns:
            timeseries_df_new[flux_col+'_ERR'] = timeseries_df_new[flux_col+'_ERR']/np.nanmedian(timeseries_df_new[flux_col])
            flux_err = np.array(timeseries_df_new[flux_col+'_ERR'])

        else:
            flux_err = np.full(len(timeseries_df_new), np.std(timeseries_df_new[flux_col]/np.nanmedian(timeseries_df_new[flux_col])))


        timeseries_df_new[flux_col] = timeseries_df_new[flux_col]/np.nanmedian(timeseries_df_new[flux_col])

        time     = np.array(timeseries_df_new.TIME)
        flux     = np.array(timeseries_df_new[flux_col])
        
        
        if len(time) == 0:
            print('check if flux is all null: ', set(timeseries_df[flux_col]))
            print('sector', sec, 'is no good, may want to search PSF')
            continue
        outlier_mask = remove_outliers(time, flux) 
        
        
        clean_time = time[~outlier_mask]
        clean_flux = flux[~outlier_mask]
        clean_err  = flux_err[~outlier_mask]
        
#         print('outlier mask', outlier_mask)
        new_time_series = timeseries_df_new[~outlier_mask]
#         
        #need to work on my flatter_lc function based on Dax's file
        if type(catalog_df) == bool:
        
            flat_flux, flat_trend = flatten_lc(clean_time, clean_flux)
        else:
            flat_flux, flat_trend = flatten_lc(clean_time, clean_flux, catalog_df = catalog_df)
        
        
#         print('catalog df', catalog_df)
#         print('checking flatthings', set(flat_flux), set(flat_trend))
        
        flat_flux_err = np.full(len(flat_flux), np.std(flat_flux))
        
#         print('checking df', new_time_series)
        
        new_time_series[flux_col+'_FLAT'] = flat_flux
        new_time_series[flux_col+'_FLAT_ERR'] = flat_flux_err
        new_time_series[flux_col+'_TREND'] = flat_trend
        new_time_series.to_csv(i, index=False)

        
        if verbose:
            plt.figure(figsize = (20, 6))

            ax = plt.gca()
            ax.set_facecolor('None')
            ax.scatter(time,flux, color = 'brown', zorder = 2, marker = '.')
            ax.plot(clean_time, flat_trend, color = 'k', zorder = 10,)
            
            plt.figure(figsize = (20, 6))
            ax1 = plt.gca()
            ax1.set_facecolor('None')
            ax1.scatter(clean_time,flat_flux, color = 'crimson', zorder = 2, marker = '.')
#             ax1.plot(clean_time, flat_flux, color = 'k', zorder = 10,)

            plt.show()
            
        total_time.extend(clean_time)
        total_flat_flux.extend(flat_flux)
        total_flat_flux_err.extend(flat_flux_err)
        
        total_flux_trend.extend(flat_trend)
        total_flux.extend(clean_flux)
        total_flux_err.extend(clean_err)

#     print('total time length', len(total_time))
    total_time          = np.array(total_time)
    total_flux          = np.array(total_flux)
    total_flux_err      = np.array(total_flux_err)
    total_flat_flux     = np.array(total_flat_flux)
    total_flat_flux_err = np.array(total_flat_flux_err)
    total_flux_trend    = np.array(total_flux_trend)
        
    if len(total_time)>0:
        new_df = pd.DataFrame({'TIME': total_time[np.argsort(total_time)], 'RAW_FLUX': total_flux[np.argsort(total_time)], 
                               'RAW_FLUX_ERR': total_flux_err[np.argsort(total_time)], 'FLUX': total_flat_flux[np.argsort(total_time)],
                              'FLUX_ERR': total_flat_flux_err[np.argsort(total_time)], 'FLUX_TREND': total_flux_trend[np.argsort(total_time)]})
        
        
        #this is getting rid of likely junk segements
#         indexes_split_unorganize = breaking_up_data(total_time, 0.75)   
#         diff_ary = np.array([max(np.array(total_time)[x])-min(np.array(total_time)[x]) for x in indexes_split_unorganize])

#         all_good_indxs =np.concatenate(list(itertools.compress(indexes_split_unorganize, diff_ary>0.5))).ravel()
#         new_df[all_good_indxs].to_csv(ticid_directory+'/'+ticid_directory.split('/')[-1]+'_'+PL+'_'+flux_type+'total.csv', index = False)


        new_df.to_csv(ticid_directory+'/'+ticid_directory.split('/')[-1]+'_'+PL+'_'+flux_type+'total.csv', index = False)
#     elif len(total_time)==0 and flux_type == 'APER_' and check_PSF==True:
#         print('lets try for PSF flux')
#         get_data(ticid_directory, flux_type='PSF_', PL = 'TGLC', verbose = verbose, catalog_df = catalog_df)
#     return np.array(total_time), np.array(total_flux), np.array(total_flux_err)

    


# In[43]:


model_path = '../model_TESS.pth'


def make_LightKurveObject(time, flux, flux_err):
    """ Convert this object to a lightkurve.lightcurve.TessLightCurve object to use for Deep Transit.

    :return: Data in a lightkurve object
    :rtype: `class` lightkurve.lightcurve.TessLightCurve
    """
    lc = lk.TessLightCurve()
    lc.time = time
    lc.flux = flux
    lc.flux_err = flux_err
    return lc


def calc_rudimentary_snr(depth, Tdur, Ntran=1):
    sigma_1hr_15_Tmag = 6283.6147036936645*1E-6

    A = (Ntran**0.5)/sigma_1hr_15_Tmag
    SNR = A*(depth)*((Tdur*24)**0.5)
    return SNR


def plot_lc_with_bboxes(lc_object, bboxes, ax=None, epoch = 0, **kwargs):
    """
    Plot light curve with bounding boxes

    Parameters
    ----------
    lc_object : `~lightkurve.LightCurve` instance

    bboxes : list or np.ndarray
                Bounding boxes in shape (N, 5)

    ax : `~matplotlib.pyplot.axis` instance
                Axis to plot to. If None, create a new one.
    kwargs : dict
                Additional arguments to be passed to `matplotlib.pyplot.plot`

    Returns
    -------
    ax : `~matplotlib.pyplot.axis` instance
                The matplotlib axes object.
    """
    with plt.style.context('grayscale'):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(12, 26), constrained_layout=False)
            ax.plot(lc_object.time.value, lc_object.flux.value, color = 'k', zorder = 1E5, **kwargs)
            ax.set_xlabel('Time - T0 (hours)', color = 'k', fontsize = 40)
            ax.set_ylabel('Normalized Flux', color = 'k', fontsize = 20)

#         else:
#             ax.plot(lc_object.time.value, lc_object.flux.value,color = 'k', **kwargs)
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        recs = []
        val = 0
        for real_mask in bboxes:
            val+=1
#             print(val)
#             print(real_mask[1] - real_mask[3] / 2)

            new_start = real_mask[1] - epoch
#             print('start', real_mask[1], real_mask[4])

            rec = Rectangle(
                (new_start - (real_mask[3])/2, real_mask[2] - (real_mask[4]/2)),
                real_mask[3],
                real_mask[4],
                facecolor='indianred',       # Transparent fill
                edgecolor='indianred',  # Mistyrose border
                linewidth=30,
                zorder=5
            )
            recs.append(rec)
#             SNR = (np.log(1-real_mask[0])/0.15)*-1 #-> SNR calc from DT
            SNR = calc_rudimentary_snr(real_mask[3], real_mask[4])
            ax.text(
                new_start+ abs(real_mask[3]),
                real_mask[2] + 1/2*abs(real_mask[4]),
                s='snr: '+f"{SNR:.2f}",
                color='r',
                verticalalignment="top",
                bbox=dict(alpha=0.75, color='None'),
                clip_on=True, 
                fontsize = 12, zorder = 1E4
            )

        pc = PatchCollection(recs, lw=0.2, zorder=5, match_original = True)
        collection = ax.add_collection(pc)
        collection.set_zorder(5)
    return ax



def DT_analysis(time, flux, flux_err, confidence, DT_Quite=True, is_flat = True):
    """ 
    @author: D. Dragomir, P.Steimle


    Function for the Deep Transit analysis (Cui et al. 2021) that returns only single transit events

    """

    # create a new dataset for the while loop where transits can be masked out
    
    
#     confidence = 1-np.exp(-0.15*snr)
    print('time len', len(time))
    if DT_Quite == True:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = open('.trash.txt', 'w')
        sys.stderr = open('.trash.txt', 'w')

        # do check for transits with DT
        DT_model = dt.DeepTransit(make_LightKurveObject(time, flux, flux_err),  is_flat=is_flat)
        bboxes = DT_model.transit_detection(model_path, confidence_threshold=confidence)

        sys.stdout = save_stdout
        sys.stderr = save_stderr

    else:
        # do check for transits with DT
        DT_model = dt.DeepTransit(make_LightKurveObject(time, flux, flux_err), is_flat=is_flat)
        bboxes = DT_model.transit_detection(model_path, confidence_threshold=confidence)

        # if only 1 or 0 boxes were found, break the loop and continue

    return bboxes
 


# In[44]:


def transit_mask(t, period, duration, T0, buffer=0.2):
    
    # Works with numba, but is not faster
    #all units given in days
#     print('this is evil', 'lim time', min(t), max(t), 'T0', T0, 'duration', duration, 'period', period) 

    mask = np.abs((t - T0 + (0.5 * period)) % period - (0.5 * period)) < duration+buffer
    return mask

def transit_mask_tensors(t, period, duration, T0):
    
    # Works with numba, but is not faster
    #all units given in days
    #this will return a boolean tensor object
    mask = pt.abs(((t - T0 + (0.5 * period)) % period) - (0.5 * period)) < (duration * 2. / 3.)
    return mask

def find_breaks(time, val = 27.):
    time = np.array(time)[np.argsort(time)] 
    t    = np.diff(time)
    inds = np.where( t>val)[0]
    return inds + 1

# def sort_data_by_times(time, flux):
#     return time[np.argsort(time)], flux[np.argsort(time)]

def running_median(data, kernel=25):
    """Returns sliding median of width 'kernel' and same length as data """
    
#     print('kernel', kernel)
    
    idx = np.arange(kernel) + np.arange(len(data) - kernel + 1)[:, None]
    idx = idx.astype(np.int64)  # needed if oversamplinfg_factor is not int
    med = np.median(data[idx], axis=1)

    # Append the first/last value at the beginning/end to match the length of
    # data and returned median
#     print('length of med (if 0, need to return 0)', len(med))
    if len(med)>0:
        first_values = med[0]
        last_values = med[-1]
        missing_values = len(data) - len(med)
        values_front = int(missing_values * 0.5)
        values_end = missing_values - values_front
        med = np.append(np.full(values_front, first_values), med)
        med = np.append(med, np.full(values_end, last_values))
        med[np.isinf(np.abs(med))] = 0

        return med
    else:
        return np.zeros(len(data))

    
    
def breaking_up_data(time, break_val = 27., min_size = 0.5):
    time = np.array(time)
    brk = np.append(np.append([0], find_breaks(time, break_val)), [len(time)])

    indexes = []
    for i in range(len(brk)-1):
        r = np.arange(brk[i],brk[i+1], 1)

        if len(r)>1:
            if np.ptp(time[r])>min_size:
#                 if min_size!= 0.5:
#                     print(np.ptp(time[r]), ' > ', min_size, '?')
                indexes.append(r)
    return indexes



def find_common_element_indices(arrays):
    if not arrays or len(arrays) < 2:
        return [], {}

    
    all_elements = [element for arr in arrays for element in arr]
    elements, indxs, cnts = np.unique(np.round(all_elements, 2), return_index = True, return_counts = True)
    
    common_elements = elements[np.where(cnts>1)]
    print(common_elements)
    
    result = {}
    for ele in common_elements:
        result[ele] = []
        for i, arr in enumerate(arrays):
            if np.isin(ele, np.round(arr, 2)):
                result[ele].append((i, list(np.round(arr,2)[0]).index(ele)))
    return common_elements, result# Example usage:




def check_multiples(arr):
    """
    Checks if an array contains elements that are multiples of each other.

    Args:
      arr: A list of integers.

    Returns:
      indxs tuples of multiple factors
    """
    indxs = []
    n = len(arr)
    for i in range(n):
        for j in range(n):
            if i != j and arr[i] != 0:
                factor = arr[i]/arr[j]
                
                if np.logical_or(np.abs(factor - np.rint(factor))<0.02, np.abs(1/factor - np.rint(1/factor))<0.02):
                    indxs.append((i, j))
    return indxs





def checking_BLS_periodicity(per, period_array, t0, t0_array):
    factors = np.array(period_array)/per
    print('factors',np.round(factors, 5))
    factor_indxs = np.where(np.logical_or(np.abs(factors - np.rint(factors))<0.03, np.abs(1/factors - np.rint(1/np.array(factors)))<0.03))[0]

    pop_per = np.nan

    keep_factor = 1
#     if 1. in np.round(factors, 7):
#         keep_factor = -1E5
#         print('its 1')

#         return per, keep_factor
    rep_indxs = np.where(np.rint(factors[factor_indxs])==1.)[0]     
#     print('indexes of repeated periods', rep_indxs)
    new_period = per
    not1_indxs = np.where(np.rint(factors[factor_indxs])!=1.)
    if len(rep_indxs)>0:
        keep_factor = -1
        val = 0
        while (1. in np.round(period_array/new_period, 1) and val<len(period_array)+1):
            val+=1                    
            new_period = catching_periods_repeated_and_offset(new_period,t0, np.array(period_array), np.array(t0_array), rep_indxs)
            if len(new_period)>0:
                new_period  = min(new_period)
                keep_factor = abs(per/new_period)
                pop_per = per
                
            elif new_period == per:
                keep_factor = -1E5
                

        if val>len(period_array):
            keep_factor = -2
    

    elif len(not1_indxs[0])>0:
        
        factors_not1 = factors[not1_indxs]
#         factors_not1[factors_not1>1] = 1.

        factors_not1 = np.unique(factors_not1)
        
        keep_factor = max(1/factors_not1)
        
        pop_per = float(set(np.array(period_array)[np.where(factors == 1/keep_factor)])[0])
        #note: the following line is assuming that generally the min period is the true period, and multiples are aliases. This allows us to run MCMC on fewer periods. However, if the true period is the longer one, we're in trouble
        new_period  = per
    
    print('final of periodicity check; period: ', new_period, ' keep factor', keep_factor)
    return new_period, keep_factor, pop_per
        

def checking_aliases_repeated_periodic_planets(per_ary, t0_ary, q_ary):
    final_indexs = np.full(len(per_ary), True)
    all_t0 = []
    for iii in range(len(per_ary)):
        all_t0.append([t0_ary[iii] + per_ary[iii]*np.arange(-50, 50)])
    rep_t0, indxs_of_rep_t0 = find_common_element_indices(all_t0)
    for t0 in rep_t0:
        indx_t0 = set(np.array(indxs_of_rep_t0[np.round(t0, 2)])[:,0])
#         print('indexes: ', indx_t0)
        bad_ndx = np.where(np.array(q_ary) == min([q_ary[x] for x in indx_t0]))
        final_indexs[bad_ndx] = False
    print('final_indxs', final_indexs)
    return final_indexs


def checking_multiples_and_duplicate_periodic_planets(per_ary, t0_ary, d_ary, q_ary): 
    final_indexs = np.full(len(per_ary), True)
    div_period_ary = np.ones(len(per_ary))
    
    multiple_indxs = check_multiples(per_ary)
    for ndxs in multiple_indxs:
        i, j = ndxs
        if np.abs(per_ary[i]/per_ary[j] - 1) < 0.03 and np.abs(d_ary[i]/d_ary[j] - 1) > 0.1:
            print('likely a binary')
            final_indexs[[i, j]] = False
            
        else:
            
            if np.abs(per_ary[i]/per_ary[j] - 1) >= 0.03:
                bad_ndx = np.where(np.array(q_ary) == min((q_ary[i], q_ary[j])))
#                 print('this should be int val', bad_ndx, 'and it should be one of these 2 vals', i, j)
                final_indexs[bad_ndx] = False
                
            if  np.abs(per_ary[i]/per_ary[j] - 1) < 0.03:
                t0_diff   = (t0_ary[i] - t0_ary[j])/per_ary[i]
                new_indxs = check_multiples([t0_diff, per_ary[i]])
                
                if len(new_indxs) >0: 
                    bad_ndx = np.where(np.array(q_ary) == min((q_ary[i], q_ary[j])))
                    good_ndx = np.where(np.array(q_ary) == max((q_ary[i], q_ary[j])))

                    final_indexs[bad_ndx] = False
                    div_period_ary[good_ndx] = max([t0_diff/per_ary[i], per_ary[i]/t0_diff])
    
    rep_t0_final_indxs = checking_aliases_repeated_periodic_planets(per_ary, t0_ary, q_ary)
                      
    final_indexs = np.logical_and(rep_t0_final_indxs, final_indexs)
    print('keeping these periods', np.array(per_ary)[final_indexs])
    return final_indexs, div_period_ary




def catching_periods_repeated_and_offset(per, t0, per_array, t0_array, rep_indxs):
    
    new_periods = []
    
    diff_tc = np.abs(np.array(t0_array)[rep_indxs] - t0)
    
    for iii in range(len(diff_tc)):
        n = np.ceil(diff_tc[iii]/per)
        min_diff_tc = np.nanmin(np.abs(diff_tc[iii] - np.array([n-1, n, n+1])*per))
        
        if round(min_diff_tc, 1) !=0:
            frac_per = per/min_diff_tc
            new_periods.append(min_diff_tc)
        
        else: 
            new_periods.append(per)
    
    if len(new_periods)>0:
        nper, indxs = np.unique(np.round(new_periods, 1), return_index=True)
        new_periods = list(np.array(new_periods)[indxs])
    
    return new_periods
                        



def checking_last_BLS_power_for_artificial_inflation(power_results):
    max_indx = 0
    if max(power_results) == power_results[-1]:
    
        max_indx = 1
        rev_power_results = power_results[::-1]
        for pwr in rev_power_results:
            if pwr == power_results[-1]:
                max_indx+=1
            else:
                break
    if max_indx == 0 or max_indx == len(power_results):
        return np.arange(len(power_results))
    else:        
        return np.arange(len(power_results)-max_indx)


        

    
def checking_BLS_odd_even_binaries(stats, t0, period, depth):
    if stats['depth_odd'][0]/stats['depth_even'][0] > 10:
        print('keeping odd - would rather keep binary as 2 objects than miss a planet from period alias')
        t0 = t0+period
        period = 2*period
        depth = stats['depth_odd'][0]

    elif stats['depth_even'][0]/stats['depth_odd'][0] > 10:
        print('keeping even - would rather keep binary as 2 objects than miss a planet from period alias')
        t0 = t0
        period = 2*period
        depth = stats['depth_even'][0]
    return t0, depth, period


def check_rules_to_continue_BLS(results, index):
    sorted_pwr = np.sort(np.unique(results.power_final))
#     stdv_pwr = np.nanstd(np.diff(results.power_final))

    stdv_pwr = np.nanstd(np.sort(results.power_final)[:-1])

#     stdv_pwr = np.nanstd(results.power_final)
    

    period = results.period[index]

    rule_1 = np.abs(np.diff(sorted_pwr[[-1, -2]])) > 2.*stdv_pwr

    pwr_copy = np.array(results.power_final).copy()
    pwr_copy[index] = -np.inf

    period_2 = results.period[np.argmax(pwr_copy)]
    
    factor = np.arange(2, 6)
    check_multipls = sorted(list(1/factor)+list(factor))


    if (not rule_1) and np.isin(np.round(period/period_2, 2), np.round(check_multipls, 2)):
        print('double checking rule 1')
        rule_1 = np.abs(np.diff(sorted_pwr[[-1, -3]])) > 2.*stdv_pwr
        pwr_copy = np.array(results.power_final).copy()
        pwr_copy[np.argmax(pwr_copy)] = -np.inf

        period_3 = results.period[np.argmax(pwr_copy)]
        
        if (not rule_1) and  np.isin(np.round(period/period_3, 2), np.round(check_multipls, 2)):
            print('double checking rule 2')
#             stdv_pwr1 = np.nanstd(np.sort(np.diff(results.power_final))[:-3])
            stdv_pwr1 = np.nanstd(np.sort(results.power_final)[:-3])

#             print('standard deviations', stdv_pwr, stdv_pwr1)
#             print('number stdev', np.abs(np.diff(sorted_pwr[[-1, -4]]))[0]/stdv_pwr1)


            rule_1 = np.abs(np.diff(sorted_pwr[[-1, -4]])) > 2.25*stdv_pwr1
    return rule_1


# def using_TLS_to_find_periodic_signals(time, flux, u, verbose = False, show_progress_info = True, save_phaseFold = True,
#                                        intransit = [], periods = [], T0 = [], Tdur = [], 
#                                        depths = [], first=True):
    
#     time_diff = max(time)-min(time)
#     print('time diff', time_diff)
#     max_per = min(time_diff, 100.)
#     if first == True:
#         intransit = np.full(len(time), False)
#         periods = []
#         T0 = []
#         Tdur = []
#         depths = []

#     time_new = np.array(time)[~intransit]
#     flux_new = np.array(flux)[~intransit]
#     if len(time_new)>0:
#         start = tm.time()

#         durations = np.linspace(0.02, 0.5, 75)
        
#         model = transitleastsquares(time_new, flux_new)
#         results = model.power(
#             period_min=0,
#             period_max=max_per,
# #             transit_depth_min=ppm*10**-6,
# #             oversampling_factor=10,
# #             duration_grid_step=1.02,
#             u=ab,
#             limb_dark='quadratic',
# #             M_star = 1,
# #             M_star_max=1.1
#             n_transits_min = 1,
#             show_progress_info = show_progress_info
#             )

                
#         index = np.argmax(results.power)
        
#         period    = results.period
#         val_triangles = min(results.power)-np.std(results.power)

# #         print('period', period, 'index ', index)
#         end = tm.time()
#         if round(results.T0, 4) in [round(x, 4) for x in T0]:
        
#             intransit = np.logical_or(intransit, transit_mask(time, results.period, results.duration, results.T0))
#             print('FOUND THE SAME PLANET: CONTINUING')
#             return using_TLS_to_find_periodic_signals(time, flux, u, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0, Tdur = Tdur, depths = depths, first = False)
#         if verbose:
            

# #             print('plot 1')
#             plt.figure(figsize = (10,6))

#             ax = plt.gca()
#             ax.set_facecolor('None')
#             ax.scatter(time_new,flux_new, color = 'k', zorder = 10, marker = '.')
            
#             faux_intransit = np.logical_or(intransit, transit_mask(time, results.period, results.duration, results.T0))
            
#             ax.scatter(np.array(time)[~faux_intransit],np.array(flux)[~faux_intransit], 
#                        color = 'r', zorder = 11, marker = '.', alpha = 0.3)

#             plt.ylabel(r'N. Flux')#, fontsize = 40)
#             plt.xlabel('Time', fontsize = 40)

#             plt.figure(figsize = (5, 5))

#             ax = plt.gca()
#             ax.set_facecolor('None')
#             ax.scatter(period,val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

#             plt.xlim(np.min(results.periods), np.max(results.periods))
#             for n in range(2, 10):
#                 ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
#                 ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)


#             plt.ylabel(r'SDE')#, fontsize = 40)
#             plt.xlabel('Period (days)')#, fontsize = 40)
        
        
#             ax.plot(results.periods, results.power, color = 'k', lw=1)
#             ax.xaxis.label.set_color('k')        #setting up X-axis label color to yellow
#             ax.yaxis.label.set_color('k')          #setting up Y-axis label color to blue


#             t0 = results.T0
#             duration = results.duration
            
#             plt.show()
#             plt.close()
            
#             plt.figure(figsize = (5, 5))
#             ax2 = plt.gca()

#             ax2.set_facecolor('None')

#             x = ((time - t0 + 0.5*period) % period) -( 0.5*period)
#             m = np.abs(x) < 0.5
#             ax2.scatter(
#                 x[m],
#                 np.array(flux)[m],
#                 color='gray',
#                 s=5,
#                 alpha=0.8,
#                 zorder=2)

#             x_new = np.linspace(-0.5, 0.5, 1000)
#             f = model.model(x_new + t0, period, duration, t0)

#             ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
#             ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
#             ax2.set_ylabel('Relative Flux')#, color = '#CC9966', fontsize = 40);
#             plt.show()
# #             print('T0: ', results.T0, 'duration: ', results.duration, 'npoints_dur: ', np.ceil(results.duration/30.))
            
            
#         if not np.abs(np.diff(np.array(sorted(results.power))[[-1, -4]]))>2*np.nanstd(results.power) or len(time_new)==0 :
#             print('FOUND NO PLANET: FINISHING THIS LC')
#             return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit
        
        
#         else: 
#             intransit = np.logical_or(intransit, transit_mask(time, results.period, (2*results.duration/24)+(1/6), results.T0[0]))
            
#             depths.append(results.depth)
#             periods.append(results.period)
#             T0.append(results.T0)
#             Tdur.append(results.duration)
#             print('FOUND A PLANET: CONTINUING')
#             return using_TLS_to_find_periodic_signals(time, flux, u, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0, Tdur = Tdur, depths = depths, first = False)


# ------------------------
# Helper Functions
# ------------------------

def compute_log_likelihood(flux, model_flux, flux_err):
    residuals = flux - model_flux
    chi2 = np.sum((residuals / flux_err)**2)
    return -0.5 * chi2

def compute_BIC(logL, n, k):
    return k * np.log(n) - 2 * logL

def compute_AIC(logL, k):
    return 2 * k - 2 * logL


def build_box_model(time, t0, duration, depth, period = -1):
    """
    Build a box-shaped transit model in the time domain (not phase-folded).
    
    Parameters:
    ----------
    time : array
        Time array in days.
    period : float
        Orbital period in days.
    t0 : float
        Transit midpoint in days.
    duration : float
        Transit duration in days.
    depth : float
        Transit depth (fractional).
    
    Returns:
    -------
    model_flux : array
        Flux model with transits applied.
    """
    model_flux = np.ones_like(time)
    # Compute phase relative to t0 without folding
    # For each time point, check if it's in transit for any cycle
    if period>-1: 
        phase_offset = (time - t0) % period
    else: 
        phase_offset = (time - t0) 
        
    in_transit = (phase_offset < duration / 2) | (phase_offset > (period - duration / 2))
    # Apply depth to in-transit points
    model_flux[in_transit] -= depth
    return model_flux

# ------------------------
# Main Recursive Function
# ------------------------

def using_BLS_recursive(time, flux, flux_err = None, intransit=None,
                            verbose=True, plot=True, max_planets=10,
                            min_SNR=8, min_SDE = 10,
                            periods=None, T0=None, Tdur=None, depths=None, first=False):
    """
    Recursive multi-planet search using GERBLS pyFastBLS + run_double and BIC/AIC-based model selection.
    """

    if intransit is None:
        intransit = np.zeros_like(time, dtype=bool)
    if flux_err is None:
        flux_err = np.std(flux) * np.ones_like(flux)

    if first:
        periods, T0, Tdur, depths = [], [], [], []
    df = 1E-4
    durations = np.linspace(0.01, 0.5, 50)

    # Mask in-transit points
    time_new, flux_new, flux_err_new = time[~intransit], flux[~intransit], flux_err[~intransit]
    if len(time_new) < 10:
        print("Stopping: insufficient data.")
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    # Prepare data for BLS
    freq_fact_prelim = df/min(durations)*(np.nanmax(time_new)-np.nanmin(time_new))**2
    freq_fact_exp = np.ceil(np.log10(freq_fact_prelim))    

    start = tm.time()

    model     = BoxLeastSquares(time_new, flux_new)
    max_per   = np.min([50., (max(time_new)-min(time_new))*4/5])
    max_dur   = np.min([0.5,  (max(time_new)-min(time_new))/2])
        

    results   = model.autopower(durations[durations<max_dur], frequency_factor = np.max([10, (10**(freq_fact_exp-1))/2]), maximum_period=max_per)#, objective='snr', )

    end = tm.time()    

    my_median = running_median(results.power, kernel = min((25, int(len(time_new)/10))))
    
    
#     print('my median', my_median)
    results['power_final'] = results.power - my_median

    check_pwr_final_indxs = checking_last_BLS_power_for_artificial_inflation(results['power_final'])
    index = np.argmax(results.power_final[check_pwr_final_indxs])

    period = results.period[index]
    t0 = results.transit_time[index]
    duration = results.duration[index]
    depth = results.depth[index]
    
    print('depth found', results.depth[index])
    
    sorted_results = np.sort(results['power_final'])

    # Compute SDE
    mad = sst.median_abs_deviation(results['power_final'])
    results['SNR'] =  results['power_final']/(mad/0.67)
    
    sde  = (results['power_final'][index] - np.mean(results['power_final'])) / np.std(results['power_final'])
#     sde2 = (sorted_results[-1] - sorted_results[-2] ) / np.std(sorted_results[:-2])

#     if verbose:
    print(f"Candidate: P={period:.4f} d, SDE={sde:.2f},min_SDE={min_SDE:.2f}, SNR = {results['SNR'][index]:.4f}, min_SNR = {min_SNR:.2f}")
#     print(f"Candidate: P={period:.4f} d, SDE={sde:.2f}, SDE2={sde2:.2f}, min_SDE={min_SDE:.2f}")

    

    mask = results['SNR'] > min_SNR


    if plot: # and np.ceil(results.duration[index]/np.nanmedian(np.diff(time)))>=3:
        plt.figure(figsize = (10, 6))
        val_triangles = min(results.SNR)-np.std(results.SNR)
        ax = plt.gca()
        ax.scatter(period, val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

        plt.xlim(np.min(results.period), np.max(results.period))
        for n in range(2, 10):
            ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
            ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
        plt.ylabel(r'SNR')#, fontsize = 40)
        plt.xlabel('Period (days)')#, fontsize = 40)

        ax.plot(results.period, results.SNR, color = 'k', lw=0.65)

        plt.show()
        plt.close()
        if duration<period:
            plt.figure(figsize = (5, 5))
            ax2 = plt.gca()

            x = ((time_new - t0 + 0.5*period) % period) -( 0.5*period)
            m = np.abs(x) < 0.5
            ax2.scatter(
                x[m],
                flux_new[m],
                color='k',
                s=5,
                alpha=0.8,
                zorder=10)

            x_new = np.linspace(-0.5, 0.5, 1000)

            f = model.model(x_new + t0, period, duration, t0)

            f2 = build_box_model(x_new+t0, t0, duration, depth, period)
            ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
#             ax2.plot(x_new, f2, color='violet', lw = 1, alpha = 0.6, zorder = 5)

#             ax2.set_xlim(-0.5, 0.5)
            ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
            ax2.set_ylabel('Relative Flux')#, color = 'k', fontsize = 40);
            plt.show()
    if not mask.any() or sde<min_SDE:
        if sde<min_SDE:
            print("Stopping: SDE below threshold.")
        if not mask.any():
            print('Stopping: SNR below threshold.')
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    try:
        stats = model.compute_stats(period, duration, t0)
        print('number transit times in baseline:', len(stats["transit_times"][stats["per_transit_count"]>0]), ' \nnumber of point in each transit: ', stats["per_transit_count"][stats["per_transit_count"]>0], '\ntransit likelihood:', stats["per_transit_log_likelihood"][stats["per_transit_count"]>0])

    except ValueError as err:
        print('getting error: ', err)

    stats = model.compute_stats(period, duration, t0)
    transit_times_all = stats["transit_times"][np.where(stats["per_transit_count"]>0)[0]]
    single = False
    if len(transit_times_all)<2:
        single = True

    repeat = False
    if len(periods)>0:# and len(factor_indxs)>0:

        new_period, keep_factor, pop_per  = checking_BLS_periodicity(period, periods, t0, T0)

        if keep_factor>0:
            repeat=False
            intransit = np.logical_or(intransit, transit_mask(time, new_period, duration, t0))
            period = new_period  
            if pop_per>0:
                print(f'popping_period: {pop_per}, and keeping period {period}, as the old is {pop_per/period}x the new')
                pop_indx = periods.index(pop_per)
               
                periods.pop(pop_indx)
                
        if keep_factor < -50 :

            intransit = np.logical_or(intransit,  transit_mask(time, new_period/2, duration, t0, buffer = 0.3))
        else:
            intransit = np.logical_or(intransit, transit_mask(time, new_period, duration, t0, buffer = 0.3))

        if keep_factor<0:
            repeat = True

    # Build models for likelihood
    model_flux = model.model(time_new, period, duration, t0)

    null_flux = np.ones_like(flux_new)
    
    logL_transit = compute_log_likelihood(flux_new, model_flux, flux_err_new)

    logL_null = compute_log_likelihood(flux_new, null_flux, flux_err_new)

    n, k = len(time_new), 3
    bic_transit = compute_BIC(logL_transit, n, 3)

    bic_null = compute_BIC(logL_null, n, 1)
    delta_BIC = bic_null - bic_transit


    if (single) or (repeat):
        print('masking this single transit or repeat detection and continuing')

#             return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit
#         else:
        if single:
            print("Candidate rejected - masking single event")
        elif repeat:
            print("Candidate rejected - masking repeat discovery transits")
        else:
                print("Candidate rejected: insufficient BIC improvement - will still mask and try again.")
            # Mask transits using your transit_mask
        intransit = np.logical_or(intransit, transit_mask(time, period, duration, t0))

        return using_BLS_recursive(time, flux, intransit=intransit,flux_err=flux_err,periods=periods, T0=T0, Tdur=Tdur, depths=depths, first=False)

    # Accept candidate
    periods.append(period)
    T0.append(t0)
    Tdur.append(duration)
    depths.append(depth)
    print('depths all', depths, 'transt durations: ', Tdur)

    if verbose:
        print(f"Accepted planet: P={period:.4f} d")

    # Mask transits using your transit_mask
    intransit = np.logical_or(intransit, transit_mask(time, period, duration, t0))

    if len(periods) >= max_planets:
        print("Reached max planets.")
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    # Recurse
    return using_BLS_recursive(time, flux, intransit=intransit,flux_err=flux_err,periods=periods, T0=T0, Tdur=Tdur, depths=depths, first=False)

    

def fitting_periodic_planets(time, flux, flux_err, pers, t0s, depths, ab, intransit, verbose=True, save_phaseFold = False, total_time = True, data_file = '.', chain_diff = 0):
#             print('period', periods_multis[jjj])
    periods  = []
    T0_vals  = []
    Tdur     = []
    depth    = []
    SNR_vals = []
    
#     params_df   = []
    
    if type(total_time) == bool:
        total_time = time


    print('len(t0s)', len(t0s))
    for iii in range(len(pers)): 
        if pers[iii]>0.25:

            params_df, conv, conv_attempt = pymc_new_general_function(time, flux, flux_err, t0s[iii], [pers[iii], ab, depths[iii]], 'Periodic')
            if len(params_df)>0:

                T0_, period_, depth_, tdur_, SNR = params_df.loc['t0', 'mean'], params_df.loc['Per', 'mean'], params_df.loc['depth', 'mean'], params_df.loc['dur', 'mean'], params_df.loc['SNR', 'mean']
                print('params df', params_df)

                print('checking convergence 1')
                pd.DataFrame({'TICID':[con.TICID], 't0':[T0_], 'per':[period_], 'depth':[depth_], 'converged': [True], 'conv_on_run':[conv_attempt]}).to_csv('../checking_convergence_output/'+str(con.TICID)+'_'+str(round(T0_, 5))+'_Yconv_per.csv')


            else:
                T0_, period_, tdur_, depth_, SNR = np.nan, np.nan, np.nan, np.nan, 0

                print('checking convergence 2')
                pd.DataFrame({'TICID':[con.TICID], 't0':[t0s[iii]], 'per':[pers[iii]], 'depth':[depths[iii]], 'converged': [False], 'conv_on_run':[np.nan]}).to_csv('../checking_convergence_output/'+str(con.TICID)+'_'+str(round(t0s[iii], 5))+'_Nconv_per.csv')


            if not np.isnan(T0_):

                periods.append(period_)
                T0_vals.append(T0_)
                Tdur.append(tdur_)
                depth.append(depth_)
                SNR_vals.append(SNR)

            gc.collect()

            intransit = np.logical_or(intransit, transit_mask(total_time, period_, float((1.5*tdur_)), float(T0_)))

    print('params df 3', params_df)
    return T0_vals, periods, depth, Tdur, SNR_vals, intransit, params_df

            
        
def searching_for_periodic_signals(data_file, ab, TLS = False, verbose = True, save_file = True, save_phaseFold = False):
    
    params = []
    periods  = []
    T0_vals  = []
    Tdur     = []
    depth    = []
    SNR_vals = []


    total_time_flux_df = pd.read_csv(data_file).dropna(subset = ['FLUX'])

    
    
    time      = np.array(total_time_flux_df['TIME'].astype(float))
    flux      = np.array(total_time_flux_df['FLUX'].astype(float))
    flux_err  = np.array(total_time_flux_df['FLUX_ERR'].astype(float))
    intransit = np.full(len(time), False)

    print('running search on all data')

    ###running search on whole dataset
    if TLS:
        periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_per = using_TLS_to_find_periodic_signals(time, flux, u = ab, intransit =intransit, verbose = verbose)

    else:

        print('recursive BLS')
        periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_per = using_BLS_recursive(time, flux, intransit = intransit, verbose = verbose, first=True)
        
#     print('depths', depth_multis)
    
    if len(T0_multis)>0:
        
        sort_indices = np.argsort(periods_multis)
        
        nt0_vals, nperiods, ndepth, nTdur, nSNR_vals, intransit, params_df = fitting_periodic_planets(time, flux, flux_err, periods_multis[sort_indices], T0_multis[sort_indices], depth_multis[sort_indices], ab, intransit_per, verbose, save_phaseFold, data_file = data_file)

        params_df.loc[len(params_df)] = [None] * len(params_df.columns) 

        params.append(params_df)

        periods.extend(nperiods)
        T0_vals.extend(nt0_vals)
        Tdur.extend(nTdur)
        depth.extend(ndepth)
        SNR_vals.extend(nSNR_vals)
    
    
    ###running search on split subsets
    print('running search on chunked data')
    
    
    
    indexes_split_unorganize = breaking_up_data(time)   
    indexes_split = sorted(indexes_split_unorganize, key=lambda x: len(x), reverse=True)
    
    if len(indexes_split) == 1 and (np.any(np.array(periods)<10.) or len(periods)==0):
        indexes_split_unorganize = breaking_up_data(time, break_val = 1.)   
        indexes_split = sorted(indexes_split_unorganize, key=lambda x: len(x), reverse=True)


#     print('split indexes lengths: ', [len(x) for x in indexes_split])
    if len(indexes_split)>1:
        for iii in range(len(indexes_split)):
            if len(indexes_split) == 1:
                print('too few indexes to run again')
                continue
    #         print('len time', len(new_time))
            intransit_split = np.array(intransit[indexes_split[iii]])
            split_time = np.array(time[indexes_split[iii]])
            split_flux = np.array(flux[indexes_split[iii]])
            split_flux_err = np.array(flux_err[indexes_split[iii]])


            if len(split_time)==0:
                print('WHY ISNT THIS WORKING')

                print(indexes_split)
                continue

            else:
                if TLS:
                    periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_small = using_TLS_to_find_periodic_signals(split_time, split_flux, u = ab, verbose = verbose)
                else:
                    periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_small = using_BLS_recursive(split_time, split_flux, verbose = verbose, intransit = np.full(len(split_time), False), periods = periods.copy(), T0 =T0_vals.copy(), Tdur = Tdur.copy(), depths = depth.copy())

            intransit[indexes_split[iii]] = np.logical_or(intransit_split, intransit_small)


            toss_bool = np.isin(np.round(periods_multis, 1), np.round(periods, 1))


            print('periods to not run again', np.array(periods_multis)[toss_bool])

            if len(np.array(periods_multis[~toss_bool]))>0:


                nt0_vals_split, nperiods_split, ndepth_split, nTdur_split, nSNR_vals_split, intransit, nparams_df = fitting_periodic_planets(split_time, split_flux, split_flux_err, list(np.array(periods_multis)[~toss_bool]), list(np.array(T0_multis)[~toss_bool]), list(np.array(depth_multis)[~toss_bool]), ab, intransit, verbose, save_phaseFold=False, total_time = time, data_file=data_file)

                nparams_df.loc[len(nparams_df)] = [None] * len(nparams_df.columns) 

                params.append(nparams_df)

                periods.extend(nperiods_split)
                T0_vals.extend(nt0_vals_split)
                Tdur.extend(nTdur_split)
                depth.extend(ndepth_split)
                SNR_vals.extend(nSNR_vals_split)

    print('done with multis ', iii+1, ':', len(indexes_split))

    print('params df 2', params)

    only_per_intransit = np.full(len(time), False)

    for iii in range(len(periods)):
        print('period is', periods[iii])
        new_transit_planet = transit_mask(time, periods[iii], Tdur[iii], T0_vals[iii])
        print('intransit '+str((iii+1)*7), new_transit_planet, len(np.where(new_transit_planet)[0]), len(new_transit_planet))

        only_per_intransit = np.logical_or(only_per_intransit,new_transit_planet)
        

    intransit_indexes =  np.where(only_per_intransit)
    
    return periods, T0_vals, Tdur, depth, only_per_intransit, SNR_vals,intransit_indexes, params




def executing_total_periodic_search(data_file, ticid, catalog_df = False, TLS = False, verbose = True, save_intrans = True, save_time = True, save_phaseFold = True):

    tm1 = tm.time()
    column_names = ['TICID', 'planet_name', 'period', 'T0', 'Tdur', 'depth', 'SNR']
    params = []
    planet_df = pd.DataFrame(columns=column_names)

    ab, smass, smass_min, smass_max, sradius, sradius_min, sradius_max = get_catalog_info(ticid, df = catalog_df)
    print('PERIODIC SEARCH')
    periods_multi, T0_multi, Tdur_multi, depth_multi, intransit, SNR, n, params = searching_for_periodic_signals(data_file, ab, TLS,  verbose, save_file = save_intrans)
    if len(params)>0:
        params = pd.concat(params)
        
    print('params df 1', params)
    print('init number of periodic planets', len(periods_multi),  periods_multi)
    nnn = 0

#     if len(set(np.round(periods_multi, 2)))<len(periods_multi):
    if len(periods_multi)>1:
        final_indxes, per_div = checking_multiples_and_duplicate_periodic_planets(periods_multi, T0_multi, depth_multi, SNR)   

        periods_multi = np.array(periods_multi)/per_div
        print('new periodic planets', len(periods_multi), periods_multi)

    elif len(periods_multi) == 1:
        final_indxes = np.array([True])
#     else: 
#         final_indxes = range(len(T0_multi)) 
    else:
        final_indxes = []
        
        
    

    for jjj in np.where(final_indxes)[0]:
        nnn+=1
        planet_name = str(nnn)
        
        print('index', jjj)
        

        planet_df.loc[len(planet_df.index)] = [int(ticid), planet_name, periods_multi[jjj], T0_multi[jjj], Tdur_multi[jjj], min([1-depth_multi[jjj], depth_multi[jjj]]), SNR[jjj]]
        
    print('done ', ticid, planet_df)

    if save_time:
        tm2 = tm.time()
        dict_time_sectors = {'TICID':[], 'time_to_run':[], 'split_searches':[]}
        dict_time_sectors['TICID'].append(ticid)
        dict_time_sectors['time_to_run'].append(tm2-tm1)
        dict_time_sectors['split_searches'].append(n)

        pd.DataFrame(dict_time_sectors).to_csv(os.path.dirname(data_file)+'/time_periodic_took_to_run.csv', index= False, mode = 'a')
    
    if len(planet_df)>0:
        
        file_name = os.path.dirname(data_file)+'/found_planet_init_params'
        kept = glob.glob(file_name+'*')

        csv_name = '.csv'
        for i in range(len(kept)):
            csv_name = '_'+str(i)+'_'+csv_name[-4:]

        planet_df.to_csv(file_name+csv_name, index = False)
        

    
    return intransit, planet_df, params


def approximate_common_denominator(float1, float2, precision=10**5):
    int1 = int(float1 * precision)
    int2 = int(float2 * precision)

    gcd = math.gcd(int1, int2)
    return gcd / precision

# In[54]:
def check_if_singles_are_periodic(T0_lst):
    new_rrr = []
    T0_vals = []
    for j in range(len(T0_lst)):
        diff = np.array(T0_lst)-T0_lst[j]
        for i in range(len(diff)):
            for k in range(len(diff)):
                if k == j or i == j:
                    continue
                else:
                    rrr = np.array(approximate_common_denominator(diff[i], diff[k]))
                    if len(rrr[rrr>2.5])>0:
                        new_rrr.append(list(rrr[rrr>2.5]))
                        T0_vals.extend([T0_lst[i], T0_lst[j], T0_lst[k]])


    set_rrr = [ele for ind, ele in enumerate(new_rrr) if ele not in new_rrr[:ind]]    
    T0_vals = list(set(T0_vals))
    
    T0_per_vals = []
    for val in list(set_rrr):
        pers = new_rrr[new_rrr == val]
        T0_min = min(T0_vals)
#         print(T0_vals)
        per_max = max(pers)
        T0_per_vals.append([T0_min, per_max, T0_vals])
    return T0_per_vals






def singles_search(ticid, data_total, intransit = [], catalog_df = False, confidence = 0.5,  verbose = True, run_1 = True, data_file = ''):
    
    print('SINGLES SEARCH')
    column_names = ['TICID', 'planet_name', 'period', 'T0', 'Tdur', 'depth']

    if not run_1:
        column_names+=['SNR']
    planet_df = pd.DataFrame(columns=column_names)
    

    df = pd.read_csv(data_total).dropna(subset = ['FLUX'])
    total_time, total_flux, total_flux_err = [np.array(df[col]) for col in ['TIME', 'FLUX', 'FLUX_ERR']]
    print('checking time again', len(total_time))

    if len(intransit)>0:
#         print('evil bs', intransit, len(intransit), len(np.where(intransit)[0]))
        total_time = total_time[~intransit]
        total_flux = total_flux[~intransit]
        total_flux_err = total_flux_err[~intransit]

    indexes_split_unorganize = breaking_up_data(total_time, break_val = 0.5, min_size = 1.)  
    all_good_indxs = []

    if len(indexes_split_unorganize)>1:
        diff_ary = np.array([max(np.array(total_time)[x])-min(np.array(total_time)[x]) for x in indexes_split_unorganize])
        all_good_indxs =np.concatenate(list(itertools.compress(indexes_split_unorganize, diff_ary>1))).ravel()
   
    if len(all_good_indxs) == 0: 
        all_good_indxs = list(range(len(total_time)))
        
#     print('all good indexes (i.e., itertools result: ', type(all_good_indxs), len(total_time))
    list_1 = list(all_good_indxs)
    list_2 = []
    for index, value in enumerate(total_time):
        list_2.append(index)
#     print('check differences in good indexes and indexes: ',  [item for item in list_1 if item not in list_2])
#     print('type of time array, ', type(total_time))
    total_time = total_time[all_good_indxs]
    total_flux = total_flux[all_good_indxs]
    total_flux_err =total_flux_err[all_good_indxs]

    ab, smass, smass_min, smass_max, sradius, sradius_min, sradius_max = get_catalog_info(ticid, df = catalog_df)
#     print('val', ab)
    
    params_df = []

    if len(total_time)>0:

        bboxes = DT_analysis(total_time, total_flux, total_flux_err, confidence)
    #     print('ran bboxes', bboxes)

        print('number singles found', len(bboxes))
        t0_singles, dur_singles, depth_singles = [],[],[]
        if len(bboxes)>0:
            n_events = len(bboxes)
            for j, boxes in enumerate(bboxes):
#                 SNR = calc_rudimentary_snr(boxes[3], boxes[4])
#                 if SNR>9:

                t0_singles.append(boxes[1])
                dur_singles.append(boxes[3])
                depth_singles.append(1-boxes[4])

                fig = plt.figure(figsize = (8, 8))

                ax = fig.add_subplot(111)
                ax.set_xlim(boxes[1]-2*boxes[3], boxes[1]+2*boxes[3])

    #             ax.set_ylim(1-3.5*np.median(total_flux_err), 1+3.5*np.median(total_flux_err))
                ax.scatter(total_time, total_flux, color = 'k', marker ='o', zorder = 1E6)
                detrended_lc = make_LightKurveObject(total_time, total_flux, total_flux_err)

                plot_lc_with_bboxes(detrended_lc, bboxes, ms=3, marker='.', lw=0, ax = ax)

                plt.show()


        t0_singles    = np.array(t0_singles)
        dur_singles   = np.array(dur_singles)
        depth_singles = np.array(depth_singles)


   

        new_T0_periodic = []
        planet_name = 0
        params_df = []

        for sss in range(len(t0_singles)):
            print(sss)
            planet_name+=1

            if run_1:
                planet_df.loc[len(planet_df.index)] = [int(ticid), int(planet_name), np.inf, t0_singles[sss], dur_singles[sss], depth_singles[sss]]


            else:

                new_params, conv, conv_attempt = pymc_new_general_function(np.array(total_time), total_flux, total_flux_err, t0_singles[sss], [dur_singles[sss], ab, depth_singles[sss]], 'Single')

                if len(new_params)>0:

                    params_df.append(new_params)


                    t0_, period_, depth_, tdur_, q = new_params.loc['t0', 'mean'], new_params.loc['Per', 'mean'], new_params.loc['depth', 'mean'], new_params.loc['dur', 'mean'], new_params.loc['SNR', 'mean']

                    print('checking convergence 3')
                    if not np.isnan(t0_):
                        planet_df.loc[len(planet_df.index)] = [int(ticid), int(planet_name), np.inf, t0_, tdur_,depth_, q]

                    pd.DataFrame({'TICID':[con.TICID], 't0':[t0_], 'per':[period_], 'depth':[depth_], 'converged': [True], 'conv_on_run':[conv_attempt]}).to_csv('../checking_convergence_output/'+str(con.TICID)+'_'+str(round(t0_, 5))+'_Yconv_single.csv')

                else:
                    print('checking convergence 4')

                    pd.DataFrame({'TICID':[con.TICID], 't0':[t0_singles[sss]], 'per':[np.nan], 'depth':[depth_singles[sss]], 'converged': [False], 'conv_on_run':[np.nan]}).to_csv('../checking_convergence_output/'+str(con.TICID)+'_'+str(round(t0_singles[sss], 5))+'_Nconv_single.csv')


#                 if not np.isnan(t0_):
#                     planet_df.loc[len(planet_df.index)] = [int(ticid), int(planet_name), np.inf, t0_, tdur_,depth_, q]
        if len(params_df)>0:
            params_df = pd.concat(params_df)
        print('params df singles', params_df)

    return planet_df, params_df





# In[55]:


#my mcmc functions


def lnprob(pars,flux,unc,time,cad,tc, u1, u2):
    t0 = pars[0]
    P = pars[1]
    rp_rs = pars[2]
    cosi = pars[3]
    a = pars[4]
    norm = pars[5]
    b = np.abs(a*cosi)
    #make sure all pars are good

    if np.abs(tc-t0)>0.25: return -np.inf
    if np.max([u1,u2,cosi,rp_rs]) > 1.: return -np.inf
    if np.min([u1,u2,cosi,rp_rs]) < 0.: return -np.inf
    if a < 1: return -np.inf
    if b > 1: return -np.inf

    flux_theo = predict_lc(time,t0,P,rp_rs,cosi,a,u1,u2,cad)
    flux=flux*norm
    unc=unc*norm
    result = 0.-0.5*np.sum(((flux_theo-flux)/unc)**2.)
    if np.isfinite(result) != True:
        #print('bad result')
        return -np.inf
    return result



def easy_data_phaseFold(indxs, *args):
    folded = [arg[indxs] for arg in args if args is not None]
    return np.array(folded)
    
def lnprob_MCMC_global(pars):
    flux,unc,time,cad,tc, u1, u2 = PARAMS
    t0 = pars[0]
    P = pars[1]
    rp_rs = pars[2]
    cosi = pars[3]
    a = pars[4]
    norm = pars[5]
    b = np.abs(a*cosi)
    #make sure all pars are good

    if np.abs(tc-t0)>0.25: return -np.inf
    if np.max([u1,u2,cosi,rp_rs]) > 1.: return -np.inf
    if np.min([u1,u2,cosi,rp_rs]) < 0.: return -np.inf
    if a < 1: return -np.inf
    if b > 1: return -np.inf

    # flux_theo = predict_lc(time,t0,P,rp_rs,cosi,a,u1,u2,cad)

    x = ((time - t0 + 0.5*P) % P) -( 0.5*P)

    flux_theo = predict_lc(x+t0,t0,P,rp_rs,cosi,a,u1,u2,cad)

    m = np.abs(x) < 0.5

    flux=flux*norm
    unc=unc*norm

    flux_theo_phase, flux_phase, unc_phase = easy_data_phaseFold(np.array(m), flux_theo, flux, unc)
    
    result = 0.-0.5*np.sum(((flux_theo_phase-flux_phase)/unc_phase)**2.)
    if np.isfinite(result) != True:
        #print('bad result')
        return -np.inf
    return result


def lnprob_MCMC_global_single(pars):
    flux,unc,time,per,cad,tc, u1, u2= PARAMS
    t0 = pars[0]
    rp = pars[1]
    cosi = pars[2]
    a = pars[3]
    norm = pars[4]
    b = np.abs(a*cosi)
    #make sure all pars are good

    if np.abs(tc-t0)>0.25: return -np.inf
#     if np.max(rp)>0.25: return -np.inf #attempting to set some upper bounds here just for the singles
    if np.max([u1,u2,cosi,rp]) > 1.: return -np.inf
    if np.min([u1,u2,cosi,rp]) < 0.: return -np.inf
    if a < 1: return -np.inf
    if b > 1: return -np.inf

    flux_theo = predict_lc(time,t0,per,rp,cosi,a,u1,u2,cad)
    flux=flux*norm
    unc=unc*norm
    result = 0.-0.5*np.sum(((flux_theo-flux)/unc)**2.)
    if np.isfinite(result) != True:
        #print('bad result')
        return -np.inf
    return result




def predict_lc(time_lc,t0,P,rp_rs,cosi,a,u1,u2,cad):
    oversample = 4
    e = 0.
    omega = np.pi/2.
    inc = np.arccos(cosi)*180./np.pi
    params = batman.TransitParams()
    params.t0  = t0
    params.per = P
    params.rp  = rp_rs
    params.a   = a
    params.inc = inc
    params.ecc = e
    params.w = omega*180./np.pi
    params.u = [u1,u2]
    params.limb_dark = "quadratic"
        
    if not cad>0:
        cad = 30.
    m = batman.TransitModel(params, time_lc ,supersample_factor = oversample, exp_time = cad/24./60.)

    flux_theo = m.light_curve(params)

    return flux_theo




# In[56]:
def extract_summary_dataframe(trace, hdi_prob=0.68):
    """
    Extracts a summary DataFrame from an ArviZ trace object with specified statistics.

    Parameters:
    - trace: ArviZ InferenceData object
    - hdi_prob: float, the probability for the HDI interval (default is 0.68)

    Returns:
    - pandas DataFrame with variables as index and columns: mean, median, sd, hdi_16%, hdi_84%, r_hat
    """
    # Get summary with specified HDI
    summary = az.summary(trace, hdi_prob=hdi_prob)

    # Compute median manually

    
    median_dataset = trace.posterior.median(dim=["chain", "draw"])
    medians = {var: float(median_dataset[var]) for var in median_dataset.data_vars}


    # Merge median into summary
    summary["median"] = medians

    # Reorder and select desired columns
    selected_columns = ['mean', 'median', 'sd', 'hdi_16%', 'hdi_84%', 'r_hat']
    custom_summary_df = summary[selected_columns]
    
    print('custom summary', custom_summary_df)

    return custom_summary_df


def sample_until_converged(model, max_attempts=5, rhat_threshold=1.1, chains=4,cores=None, mp_context="spawn"):
    # Get all free random variables in the model
    
    cores = min(chains, os.cpu_count() or 1) if cores is None else cores

    free_vars = model.free_RVs
    if not free_vars:
        raise ValueError("No free random variables found for sampling.")
    print('free vars', free_vars)
    # Use Metropolis for all free RVs
#     step = pm.Metropolis(vars=free_vars)

    step = pm.DEMetropolisZ(vars=free_vars)#, target_accept=0.8) 

    for attempt in range(1, max_attempts + 1):
        print(f"Sampling attempt {attempt}...")
        trace = pm.sample(step=step, draws=5000*attempt, tune=2000*attempt, chains=chains, cores = cores, 
            # use a safe, explicit multiprocessing context
            mp_ctx=mp.get_context(mp_context),
            # avoid identical RNG streams across chains
            random_seed=list(range(chains)),
)

        summary = az.summary(trace)
        if (summary['r_hat'] < rhat_threshold).all():
            print(f"Converged on attempt {attempt}")
            return trace, attempt
#         print('checking nans trace', trace.posterior['SNR'])
        print('checking nanas summary', az.summary(trace))
        print(f"Attempt {attempt} failed to converge. Retrying...")

    raise RuntimeError("Model did not converge after multiple attempts.")


def min_relative_ess(idata, total_draws):
    """
    Minimum relative ESS across all posterior variables.
    ArviZ versions differ on `relative=` availability, so handle both.
    """
    try:
        ess_rel = az.ess(idata, method="bulk", relative=True)
        return float(ess_rel.to_array().min())
    except TypeError:
        ess_abs = az.ess(idata, method="bulk")
        return float(ess_abs.to_array().min()) / float(total_draws)




def sample_until_converged_smc(
    model,
    ess_ratio_target=0.70,     # target fraction of draws considered "effective"
    max_attempts=5,            # retries with increasing population size
    base_draws=2000,           # starting population per chain
    chains=4,
    cores=None,
    random_seed=None,          # int or list[int]; if None, will use a simple one-liner
    var_names=None,            # optional quick-look names to summarize
    progressbar=True,
):
    """
    Run SMC repeatedly, increasing draws until min relative ESS meets target.
    Returns (idata, attempt).
    """
    # Cores: one process per chain, capped at available CPUs
    cores = min(chains, os.cpu_count() or 1) if cores is None else cores

    # Seeds: keep it simple and predictable
    if random_seed is None:
        # One-liner Mallory suggested
        seeds = list(2026 + np.array(range(chains)))
    elif isinstance(random_seed, int):
        # Same base int per chain, still simple and reproducible
        seeds = [int(random_seed)] * chains
    else:
        seeds = list(random_seed)  # assume list[int]
        if len(seeds) != chains:
            raise ValueError(f"random_seed list length {len(seeds)} != chains {chains}")

    with model:
        # Names of free RVs once

        for attempt in range(1, max_attempts + 1):
            draws = int(base_draws * attempt)
            print(f"[SMC attempt {attempt}] draws={draws}, chains={chains}, cores={cores}")

            idata = pm.sample_smc(
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seeds,              # your per-chain seeds
                threshold=ess_ratio_target,     # ESS fraction for resampling
                return_inferencedata=True,
                progressbar=progressbar,
                compute_convergence_checks=False,
            )

            total_draws = draws * chains

            # 4) Your ESS metric (uses your existing function)
            min_ratio = min_relative_ess(idata, total_draws)
            print(f"  min relative ESS ≈ {min_ratio:.3f}")

            # Optional quick-look summaries
            if var_names:
                try:
                    print(az.summary(idata, var_names=var_names))
                except Exception:
                    pass
            try:
                print(az.summary(idata, var_names=["Per", "rp_rs", "a_rs", "b", "t0"]))
            except Exception:
                pass

            if min_ratio >= ess_ratio_target:
                print(f"Converged by ESS on attempt {attempt}")
                return idata, attempt

    print("Returning last attempt; target ESS not met")
    return idata, attempt

                                


class BatmanOp(Op):
    itypes = [pt.dvector, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def make_node(self, *inputs):
        # Convert all inputs to tensors if they aren't already
        converted_inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        return Apply(self, converted_inputs, [o() for o in self.otypes])

    def perform(self, node, inputs, outputs):
        time, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, cad = inputs
        params = batman.TransitParams()

        params.t0 = float(t0)
        params.per = float(per)
        params.rp = float(rp_rs)
        params.a = float(a_rs)
        params.inc = float(inc)
        params.u = [float(u1), float(u2)]
        params.ecc = float(ecc)
        params.w = 90.0
        params.u = [float(u1), float(u2)]
        params.limb_dark = "quadratic"
        m = batman.TransitModel(params, time, supersample_factor=4, exp_time=cad/24./60.)
        outputs[0][0] = m.light_curve(params)

    def grad(self, inputs, g_outputs):
        # For now, return zeros (no gradient)
        return [pt.zeros(inp.shape, dtype=pt_config.floatX) for inp in inputs]
    
def set_up_variables_for_pymc_fit(time, flux, unc, t0, other_pars, type_fn):
    """
    Returns:
        use_time, use_flux, use_unc, per, u1, u2, depth, cad

    cad is a robust effective cadence in MINUTES.
    """
    mask = np.logical_or(np.logical_or(np.isnan(flux),  np.isnan(time)),  np.isnan(unc))
    time = time[~mask]
    unc  = unc[~mask]
    flux = flux[~mask]

    cad = np.nanpercentile(np.clip(np.diff(np.unique(time))*60.*24., 200/60, 30), 95) #minutes

    if type_fn == 'Single':
        dur, ab, depth = other_pars
        k = np.sqrt(depth)
        per1 = np.max(time)-np.min(time)
        per2 = ((3*np.pi/con.G/con.rho_star)** 0.5 ) * (dur/np.pi/(1+k)) ** 1.5
        per = np.max([per1, per2])
        print('time difference', np.max(time)-np.min(time), 'checking duration units (must be < 0.5 for days, <12 for hours)', dur)
        if per<10:
            per = 27.8
        
        indxes = np.where(np.abs(time-t0)<(1.+dur))
        
        use_time = np.array(time)[indxes]
        use_flux = np.array(flux)[indxes]
        use_unc  = np.array(unc)[indxes]
    elif type_fn == 'Periodic':
        per, ab, depth = other_pars
        use_time = np.array(time)
        use_flux = np.array(flux)
        use_unc  = np.array(unc)
        
        print(per, type(per))
        
        a_smaj_guess = float((con.G * con.rho_star * (per ** 2) / 3 / np.pi) ** (1/3))
        dur =  min([0.5/3, (float(per) / float(a_smaj_guess * np.pi))])  # days

    u1, u2 = ab

    return use_time, use_flux, use_unc, float(per), u1, u2, float(depth), 1.5*float(dur), cad


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





def pymc_new_general_function(time, flux, unc, T0, other_pars, type_fn,
                              verbose=True, keep_ld_fixed=True):

    # Unpack and precompute using your existing helper
    time, flux, unc, Per, u1, u2, Depth, pTdur, cad = set_up_variables_for_pymc_fit(
        time, flux, unc, T0, other_pars, type_fn
    )
    
    batman_op = BatmanOp()

    # --- Tightening bounds and simple physics helpers (outside the model) ---

    # Build contiguous observation windows from actual timestamps
    windows = make_windows_from_time_stamps(np.array(time), gap_threshold=0.5)

    # Count how many integer k place transit centers inside these windows
    nobs_est = 0
    for s, e in windows:
        k_low  = np.ceil((s - T0) / Per)
        k_high = np.floor((e - T0) / Per)
        nobs_est += int(max(0, k_high - k_low + 1))

    ecc = 0.

    with pm.Model() as model:

        # Use a duration-sized box for t0 in both modes
        t0 = pm.Uniform("t0", lower=T0 - pTdur, upper=T0 + pTdur)

        if type_fn == 'Single':
            print('single')
            # Keep your Single logic, but give Per a modest bound window
#             per   = pm.Uniform( "Per",  lower=max(0.25, Per * 0.5), upper=Per * 2.5)
#             ecc   = pm.TruncatedNormal("Eccen", mu=0., sigma=0.25, lower=0, upper=1)
            # a/R*: tie to sampled per
            a_rs_from_Per_init = float(((con.G * con.rho_star * (Per ** 2)) / (3.0 * np.pi)) ** (1.0 / 3.0))
#             a_rs_mu = pm.Deterministic("a_rs_mu", (con.G * con.rho_star * (per ** 2) / 3 /np.pi) ** (1/3))
#             a_rs    = pm.TruncatedNormal("a_rs", mu=a_rs_mu, sigma=5.0, lower=1.0)

            a_rs = pm.TruncatedNormal("a_rs", mu=a_rs_from_Per_init, sigma=5.0, lower=1.0,
                              initval=a_rs_from_Per_init)

            per = pm.Deterministic("Per", pt.sqrt((3.0 * np.pi) / (con.G * con.rho_star)) * a_rs ** 1.5)

            fold_this = False
            
        elif type_fn == 'Periodic':
            print('periodic, period =', Per, 'days, transits observed = ', nobs_est)

            # Period bounds:
            #  - many transits: very tight uniform (±1%)
            #  - otherwise: TN with ±20% support and sigma ~ 5% of Per

            if nobs_est > 3:
                P_lower = max(0.25, Per * 0.99)
                P_upper = Per * 1.01
                P_sigma = None
                fold_this = True
                # MANY TRANSITS MODE (DECOUPLED)
                # 1. Keep tight uniform for ephemeris
                per = pm.Uniform("Per", lower=P_lower, upper=P_upper)

                # 2. DO NOT tie a_rs to Per via Kepler’s law
                #    Instead give an uninformative wide prior
                a_rs = pm.Uniform("a_rs", lower=1.0, upper=300.0)

            else:
                P_lower = max(0.25, Per * 0.80)
                P_upper = Per * 1.20
                P_sigma = max(0.1, 0.05 * Per)

                # LOW/FEW TRANSITS MODE (original behavior)
                per = pm.TruncatedNormal("Per", mu=Per, sigma=P_sigma, lower=P_lower, upper=P_upper)
                # original line
                
                print('con rho_star', con.rho_star, type(con.rho_star))
                fold_this = False
                a_rs_mu = pm.Deterministic("a_rs_mu", (con.G * con.rho_star * (per ** 2) / 3 /np.pi) ** (1/3))
                a_rs = pm.TruncatedNormal("a_rs", mu=a_rs_mu, sigma=3., lower=1.0)
       
        # Depth and geometry
        rp_rs = pm.TruncatedNormal("rp_rs", mu=pt.sqrt(Depth),
                                   sigma=pt.maximum(0.02, 0.5 * pt.sqrt(Depth)),
                                   lower=0, upper=1)
        b     = pm.TruncatedNormal('b', mu=0, sigma=0.01, lower=0, upper=1)

        depth  = pm.Deterministic('depth', rp_rs**2)

        # 1) Inclination: clip to arccos domain
        cosi = pm.Deterministic("cosi", pt.clip(b / a_rs, -1.0 + 1e-12, 1.0 - 1e-12))
        inc  = pm.Deterministic("inclination", pt.arccos(cosi) * 180.0 / np.pi)

        # 2) Duration terms: guard tiny denominators / underflow
        eps  = 1e-12
        root = pt.sqrt(pt.clip(1.0 - b**2, 1e-12, 1.0))
        T_dur0 = per / ((a_rs + eps) * np.pi)
        tau    = pm.Deterministic('tau', rp_rs * T_dur0 / root)
        dur    = pm.Deterministic('dur', root * T_dur0 + tau)
        win    = pm.Deterministic('win', dur * 2.)



        # Masks
        if type_fn == 'Periodic':
            intran_mask = transit_mask_tensors(time, per, dur, t0)  # boolean PyTensor
        elif type_fn == 'Single':
            intran_mask = pt.abs(time - t0) < (dur / 2.)

        outran_mask = pt.invert(intran_mask)

        # Out-of-transit scatter estimate
        out_flux = flux * outran_mask  # zeros elsewhere
        count = pt.maximum(pt.sum(outran_mask), 1)
        mean_out = pt.sum(out_flux) / count
        std_out = pt.sqrt(pt.sum(outran_mask * (flux - mean_out)**2) / count)

        N_tran = pt.sum(intran_mask)
        uq = pt.ones_like(flux) * std_out
        sigs = pt.switch(N_tran > 0, pt.mean(pt.where(intran_mask, uq, 0)), 1e6)

        # SNR diagnostic
        print('N_intran', pm.draw(N_tran), 'depth', pm.draw(depth), 'sig', pm.draw(sigs))
        SNR_val = pt.switch(pt.gt(N_tran, 0), pt.sqrt(N_tran) * depth / sigs, 0)
        SNR_clipped = pt.clip(SNR_val, 0, 1e4)
        SNR_final = pt.where(pt.eq(SNR_clipped, 1e4), 1, SNR_clipped)
        if not fold_this:
            SNR = pm.Deterministic("SNR", SNR_final)

        norm = pm.Deterministic("norm", median_pytensor(out_flux))

        # Likelihood
        if fold_this:
            folded_phase = ((time - T0 + 0.5 * Per) % Per) - (0.5 * Per)
            
            sort_indx = np.argsort(folded_phase)
            
            phase = folded_phase[sort_indx]
            use_index = np.abs(phase) < min([0.5, 3*pTdur])
            
            dt_minutes_min = np.nanpercentile(np.diff(np.unique(np.sort(time))), 5) * 24.0 * 60.0
            p_cad = float(np.clip(dt_minutes_min, 0.2, 60.0))

            p_flux_model = batman_op(phase[use_index] + T0, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, p_cad)

            intran_mask = transit_mask_tensors(phase + t0, per, dur, t0)
            std_out = pt.sqrt(pt.sum(outran_mask * (flux - mean_out)**2) / count)

            N_tran = pt.sum(intran_mask)
            uq = pt.ones_like(flux) * std_out
            sigs = pt.switch(N_tran > 0, pt.mean(pt.where(intran_mask, uq, 0)), 1e6)
            print('N_intran', pm.draw(N_tran), 'depth', pm.draw(depth), 'sig', pm.draw(sigs))

            SNR_val = pt.switch(pt.gt(N_tran, 0), pt.sqrt(N_tran) * depth / sigs, 0)
            SNR_clipped = pt.clip(SNR_val, 0, 1e4)
            SNR_final = pt.where(pt.eq(SNR_clipped, 1e4), 1, SNR_clipped)
            SNR = pm.Deterministic("SNR", SNR_final)

            pm.Normal("obs", mu=p_flux_model* norm, sigma=unc[sort_indx][use_index], observed=flux[sort_indx][use_index])
        else:
            flux_model = batman_op(time, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, cad)
            pm.Normal("obs", mu=flux_model * norm, sigma=unc, observed=flux)

    # Sampling (use your SMC wrapper; no custom start needed)
    with model:
        try:
            trace, conv_attempt = sample_until_converged(model)
            summary = extract_summary_dataframe(trace)
        except RuntimeError as error:
            return (
                pd.DataFrame(columns=['mean', 'median', 'sd', 'hdi_16%', 'hdi_84%', 'r_hat']),
                False,
                np.nan
            )

    if verbose:
        az.plot_trace(trace)
        az.plot_posterior(trace)
        plt.show()

    print('summary', summary)
    return summary, True, conv_attempt


def flatten_summary_blocks(F):
    """
    Transforms a summary DataFrame F with np.nan separator rows into a flattened format.
    Each block of rows (separated by NaN rows) becomes a single row in the output,
    with columns named as 'variable_stat'.

    Parameters:
    - F: pandas DataFrame with summary blocks separated by rows of NaNs

    Returns:
    - pandas DataFrame with one row per block and flattened columns
    """
    blocks = []
    current_block = []

    for idx, row in F.iterrows():
        if row.isnull().all():
            if current_block:
                blocks.append(pd.DataFrame(current_block, columns=F.columns, index=[r.name for r in current_block]))
                current_block = []
        else:
            current_block.append(row)

    # Add the last block if it exists
    if current_block:
        blocks.append(pd.DataFrame(current_block, columns=F.columns, index=[r.name for r in current_block]))

    # Flatten each block into a single row
    flattened_rows = []
    for block in blocks:
        flat_row = {}
        for var_name, row in block.iterrows():
            for col in block.columns:
                flat_row[f"{var_name}_{col}"] = row[col]
        flattened_rows.append(flat_row)

    # Create the final DataFrame
    final_df = pd.DataFrame(flattened_rows)

    return final_df
    

def sort_arrays_by_time(total_time, *args):
#     for arg in args: 
#         print(type(arg))
#         print('arg', len(arg))
#         print(args[:10])
    print('len total time', len(total_time))
    return [np.array(arg)[np.argsort(total_time)] for arg in args] 

def sort_arrays_by_index(index_lst, *args):
    return [[arg[index] for index in index_lst] for arg in args]


def bin_by_time_many_args(time,time_size_of_bins, **params):
    time = np.array(time).byteswap().newbyteorder() 

    interval = time_size_of_bins/60./24.    
    
    dict_params = {k:np.array(v).byteswap().newbyteorder() for k,v in  params.items()}
    dict_params['time'] = time        

    df = pd.DataFrame(dict_params, dtype=object)
    df = pd.concat([df, df])
    numbins = np.array(list(range(int(np.ceil((max(time)-min(time))/interval))+1)))
    bins = np.array([min(time)+x*interval for x in numbins])
    df['time_bins'] = pd.cut(df.time, bins)
    new_time = [x for x in df.groupby('time_bins').mean()['time'] if not math.isnan(x)]

    new_dict = {'time': new_time}
    for key, value in params.items():
        new_arg = [x for x in df.groupby('time_bins').mean()[key] if not math.isnan(x)]
        new_dict[key] = new_arg
        
    return new_time, new_dict


def bin_data_with_diff_cadences_many_args(total_time, min_cad = 0, **params):
    time     = np.array(total_time[np.argsort(total_time)])
    for key, value in params.items():
        params[key] = sort_arrays_by_time(total_time, value)[0]
    
    new_time = []
    
    indexes_split_unorganize = breaking_up_data(time, 1.)   
    indexes_split = sorted(indexes_split_unorganize, key=lambda x: len(x), reverse=True)
    
    cadences = [min_cad]
    for indx in indexes_split:
        split_time = time[indx]
        med_cadence = np.nanmin(np.diff(split_time))
        cadences.append(med_cadence)
        
    max_cadence = np.nanmax(cadences)
#     print('max cadence', max_cadence*60*24)
    
    dict_lst = []
    for indx in indexes_split:
#         print('len of indx? ', len(indx), len(time))
        filter_dict = {k:np.array(v)[indx] for k,v in  params.items()}
        cad = np.nanmin(np.diff(time[indx]))
#         print('other cadence', cad*60*24)
        if np.ceil(cad*60*24)<np.ceil(max_cadence*60*24):
#             print('need to bin')
            binned_time, all_params_dict = bin_by_time_many_args(time[indx], max_cadence*60*24, **filter_dict)       
#             print('these should be different', len(binned_time), len(time[indx]))
            new_time.extend(binned_time)
            dict_lst.append(all_params_dict)
        else:
            new_time.extend(time[indx])
#             filter_dict['time'] = time[indx]
            dict_lst.append(filter_dict)

    binned_dict = {}
    binned_dict['time'] = new_time

    for k in dict_lst[0].keys():
        if k!='time':
            binned_dict[k] = np.concatenate(list(binned_dict[k] for binned_dict in dict_lst))
#     print('binned dictionary', binned_dict)
    binned_args = sort_arrays_by_time(np.array(new_time), *binned_dict.values())
    
#     print('these should be different', len(total_time), len(binned_args[0]))
#     print('num args', len(binned_args))
    return binned_args



def creating_broken_axes_plots_for_DV_report_min_plot(time, flux, err, binned_time = [], binned_flux=[], binned_err = [], gs=False, subplot_val = None, ratios = []):
    
    
    if len(ratios)==0:
        diff_time_arrays = np.array([max(x)-min(x) for x in split_times])
        min_diff_time_arrays = min(diff_time_arrays)
        ratios = diff_time_arrays/min_diff_time_arrays

    if gs == False : 
        fig, axes = plt.subplots(1, len(ratios), figsize = [50, 10], sharey=True, 
                             gridspec_kw={'width_ratios': ratios})

    if gs != False:
        axes = [plt.subplot(gs[subplot_val, x]) for x in range(len(ratios))]
        
        
    if len(binned_time) == 0:
        binned_time, binned_flux, binned_err = bin_data_with_diff_cadences_many_args(time, flux = flux, err = err)

    indexes_split = breaking_up_data(time)   
    binned_indexes_split = breaking_up_data(binned_time)   

    split_times, split_fluxes, split_err = sort_arrays_by_index(indexes_split, time, flux, err)
    binned_split_times, binned_split_fluxes, binned_split_err = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err)


    
    d = .03 # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False)

#     for lll in split_times:
#         print(max(lll))
#         print(min(lll))
    
    for iii in range(len(axes)):

        axes[iii].set_xlim(min(split_times[iii])-1.5, max(split_times[iii])+1.5)
        axes[iii].scatter(split_times[iii], split_fluxes[iii], color = 'lightgrey', s = 5, zorder = 0)
        axes[iii].scatter(binned_split_times[iii], binned_split_fluxes[iii], color = 'k', s = 5, zorder = 1)
        axes[iii].tick_params(labelsize = 7)

        
    for jjj in range(len(axes)-1):
        axes[jjj].spines['right'].set_visible(False)
        
        axes[jjj].plot((1-d/ratios[jjj],1+d/ratios[jjj]),(-d,+d), **kwargs) # top-right diagonal
        axes[jjj].plot((1-d/ratios[jjj],1+d/ratios[jjj]),(1-d,1+d), **kwargs) # bottom-right diagonalaxes[1].plot((1-d,1+d), (-d,+d), **kwargs)

        kwargs.update(transform=axes[jjj+1].transAxes) # switch to the bottom axes

        axes[jjj+1].plot((-d/ratios[jjj+1],d/ratios[jjj+1]),(-d,+d), **kwargs) # top-right diagonal
        axes[jjj+1].plot((-d/ratios[jjj+1],d/ratios[jjj+1]),(1-d,1+d), **kwargs) # bottom-right diagonalaxes[1].plot((1-d,1+d), (-d,+d), **kwargs)

        axes[jjj+1].spines['left'].set_visible(False)
        axes[jjj+1].yaxis.set_ticks_position('none')
        axes[jjj+1].tick_params(axis = 'y', color = 'none', labelcolor = 'none')
    axes[0].tick_params(labelsize = 7)
    axes[0].set_ylabel('Relative Flux', fontsize = 9,rotation='vertical',)
    axes[int(np.floor((len(axes)-1)/2))].set_xlabel('Time (BJD - 2457000)', fontsize = 9)


#     plt.text(0.441, posit[1], 'Time (BJD - 2457000)', va='center', fontsize = 14,  fontweight = 'normal')

#     plt.text(0.09, 0.5, 'Relative Flux', va='center', fontsize = 15,  rotation='vertical', fontweight = 'normal')
#     plt.savefig('evil')
    return axes


# In[57]:




def find_t0_vals_within_time(min_t, max_t, t0, period):
    # Written by Mallory Harris

    # Description: for multi-planet system - uses times of transits previously found to mask when searching for additional planets

    # Arguments : time    = array of time values
    #             t0      = mid-tranist time of previously found planets
    #             Tdur    = transit duration of previously found planets
    #             period  = period of previously found planets

    # Return    : epoch_durations  = list of tuples with start and end times for each tranist in observing window

    num_per_before = 0-int(np.floor(np.abs(t0-min_t)/period))
    num_per_after = int(np.floor(np.abs(t0-max_t)/period))
    
#     print('period range', num_per_before-1, num_per_after+2)
    epochs = t0 + period*np.array(range(num_per_before-1, num_per_after+2))
    epochs = epochs[(epochs<max_t) & (epochs>min_t)]
#     print('should be sorted times', min_t, epochs, max_t)

    return epochs


#All constants that presist throughout the functions are defined in ALL CAPS
NUM_WALKERS = 20
NCHAIN = 5000
BCHAIN = 4000
CADENCES = [20/60, 2., 200/60,  10., 30.]
CAD_PS = 2.
CAD_FFI = 30.


def creating_first_DV_report_page(ticid, data_filename, planet_df, catalog_df, intransit=[], APER=False, eleanor=False, **other_pipelines):

    with PdfPages('../DV_reports/'+os.path.dirname(data_filename).split('/')[-1][:-6]+'.pdf') as pdf: #+str(planet_num)+'.pdf') as pdf:
        print('working on ' + '../DV_reports/'+os.path.dirname(data_filename).split('/')[-1][:-6]+'.pdf')#')#+str(planet_num)+'.pdf')

        df = pd.read_csv(data_filename)
        time, flux, err, trend, raw, raw_err = [np.array(df[col]) for col in ['TIME', 'FLUX', 'FLUX_ERR', 'FLUX_TREND', 'RAW_FLUX', 'RAW_FLUX_ERR']]
        
        binned_time, binned_flux, binned_err,  binned_trend, binned_raw, binned_rerr = bin_data_with_diff_cadences_many_args(time, flux = flux, err = err, trend = trend, raw = raw, raw_err = raw_err)
#         print('binned_times', binned_times)
        indexes_split = breaking_up_data(time)   
    #     indexes_split = sorted(indexes_split_unorganize, key=lambda x: min(x), reverse=True)
        binned_indexes_split = breaking_up_data(binned_time)   
    #     binned_indexes_split = sorted(binned_indexes_split_unorganize, key=lambda x: min(x), reverse=True)
        split_times, split_fluxes, split_err, split_raw, split_rerr = sort_arrays_by_index(indexes_split, time, flux, err, raw, raw_err)
        binned_split_times, binned_split_fluxes, binned_split_err, binned_split_raw, binned_split_rerr = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err, binned_raw, binned_rerr)

        diff_time_arrays = np.array([max(x)-min(x) for x in split_times])
        print('length of different time arrays: ', diff_time_arrays)
        min_diff_time_arrays = min(diff_time_arrays)
        ratios = diff_time_arrays/min_diff_time_arrays

        num_plots = 3
        if eleanor:
            num_plots+=1
        if APER:
            num_plots+=1
            
        if len(other_pipelines)>0:
            num_plots+=len(other_pipelines)
        
        
        fig0 = plt.figure(figsize=(8.5, 11),constrained_layout=True,dpi=100)
        gs = fig0.add_gridspec(1,2,width_ratios=[4.25, 1], wspace = 0.1) #create grid for subplots - makes it easier to assign where each plot goes
        
        
        gs0 = gs[0].subgridspec(7, len(split_times), wspace=0.02, width_ratios = ratios)
        gs1 = gs[1].subgridspec(1, 1)   
        
        ymin = np.nanmin([np.percentile(raw,1),np.percentile(flux, 0.5),1.-(max(planet_df['Depth']))]) #define y-axis limits by percentages to avoid using outliers 
        ymax = np.nanmax([np.percentile(raw,99.25),np.percentile(flux,99.5)])
        delta_y = np.abs(ymax-ymin)
        ymin = ymin-(delta_y*.05) #make sure ymin allows for all data        

        subplot = 0
        
        axes1  =creating_broken_axes_plots_for_DV_report_min_plot(time, raw, raw_err, binned_time, binned_raw, binned_rerr, gs0, subplot, ratios)

        axes = axes1

        for ax in axes1:
            min_vals, max_vals = ax.get_xlim()
            new_indexes = [(time>min_vals) & (time<max_vals)][0]
            ax.plot(time[new_indexes], trend[new_indexes], color = 'r', lw = 1, zorder = 10)
            ax.set_ylim(ymin, ymax)

        subplot+=1
        
        
        ymin2 = np.nanmin([np.percentile(flux, 0.25)])*0.95#,1.-(max(planet_df['Depth']))]) #define y-axis limits by percentages to avoid using es 
        ymax2 = np.percentile(flux,99.5)*1.05
        delta_y2 = np.abs(ymax2-ymin2)
        ymin2 = ymin2-(delta_y2*.05) #make sure ymin allows for all data        

        per_planets_df = planet_df[planet_df['Ptype']=='Period']

        if len(per_planets_df)>0:
            
            axes2  =creating_broken_axes_plots_for_DV_report_min_plot(time, flux, err,binned_time, binned_flux, binned_err, gs0, subplot, ratios)

            for ax in axes2:
                ax.set_ylim(ymin2, ymax2)
                min_vals, max_vals = ax.get_xlim()
                split_time = time[(time>min_vals) & (time<max_vals)]
                cad = np.min(np.diff(split_time))

                for indx, planet in per_planets_df.iterrows():
                    model_time = np.arange(min_vals,max_vals,cad) #creates a uniformly spaced around spanning the length of time measurements taken in 30 minute intervals
                    model_flux = predict_lc(model_time, planet.T0, planet.Period, planet.Rad_p, planet.Cosi, planet.Semi_Maj, planet.u1, planet.u2, cad)#*planet.Norm #create a model of the transit
                    model_flux = model_flux/np.nanmedian(model_flux)
                    ax.plot(model_time, model_flux, color = 'C'+str(indx), lw = 2, alpha = 0.7, zorder = 1E3)
         
            axes = axes+axes2
        subplot+=1

        times_ot, fluxes_ot, err_ot = time[~intransit], flux[~intransit],err[~intransit]
        
        

        binned_time_ot, binned_flux_ot, binned_err_ot = bin_data_with_diff_cadences_many_args(times_ot, flux = fluxes_ot, err = err_ot)
        
        
        single_planet_df = planet_df[planet_df['Ptype']=='Single'].reset_index(drop = True)

        if len(single_planet_df)>0:

            axes3  = creating_broken_axes_plots_for_DV_report_min_plot(times_ot, fluxes_ot, err_ot, binned_time_ot,binned_flux_ot, binned_err_ot, gs0, subplot, ratios)

            for ax in axes3:
                min_vals, max_vals = ax.get_xlim()
                split_time = times_ot[(times_ot>min_vals) & (times_ot<max_vals)]
                cad = np.min(np.diff(split_time))

                for indx, planet in single_planet_df.iterrows():
#                     bboxes = DT_analysis(split_time, fluxes_ot[(times_ot>min_vals) & (times_ot<max_vals)], err_ot[(times_ot>min_vals) & (times_ot<max_vals)], confidence = 0.65)
#                     detrended_lc = make_LightKurveObject(times_ot, fluxes_ot, err_ot)
#                 #         print(detrended_lc)
#                     plot_lc_with_bboxes(detrended_lc, bboxes, ms=3, marker='.', lw=0, ax = ax)

                    model_time = np.arange(min_vals, max_vals,cad) 
                    model_flux = predict_lc(model_time, planet.T0, planet.Period, planet.Rad_p, planet.Cosi,planet.Semi_Maj, planet.u1, planet.u2, cad)#*planet.Norm #create a model of the transit
                    
                    model_flux = model_flux/np.nanmedian(model_flux)

                    ax.plot(model_time, model_flux, color = 'C'+str(indx+len(per_planets_df)), lw = 2, alpha = 0.7, zorder = 10)
                    ax.set_ylim(ymin2, ymax2)

        
            axes = axes+axes3
            
            
        subplot +=1

        if APER:
            subplot+=1
            time, flux, err = pd.read_csv(glob.glob(outdir+'*APER*.csv'))
            
            binned_time, binned_flux, binned_err = bin_data_with_diff_cadences_many_args( time, flux = flux, err = err)
#             print('binned_times', binned_time)
            
            indexes_split = breaking_up_data(time)   
            binned_indexes_split = breaking_up_data(binned_time)   <ma
    
            split_times, split_fluxes, split_err, split_raw, split_rerr = sort_arrays_by_index(indexes_split, time, flux, err, raw, raw_err)
            binned_split_times, binned_split_fluxes, binned_split_err, binned_split_raw, binned_split_rerr = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err)
            axes_n = creating_broken_axes_plots_for_DV_report_min_plot(time, flux, err,binned_time, binned_flux, binned_err, gs0, subplot, ratios)
            for ax in axes_n:
                ax.set_ylim(ymin2, ymax2)
                min_vals, max_vals = ax.get_xlim()
                split_time = times_ot[(times_ot>min_vals) & (times_ot<max_vals)]
                cad = np.min(np.diff(split_time))
            axes = axes+axes_n

        if eleanor:
            subplot+=1

            time, flux, err = pd.read_csv(glob.glob(outdir+'*eleanor*.csv'))
            
            binned_time, binned_flux, binned_err = bin_data_with_diff_cadences_many_args( time, flux = flux, err = err)
#             print('binned_time', binned_time)
            
            indexes_split = breaking_up_data(time)   
            binned_indexes_split = breaking_up_data(binned_time)   
    
            split_times, split_fluxes, split_err, split_raw, split_rerr = sort_arrays_by_index(indexes_split, time, flux, err, raw, raw_err)
            binned_split_times, binned_split_fluxes, binned_split_err, binned_split_raw, binned_split_rerr = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err)
            axes_n = creating_broken_axes_plots_for_DV_report_min_plot(time, flux, err,binned_time, binned_flux, binned_err, gs0, subplot, ratios)
            for ax in axes_n:
                ax.set_ylim(ymin2, ymax2)
                min_vals, max_vals = ax.get_xlim()
                split_time = times_ot[(times_ot>min_vals) & (times_ot<max_vals)]
                cad = np.min(np.diff(split_time))
            axes = axes+axes_n

        if len(other_pipelines)>0:
            for pip in other_pipelines:
                subplot+=1
                time, flux, err = pd.read_csv(glob.glob(outdir+'*'+pip+'*.csv'))
            
                binned_time, binned_flux, binned_err = bin_data_with_diff_cadences_many_args( time, flux = flux, err = err)
#                 print('binned_time', binned_time)

                indexes_split = breaking_up_data(time)   
                binned_indexes_split = breaking_up_data(binned_time)   

                split_times, split_fluxes, split_err, split_raw, split_rerr = sort_arrays_by_index(indexes_split, time, flux, err, raw, raw_err)
                binned_split_times, binned_split_fluxes, binned_split_err, binned_split_raw, binned_split_rerr = sort_arrays_by_index(binned_indexes_split, binned_time, binned_flux, binned_err)
                axes_n = creating_broken_axes_plots_for_DV_report_min_plot(time, flux, err,binned_time, binned_flux, binned_err, gs0, subplot, ratios)
                for ax in axes_n:
                    ax.set_ylim(ymin2, ymax2)
                    min_vals, max_vals = ax.get_xlim()
                    split_time = times_ot[(times_ot>min_vals) & (times_ot<max_vals)]
                    cad = np.min(np.diff(split_time))
                axes = axes+axes_n
    

            

        for ax in axes:
            min_vals, max_vals = ax.get_xlim()
            ymin_, ymax_ = ax.get_ylim()
            delta_y = abs(ymax_ - ymin_)
            for indx, planet in planet_df.iterrows():
                epochs = find_t0_vals_within_time(min_vals, max_vals, planet['T0'], planet['Period'])
#                 print('planet num', planet['Planet_Num'])
                if planet['Ptype']=='Single':
                    ax.scatter(epochs, np.full(len(epochs) ,ymin_ + 0.1*delta_y), marker='^', color = 'C'+str(indx), s=50, zorder = 1000)
                else:
                    ax.scatter(epochs, np.full(len(epochs) ,ymin_ + 0.05*delta_y), marker='^', color = 'C'+str(indx), facecolors='none', s=30, zorder = 5000)

                    
                    
                    
        ax_fin = plt.subplot(gs1[:,-1]) #for the last subplot, print text
        txtstr = 'TICID='+ str(ticid)                              +'\n'\
            +'RA='   + str(round(float(catalog_df.RA), 8))                   +'\n'\
            +'DEC='  + str(round(float(catalog_df.DEC), 8))            +'\n'\
            +'R_*='  + str(round(float(catalog_df.Rad), 5))         +'[R_s]'   +'\n'\
            +'M_*='  + str(round(float(catalog_df.Mass), 5))        +'[M_s]'   +'\n'\
            +'Teff=' + str(round(float(catalog_df.Teff), 2))     +'[K]'     +'\n'\
            +'Tmag=' + str(round(float(catalog_df.Tmag), 3))                   +'\n'\
            +'Vmag=' + str(round(float(catalog_df.Vmag), 3))                   +'\n'\
            +'Jmag=' + str(round(float(catalog_df.Jmag), 3))                   +'\n'\
            +'Cont=' + str(round(float(catalog_df.ContRatio), 3))                   +'\n'\
            +'----- Planet Parmas -----'                              +'\n'\
        
        for indx, planet in planet_df.iterrows():
            txtstr = txtstr + '--' +'Planet Num='+ str(int(planet.Planet_Num))+'--' +'\n'\
            +'Planet Type='+str(planet.Ptype) +'\n'\
            +'R_p='  + '{:3.5}'.format(str(planet.Rad_p*float(catalog_df.Rad)*109.122))    +'[R_e]'   +'\n'\
            +'t0='   + '{:4.9}'.format(str(planet.T0))       +'[TJD]'   +'\n'\
            +'depth='+ '{:1.6}'.format(str(planet.Depth))               +'\n'\
            +'T='    + '{:2.5}'.format(str(planet.Dur))      +'[h]'     +'\n'\
            +'P_c='  + '{:5.6}'.format(str(planet.Period))      +'[d]'     +'\n'


        if len(planet_df)==0:
            txtstr = txtstr + '\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'+'\n'
#         plt.axis([0,1,0,1])
        ax_fin.text(0.05, 0.98, txtstr, transform=ax_fin.transAxes, 
        verticalalignment='top', horizontalalignment='left', fontsize = 10)
#         plt.text(0., 0., txtstr,fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        
        pdf.savefig()
        plt.clf()
        plt.close('all')
        

       