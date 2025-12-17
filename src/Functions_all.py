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

import numpy        as np
import pandas       as pd
import time         as tm 
import lightkurve   as lk
import deep_transit as dt

# import mr_forecast as mr

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


# import eleanor

import warnings
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))


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


def remove_outliers(time, flux, sigma_lower=5.0, sigma_upper=2.5, **kwargs):
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
#             ax.text(
#                 new_start - real_mask[3],
#                 real_mask[2] + real_mask[4]*3 / 4 -0.03,
#                 s='depth: '+f"{real_mask[4]:.4f}",
#                 color='teal',
#                 verticalalignment="top",
#                 bbox=dict(alpha=0.75, color='None'),
#                 clip_on=True, 
#                 fontsize = 12, 
#             )mo

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

    
    
def breaking_up_data(time, break_val = 27.):
    time = np.array(time)
    brk = np.append(np.append([0], find_breaks(time, break_val)), [len(time)])

    indexes = []
    for i in range(len(brk)-1):
        r = np.arange(brk[i],brk[i+1], 1)

        if len(r)>1:
            num_pts_in_6hrs = np.nanmin(np.diff(time[r]))/24/4
        
            if len(time[r])>num_pts_in_6hrs:
                indexes.append(r)
    return indexes

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



def order_of_magnitude(number):
  """Calculates the order of magnitude of a number.

  Args:
    number: The number to calculate the order of magnitude of.

  Returns:
    The order of magnitude of the number.
  """
  if number == 0:
    return 0
  return math.floor(math.log10(abs(number)))


def checking_last_BLS_power_for_artificial_inflation(power_results):
    if max(power_results) == power_results[-1]:
    
        max_indx = 1
        rev_power_results = power_results[::-1]
        for pwr in rev_power_results:
            if pwr == power_results[-1]:
                max_indx+=1
            else:
                break
        return np.arange(len(power_results)-max_indx)
    else:
        return np.arange(len(power_results))
        


def catching_periods_repeated_and_offset(per, t0, per_array, t0_array, rep_indxs):
    
    new_periods = []
    
    diff_tc = np.abs(np.array(t0_array)[rep_indxs] - t0)
#     print('diff_tc', diff_tc)
    for iii in range(len(diff_tc)):
        n = np.ceil(diff_tc[iii]/per)
#         print('what is n?', n, 'what is t0?', t0, t0_array)
        min_diff_tc = np.nanmin(np.abs(diff_tc[iii] - np.array([n-1, n, n+1])*per))
#         print('minimum difference between vals', min_diff_tc, 'period is', per)
        if round(min_diff_tc, 1) !=0:
            frac_per = per/min_diff_tc
            new_periods.append(min_diff_tc)
        else: 
            new_periods.append(per)
    if len(new_periods)>0:
        nper, indxs = np.unique(np.round(new_periods, 1), return_index=True)
        new_periods = list(np.array(new_periods)[indxs])
#     if len(new_periods)>1:
#     print('new periods ', new_periods)
    return new_periods
                        


# def check_float_to_int_diff(val, threshold):
#     if

def checking_BLS_periodicity(per, period_array, t0, t0_array):
    factors = np.array(period_array)/per
    print('factors',np.round(factors, 5))
    factor_indxs = np.where(np.logical_or(np.abs(factors - np.rint(factors))<0.03, np.abs(1/factors - np.rint(1/np.array(factors)))<0.03))[0]

    keep_factor = 1
    if 1. in np.round(factors, 7):
        keep_factor = -1E5
        print('its 1')

        return per, keep_factor
    rep_indxs = np.where(np.rint(factors[factor_indxs])==1.)[0]     
#     print('indexes of repeated periods', rep_indxs)
    new_period = per
    keep_factor = 1
#     print('keep factor 1', keep_factor)

    if len(rep_indxs)>0:
        keep_factor = -1
        val = 0
        while (1. in np.round(period_array/new_period, 1) and val<len(period_array)+1):
            val+=1                    
            new_period = catching_periods_repeated_and_offset(new_period,t0, np.array(period_array), np.array(t0_array), rep_indxs)
            if len(new_period)>0:
                new_period = new_period[0]
            elif new_period == per:
                keep_factor = -1E5
                

        if val>len(period_array):
            keep_factor = -2

    elif len(np.where(np.rint(factors[factor_indxs])!=1.)[0])>0:
        
        factors_not1 = factors[np.where(np.rint(factors[factor_indxs])!=1.)]
        factors_not1[factors_not1>1] = 1.

        factors_not1 = np.unique(factors_not1)
        
        keep_factor = max(1/factors_not1)
        
        #note: the following line is assuming that generally the min period is the true period, and multiples are aliases. This allows us to run MCMC on fewer periods. However, if the true period is the longer one, we're in trouble
        new_period  = per
    
    print('final of periodicity check; period: ', new_period, ' keep factor', keep_factor)
    return new_period, keep_factor
        

        

    
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



def using_BLS_to_find_periodic_signals(time, flux, intransit, verbose = True, show_progress_info = True, save_phaseFold = True, periods = [], T0 = [], Tdur = [], depths = [], first=False):
    
    if first == True:
        periods = []
        T0 = []
        Tdur = []
        depths = []

    time_new = time[~intransit]
    flux_new = flux[~intransit]
#     print('time length', len(time_new), 'time difference', max(time_new)-min(time_new))
    df = 1E-4
    
#     print('frequency factor will be: ', np.max([1., 10**(freq_fact_exp-1)/2]))
    
    if len(time_new)>0:
        durations = np.linspace(0.01, 0.5, 50)

        freq_fact_prelim = df/min(durations)*(np.nanmax(time_new)-np.nanmin(time_new))**2
        freq_fact_exp = np.ceil(np.log10(freq_fact_prelim))    

        start = tm.time()

        model     = BoxLeastSquares(time_new, flux_new)
        max_per   = np.min([100., (max(time_new)-min(time_new))*3/5])
        results   = model.autopower(durations, frequency_factor = np.max([10, (10**(freq_fact_exp-1))/2]), maximum_period=max_per)#, objective='snr', )
        
        end = tm.time()    
                
        my_median = running_median(results.power, kernel = min((25, int(len(time_new)/10))))
        results['power_final'] = results.power - my_median
        
        check_pwr_final_indxs = checking_last_BLS_power_for_artificial_inflation(results['power_final'])
        index = np.argmax(results.power_final[check_pwr_final_indxs])

        period = results.period[index]
        t0 = results.transit_time[index]
        duration = results.duration[index]
        depth = results.depth[index]
        
        print('BLS period', period, ' transit duration', duration*24, ' hours; ', 'duration in days: ', duration)
                
        print('planet run: BLS time ', (end-start)/60, 'minutes')
        
        if verbose: # and np.ceil(results.duration[index]/np.nanmedian(np.diff(time)))>=3:
            plt.figure(figsize = (10, 6))
            val_triangles = min(results.power_final)-np.std(results.power_final)
            ax = plt.gca()
            ax.scatter(period, val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

            plt.xlim(np.min(results.period), np.max(results.period))
            for n in range(2, 10):
                ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
                ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
            plt.ylabel(r'SDE')#, fontsize = 40)
            plt.xlabel('Period (days)')#, fontsize = 40)
        
            ax.plot(results.period, results.power_final, color = 'k', lw=0.65)
            
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

                ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
    #             ax2.set_xlim(-0.5, 0.5)
                ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
                ax2.set_ylabel('Relative Flux')#, color = 'k', fontsize = 40);
                plt.show()
                
#             print('T0: ', results.transit_time[index], 'duration: ', results.duration[index], 'npoints_dur: ', np.ceil(results.duration[index]/np.nanmedian(np.diff(time))))

        sorted_results = np.sort(results.power_final)
#         stdv = np.nanstd(np.diff(results.power_final))

        stdv = np.nanstd(sorted_results[:-5])
        first_5_num_std = np.array([np.abs(np.diff(sorted_results[[-1, -1*x]]))[0]/stdv for x in range(2, 7)])

        print('first 5 num stds', first_5_num_std)
        
        if period>50 and (not np.any(first_5_num_std>2.75)):
            for key in results.keys():
                if key!='objective' and key!='period':
                    results[key] = results[key][results.period<50]
            results.period = results.period[results.period<50]
            index = np.argmax(results.power_final)

            period = results.period[index]
            print('new period < 50', period)
            t0 = results.transit_time[index]
            duration = results.duration[index]
            depth = results.depth[index]


            sorted_results = np.sort(results.power_final)
#             stdv = np.nanstd(np.diff(results.power_final))
            stdv = np.nanstd( sorted_results[:-5])

            first_5_num_std = np.array([np.abs(np.diff(sorted_results[[-1, -1*x]]))[0]/stdv for x in range(2, 7)])

            print('first 5 num stds - short periods', first_5_num_std)

        rule_1 = check_rules_to_continue_BLS(results, index)
                
    if (not rule_1 and (not (np.any(first_5_num_std>5) and first_5_num_std[0]>0.7))) or len(time_new)==0 :
        print(f'reason: not rule_1 {not rule_1}, no good std {not np.any(first_5_num_std>5)}, len time == 0 {len(time_new) == 0}') 

#             print('checking rules', rule_1, len(time_new))
        print('done with BLS')
        print('per', periods, 'kept:', len(periods)>0)

        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    elif np.any(first_5_num_std>3.) and first_5_num_std[0]>0.7:
        try:
            stats = model.compute_stats(period, duration, t0)
            print('current t0', t0)
#             t0_new = stats["transit_times"][np.where(stats['per_transit_log_likelihood']==max(stats["per_transit_log_likelihood"]))[0]][-1]
#             print('new t0: ', t0_new)s
#             t0=t0_new
            print('number transit times in baseline:', len(stats["transit_times"][stats["per_transit_count"]>0]), ' \nnumber of point in each transit: ', stats["per_transit_count"][stats["per_transit_count"]>0], '\ntransit likelihood:', stats["per_transit_log_likelihood"][stats["per_transit_count"]>0])
            
            transit_times_all = stats["transit_times"][np.where(stats["per_transit_count"]>0)[0]]
#             for zzz in range(len(transit_times_all)):
#                 t0_x =transit_times_all[zzz]
#                 fig,ax  = plt.subplots(figsize=(5,5))
                
# #                 print(time_new)
#                 good_locations = np.where(np.logical_and(t0_x-duration-1/4<time_new, t0_x+duration+1/4>time_new))
                
# #                 print('good locations', good_locations)
#                 ax.scatter(time_new[good_locations], flux_new[good_locations], color = 'C'+str(zzz))
#                 miny, maxy = ax.get_ylim()
#                 ax.vlines(t0_x, miny, maxy,)
#                 ax.set_ylim(miny, maxy)
#                 plt.show()
            array_targets = np.array(stats["per_transit_count"])

#             bad_transits = stats["transit_times"][np.where(stats["per_transit_log_likelihood"]<-5E-5)[0]]
#             if len(bad_transits)>0 and len(stats["transit_times"][stats["per_transit_count"]>0])<10:
#                 for t0 in bad_transits:
#                     print('bad transit', t0)
#                     intransit = np.logical_or(intransit, transit_mask(time,  period*7, duration+1/3, t0))
#                 return using_BLS_to_find_periodic_signals(time, flux, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0,  Tdur = Tdur, depths = depths, first = False)
                 

            if len(np.where(array_targets>0)[0])<2:
                print('BLS marked this as a single transit - I dont know why, but Im dropping it')


                intransit = np.logical_or(intransit, transit_mask(time,  period, (duration)+1/6, t0))

            
                return using_BLS_to_find_periodic_signals(time, flux, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0,  Tdur = Tdur, depths = depths, first = False)
            else:
                print('checking for odd/even binaries')
                t0, depth, period = checking_BLS_odd_even_binaries(stats, t0, period, depth)


        except ValueError as err:
            print('getting error: ', err)


        if len(periods)>0:# and len(factor_indxs)>0:

            new_period, keep_factor = checking_BLS_periodicity(period, periods, t0, T0)

            if keep_factor>0:

                intransit = np.logical_or(intransit, transit_mask(time, new_period, duration+1/6, t0))
            elif keep_factor < -50 :
                intransit = np.logical_or(intransit, transit_mask(time, new_period/2, duration+1/6, t0))
            else:
                intransit = np.logical_or(intransit, transit_mask(time, new_period, duration+1/3, t0))

            if new_period>0.5 and keep_factor>0:

                print('kept period 1')
                depths.append(results.depth[index])
                periods.append(new_period)
                # T0.append(last_t0)
                T0.append(t0)
                Tdur.append(duration)
        else:
            print('kept this period', period)

#                 intransit = np.logical_or(intransit, transit_mask(time, results.period[index], (2*results.duration[index]/24)+(1/6), results.transit_time[index]))
            intransit = np.logical_or(intransit, transit_mask(time, period, duration+(1/6), t0))

            depths.append(1-depth)
            periods.append(period)
            T0.append(t0)
            Tdur.append(duration)
        return using_BLS_to_find_periodic_signals(time, flux, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0,  Tdur = Tdur, depths = depths, first = False)



    else: 
        print('got signal, but probably not periodic; blocking intransit and searching again')
        intransit = np.logical_or(intransit, transit_mask(time, results.period[index], duration+(1/6), results.transit_time[index]))


        return using_BLS_to_find_periodic_signals(time, flux, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0,  Tdur = Tdur, depths = depths, first = False)



# def using_BLS_to_find_periodic_signals(time, flux, verbose = True, show_progress_info = True, save_phaseFold = True, intransit = [], periods = [], T0 = [], Tdur = [], depths = [], first=True):
    
#     if first == True:
#         intransit = np.full(len(time), False)
#         periods = []
#         T0 = []
#         Tdur = []
#         depths = []

#     time_new = time[~intransit]
#     flux_new = flux[~intransit]
    
#     if len(time)>500:
#         mag_factor = 1
#     else:
#         mag_factor = 10**order_of_magnitude(len(time))
#     if len(time_new)>0:
#         start = tm.time()

#         durations = np.linspace(0.01, 0.5, 50)
#         model     = BoxLeastSquares(time_new, flux_new)
#         max_per = np.min([100., (max(time_new)-min(time_new))*2/3])
        
#         results   = model.autopower(durations, frequency_factor = 100/mag_factor, maximum_period=max_per)#, objective='snr', )
        
#         if np.all(np.abs(results.power) == np.inf): 
#             results.power = np.random.normal(0, 1e-5, len(results.power))
#         print('time len: ', len(time))
#         my_median = running_median(results.power, kernel = min((25, int(len(time_new)/10))))
#         results['power_final'] = results.power - my_median
# #         print('checking median', my_median, set(my_median))
# #         print('checking results.power', results.power)
        
        
        
#         index = np.argmax(results.power_final)
#         period = results.period[index]
                   
#         end = tm.time()


#         t0 = results.transit_time[index]
#         duration = results.duration[index]
#         depth = results.depth[index]
        
#         print('BLS period', period, ' transit duration', duration*24, ' hours')
                
#         print('planet run: BLS time ', (end-start)/60, 'minutes, num of points intransit', len(np.where(intransit)[0]))

#         if verbose: # and np.ceil(results.duration[index]/np.nanmedian(np.diff(time)))>=3:
#             plt.figure(figsize = (10, 6))
#             val_triangles = min(results.power_final)-np.std(results.power_final)
#             ax = plt.gca()
#             ax.scatter(period, val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

#             plt.xlim(np.min(results.period), np.max(results.period))
#             for n in range(2, 10):
#                 ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
#                 ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
#             plt.ylabel(r'SDE')#, fontsize = 40)
#             plt.xlabel('Period (days)')#, fontsize = 40)
        
#             ax.plot(results.period, results.power_final, color = 'k', lw=0.65)
            
#             plt.show()
#             plt.close()
#             if duration<period:
#                 plt.figure(figsize = (5, 5))
#                 ax2 = plt.gca()

#                 x = ((time - t0 + 0.5*period) % period) -( 0.5*period)
#                 m = np.abs(x) < 0.5
#                 ax2.scatter(
#                     x[m],
#                     flux[m],
#                     color='k',
#                     s=5,
#                     alpha=0.8,
#                     zorder=10)

#                 x_new = np.linspace(-0.5, 0.5, 1000)

#                 f = model.model(x_new + t0, period, duration, t0)

#                 ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
#     #             ax2.set_xlim(-0.5, 0.5)
#                 ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
#                 ax2.set_ylabel('Relative Flux')#, color = 'k', fontsize = 40);
#                 plt.show()
# #             print('T0: ', results.transit_time[index], 'duration: ', results.duration[index], 'npoints_dur: ', np.ceil(results.duration[index]/np.nanmedian(np.diff(time))))


#         sorted_results = np.sort(np.unique(results.power_final))
#         stdv = np.nanstd(np.diff(results.power_final))
        
        
#         first_5_num_std = np.array([np.abs(np.diff(sorted_results[[-1, -1*x]]))[0]/stdv for x in range(2, 7)])

#         print('first 5 num stds', first_5_num_std)
#         if np.any(first_5_num_std>3):
#             try:
#                 stats = model.compute_stats(period, duration, t0)
#                 if stats['depth'][1] > 10*stats['depth'][0]:

#                     if stats['depth_odd'][0]/stats['depth_even'][0] > 10:
#                         print('keeping odd - would rather keep binary as 2 objects than miss a planet from period alias')
#                         t0 = t0+period
#                         period = 2*period
#                         depth = stats['depth_odd'][0]

#                     elif stats['depth_even'][0]/stats['depth_odd'][0] > 10:
#                         print('keeping even - would rather keep binary as 2 objects than miss a planet from period alias')
#                         t0 = t0
#                         period = 2*period
#                         depth = stats['depth'][0]

# #                     if period != period_:
# #                         print(f'we got here, and these are the old params: per={period}, t0={t0} and depth={depth}; and the new ones: per={period_}, t0={t0_} and depth={depth_}. The duration is {results.duration[index]}')

# #                         intransit = np.logical_or(intransit, transit_mask(time, period_, (2*results.duration[index]/24)+(1/4), t0_))

# #                         depths.append(1-depth_)
# #                         periods.append(period_)
# #                         # T0.append(last_t0)
# #                         T0.append(t0_)
# #                         Tdur.append(results.duration[index])
# #                         return using_BLS_to_find_periodic_signals(time, flux, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0, Tdur = Tdur, depths = depths, first = False)

#             except ValueError as err:
#                 print('getting error: ', err)

        
        

#             if len(periods)>0:# and len(factor_indxs)>0:
            
# #             periods = np.array(periods)
#                 factors = np.array(periods)/period
#                 factor_indxs = np.where(np.logical_or(np.abs(factors - np.rint(factors))<0.03, np.abs(1/factors - np.rint(1/np.array(factors)))<0.03))[0]

#                 rep_indxs = np.where(np.rint(factors[factor_indxs])==1.)[0]     
#                 if len(rep_indxs)>0:
#                     new_period = period
#                     val = 0
#                     while (1. in np.round(periods/new_period, 1) and val<len(periods)+1):
#                         val+=1                    
#                         new_period = catching_periods_repeated_and_offset(new_period,t0, np.array(periods), np.array(T0))
#                         if len(new_period)>0:
#                             new_period = new_period[0]

#                     print('new period', new_period, type(new_period))
#                     if val>len(periods) and type(new_period) == np.float64:
#                         print(' we went 1')
#                         intransit = np.logical_or(intransit, transit_mask(time, new_period, (2*results.duration[index]/24)+(1/3), results.transit_time[index]))

#                     elif val<=len(periods) and type(new_period) == np.float64:
#                         print(' we went 2')

#                         intransit = np.logical_or(intransit, transit_mask(time, new_period, (2*results.duration[index]/24)+(1/6), results.transit_time[index]))
#                         if new_period>0.5:
#     #                         print(f'while the new period is {np.array(periods)[rep_indxs]}, there appears to be an offset in these transits - lets keep the difference = {min_diff_tc}  which is {frac_per} of the prevous period {period} and know this could be a binary. ')

#                             depths.append(1-results.depth[index])
#                             periods.append(new_period)
#                             # T0.append(last_t0)
#                             T0.append(results.transit_time[index])
#                             Tdur.append(results.duration[index])

#                 elif len(np.where(np.rint(factors[factor_indxs])!=1.)[0])>0:
#                     keep_factor = np.rint(factors[factor_indxs])
#                     keep_factor = keep_factor[keep_factor != 1.]

#                     keep_factor[keep_factor<1] = 1/keep_factor[keep_factor<1]
#                     keep_factor = np.unique(keep_factor)
#                     print('here are the fractions of the true period, ', keep_factor)

#                     new_period = period

#                     intransit = np.logical_or(intransit, transit_mask(time, new_period, (2*results.duration[index]/24)+(1/4), results.transit_time[index]))

#                     if new_period>0.5:
#                         print('this is as repeat, but we probably found the wrong alias first! going to keep period: ', new_period, ' == period/', keep_factor, 'as well as old period')

#                         depths.append(1-results.depth[index])
#                         periods.append(new_period)
#                         # T0.append(last_t0)
#                         T0.append(results.transit_time[index])
#                         Tdur.append(results.duration[index])

#                 return using_BLS_to_find_periodic_signals(time, flux, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0, Tdur = Tdur, depths = depths, first = False)


#         rule_1 = np.abs(np.diff(sorted_results[[-1, -2]])) > 2.*stdv
# #         print('rule 1', rule_1)
 
#         pwr_copy = np.array(results.power_final).copy()
#         pwr_copy[index] = -np.inf

#         # Get the index of the maximum value in the modified array

#         period_2 = results.period[np.argmax(pwr_copy)]

#         if (not rule_1) and np.isin(np.round(max([period, period_2])/min([period, period_2]),1), [2.0, 0.5]):
#             print('double checking rule 1')
                        
#             rule_1 = np.abs(np.diff(sorted_results[[-1, -3]])) > 2.*stdv
#             pwr_copy = np.array(results.power_final).copy()
#             pwr_copy[np.argmax(pwr_copy)] = -np.inf

#         # Get the index of the maximum value in the modified array

#             period_3 = results.period[np.argmax(pwr_copy)]

#             if (not rule_1) and (round(max([period, period_3])/min([period, period_3]),1) in [2.0, 0.5]):
#                 print('double checking rule 2')
#                 stdv1 = np.nanstd(np.sort(np.diff(results.power_final))[:-3])
                
#                 print('standard deviations', stdv, stdv1)
#                 print('number stdev',np.abs(np.diff(sorted_results[[-1, -4]]))[0]/stdv1)
                
                
#                 rule_1 = np.abs(np.diff(sorted_results[[-1, -4]])) > 2.25*stdv1
                
                
#         if not rule_1 or len(time_new)==0 :

#             print('checking rules', rule_1, len(time_new))
#             print('done with BLS')
#             print('per', periods, 'kept:', len(periods)>0)

#             return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit
#         else: 
#             print('got here')
#             intransit = np.logical_or(intransit, transit_mask(time, results.period[index], (2*results.duration[index]/24)+(1/6), results.transit_time[index]))
            
            
#             depths.append(1-depth)
#             periods.append(period)
#             T0.append(t0)
#             Tdur.append(duration)
#             return using_BLS_to_find_periodic_signals(time, flux, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0,  Tdur = Tdur, depths = depths, first = False)
#     else:
#         print('done with BLS')
#         print('per', periods, 'kept:', len(periods)>0)

#         return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit




def using_TLS_to_find_periodic_signals(time, flux, u, verbose = False, show_progress_info = True, save_phaseFold = True,
                                       intransit = [], periods = [], T0 = [], Tdur = [], 
                                       depths = [], first=True):
    
    time_diff = max(time)-min(time)
    print('time diff', time_diff)
    max_per = min(time_diff, 100.)
    if first == True:
        intransit = np.full(len(time), False)
        periods = []
        T0 = []
        Tdur = []
        depths = []

    time_new = np.array(time)[~intransit]
    flux_new = np.array(flux)[~intransit]
    if len(time_new)>0:
        start = tm.time()

        durations = np.linspace(0.02, 0.5, 75)
        
        model = transitleastsquares(time_new, flux_new)
        results = model.power(
            period_min=0,
            period_max=max_per,
#             transit_depth_min=ppm*10**-6,
#             oversampling_factor=10,
#             duration_grid_step=1.02,
            u=ab,
            limb_dark='quadratic',
#             M_star = 1,
#             M_star_max=1.1
            n_transits_min = 1,
            show_progress_info = show_progress_info
            )

                
        index = np.argmax(results.power)
        
        period    = results.period
        val_triangles = min(results.power)-np.std(results.power)

#         print('period', period, 'index ', index)
        end = tm.time()
        if round(results.T0, 4) in [round(x, 4) for x in T0]:
        
            intransit = np.logical_or(intransit, transit_mask(time, results.period, (2*results.duration/24)+(1/6), results.T0))
            print('FOUND THE SAME PLANET: CONTINUING')
            return using_TLS_to_find_periodic_signals(time, flux, u, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0, Tdur = Tdur, depths = depths, first = False)
        if verbose:
            

#             print('plot 1')
            plt.figure(figsize = (10,6))

            ax = plt.gca()
            ax.set_facecolor('None')
            ax.scatter(time_new,flux_new, color = 'k', zorder = 10, marker = '.')
            
            faux_intransit = np.logical_or(intransit, transit_mask(time, results.period, (2*results.duration/24)+(1/6), results.T0))
            
            ax.scatter(np.array(time)[~faux_intransit],np.array(flux)[~faux_intransit], 
                       color = 'r', zorder = 11, marker = '.', alpha = 0.3)

            plt.ylabel(r'N. Flux')#, fontsize = 40)
            plt.xlabel('Time', fontsize = 40)

            plt.figure(figsize = (5, 5))

            ax = plt.gca()
            ax.set_facecolor('None')
            ax.scatter(period,val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

            plt.xlim(np.min(results.periods), np.max(results.periods))
            for n in range(2, 10):
                ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
                ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)


            plt.ylabel(r'SDE')#, fontsize = 40)
            plt.xlabel('Period (days)')#, fontsize = 40)
        
        
            ax.plot(results.periods, results.power, color = 'k', lw=1)
            ax.xaxis.label.set_color('k')        #setting up X-axis label color to yellow
            ax.yaxis.label.set_color('k')          #setting up Y-axis label color to blue


            t0 = results.T0
            duration = results.duration
            
            plt.show()
            plt.close()
            
            plt.figure(figsize = (5, 5))
            ax2 = plt.gca()

            ax2.set_facecolor('None')

            x = ((time - t0 + 0.5*period) % period) -( 0.5*period)
            m = np.abs(x) < 0.5
            ax2.scatter(
                x[m],
                np.array(flux)[m],
                color='gray',
                s=5,
                alpha=0.8,
                zorder=2)

            x_new = np.linspace(-0.5, 0.5, 1000)
            f = model.model(x_new + t0, period, duration, t0)

            ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
            ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
            ax2.set_ylabel('Relative Flux')#, color = '#CC9966', fontsize = 40);
            plt.show()
#             print('T0: ', results.T0, 'duration: ', results.duration, 'npoints_dur: ', np.ceil(results.duration/30.))
            
            
        if not np.abs(np.diff(np.array(sorted(results.power))[[-1, -4]]))>2*np.nanstd(results.power) or len(time_new)==0 :
            print('FOUND NO PLANET: FINISHING THIS LC')
            return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit
        
        
        else: 
            intransit = np.logical_or(intransit, transit_mask(time, results.period, (2*results.duration/24)+(1/6), results.T0[0]))
            
            depths.append(results.depth)
            periods.append(results.period)
            T0.append(results.T0)
            Tdur.append(results.duration)
            print('FOUND A PLANET: CONTINUING')
            return using_TLS_to_find_periodic_signals(time, flux, u, intransit=intransit, verbose = verbose,  periods = periods, T0 = T0, Tdur = Tdur, depths = depths, first = False)


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

def using_GERBLS_recursive(time, flux, flux_err = None, intransit=None,
                            verbose=True, plot=True, max_planets=5,
                            min_SDE=4, min_delta_BIC=50, use_AIC=False,
                            periods=None, T0=None, Tdur=None, depths=None, first=False):
    """
    Recursive multi-planet search using GERBLS pyFastBLS + run_double and BIC/AIC-based model selection.
    """
    start = tm.time()
    print('start')
    if intransit is None:
        intransit = np.zeros_like(time, dtype=bool)
    if flux_err is None:
        flux_err = np.std(flux) * np.ones_like(flux)

    if first:
        periods, T0, Tdur, depths = [], [], [], []

    # Mask in-transit points
    time_new, flux_new, flux_err_new = time[~intransit], flux[~intransit], flux_err[~intransit]
    if len(time_new) < 10:
        print("Stopping: insufficient data.")
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    # Prepare data for GERBLS
    data = gerbls.pyDataContainer()
    data.store(time_new, flux_new, flux_err_new)
    min_period =  0.5
    max_period = np.min([50., (max(time_new)-min(time_new))*2/3])
    t_samp = np.median(np.diff(time_new))
    print('t_samp', t_samp)#, 'length time', len(time_new), 'time', time_new, 'time differences', np.diff(time_new), 'flux differences', np.diff(flux_new))
    
    if t_samp == 0:
        print('Im unhappy')
    t_samp = max([t_samp, 10/60/24])
    
    durations = np.linspace(0.01, 0.5, 50)

    # Initialize and setup BLS object
    bls = gerbls.pyFastBLS()
    bls.setup(data, min_period, max_period,
              t_samp=t_samp,
#               duration_mode='constant',
              durations=durations)  # durations in days

    # Run BLS with double precision
    bls.run_double(verbose = True)
    blsa = gerbls.pyBLSAnalyzer(bls)
    # Extract best candidate
    
    end = tm.time()
    print('end. took: ', abs(start-end)/60, 'minutes')
    
    my_median = running_median(blsa.dchi2, kernel = min((25, int(len(time_new)/15))))

    idx      = np.argmax(blsa.dchi2-my_median)
    
    period   = blsa.P[idx]
    t0       = blsa.t0[idx]
    duration = blsa.dur[idx]
    depth    = blsa.dmag[idx]

    # Compute SDE
    sde = (blsa.dchi2[idx] - np.median(blsa.dchi2)) / np.std(blsa.dchi2)
    model = blsa.generate_models(1)[0]    
    if verbose:
        print(f"Candidate: P={period:.4f} d, SDE={sde:.2f}")

    
    if sde < min_SDE:
        print("Stopping: SDE below threshold.")
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    if plot: # and np.ceil(results.duration[index]/np.nanmedian(np.diff(time)))>=3:
        plt.figure(figsize = (10, 6))
        val_triangles = min(blsa.dchi2)-np.std(blsa.dchi2)
        ax = plt.gca()
        ax.scatter(period, val_triangles, color = 'r', marker = '^', s=20, zorder = 10)
        ax.set_xlim(blsa.P.min(), blsa.P.max())
        for n in range(2, 10):
            ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
            ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
        plt.ylabel(r'SDE')#, fontsize = 40)
        plt.xlabel('Period (days)')#, fontsize = 40)

        ax.plot(blsa.P, blsa.dchi2-my_median, color = 'k', lw=0.65)

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

            f = build_box_model(x_new, 0, duration, depth, period)

            ax2.plot(x_new, f, color='grey', lw = 1, alpha = 0.6, zorder = 5)
#             ax2.set_xlim(-0.5, 0.5)
            ax2.set_xlabel('Phase')#, color = 'k', fontsize = 40)
            ax2.set_ylabel('Relative Flux')#, color = 'k', fontsize = 40);
            plt.show()

    # Build models for likelihood
    model_flux = build_box_model(time_new, t0, duration, depth, period)
    null_flux = np.ones_like(flux_new)

    logL_transit = compute_log_likelihood(flux_new, model_flux, flux_err_new)
    logL_null = compute_log_likelihood(flux_new, null_flux, flux_err_new)

    n, k = len(time_new), 4
    bic_transit = compute_BIC(logL_transit, n, k)
    bic_null = compute_BIC(logL_null, n, 1)
    delta_BIC = bic_null - bic_transit

    if verbose:
        print(f"ΔBIC={delta_BIC:.2f} (threshold={min_delta_BIC})")

    if use_AIC:
        aic_transit = compute_AIC(logL_transit, k)
        print(f"AIC={aic_transit:.2f}")

    if delta_BIC < min_delta_BIC:
        print("Candidate rejected: insufficient BIC improvement - will still mask and try again.")
        # Mask transits using your transit_mask
        intransit = np.logical_or(transit_mask(time, period, duration, t0))

        return using_GERBLS_recursive(time, flux, intransit=intransit,flux_err=flux_err,verbose=verbose, plot=plot, max_planets=max_planets,min_SDE=min_SDE, min_delta_BIC=min_delta_BIC, use_AIC=use_AIC,periods=periods, T0=T0, Tdur=Tdur, depths=depths, first=False)

    # Accept candidate
    periods.append(period)
    T0.append(t0)
    Tdur.append(duration)
    depths.append(depth)
    print('depths', depths)

    if verbose:
        print(f"Accepted planet: P={period:.4f} d")

    # Mask transits using your transit_mask
    intransit  = np.logical_or(intransit, transit_mask(time, period, duration, t0))

    if len(periods) >= max_planets:
        print("Reached max planets.")
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    # Recurse
    return using_GERBLS_recursive(time, flux, intransit=intransit,flux_err=flux_err,verbose=verbose, plot=plot, max_planets=max_planets,min_SDE=min_SDE, min_delta_BIC=min_delta_BIC, use_AIC=use_AIC,periods=periods, T0=T0, Tdur=Tdur, depths=depths, first=False)
        

def using_BLS_recursive(time, flux, flux_err = None, intransit=None,
                            verbose=True, plot=True, max_planets=10,
                            min_SDE=10, min_delta_BIC=75, use_AIC=False,
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
    max_per   = np.min([50., (max(time_new)-min(time_new))*3/5])
    results   = model.autopower(durations, frequency_factor = np.max([10, (10**(freq_fact_exp-1))/2]), maximum_period=max_per)#, objective='snr', )

    end = tm.time()    

    my_median = running_median(results.power, kernel = min((25, int(len(time_new)/10))))
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
    sde  = (results['power_final'][index] - np.median(results['power_final'])) / np.std(results['power_final'])
    sde2 = (sorted_results[-1] - sorted_results[-2] ) / np.std(sorted_results[:-2])

    if verbose:
        print(f"Candidate: P={period:.4f} d, SDE={sde:.2f}, SDE2={sde2:.2f}")

    
    if (sde < min_SDE):# or (sde2 < 0.5):
        print("Stopping: SDE below threshold.")
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    if plot: # and np.ceil(results.duration[index]/np.nanmedian(np.diff(time)))>=3:
        plt.figure(figsize = (10, 6))
        val_triangles = min(results.power_final)-np.std(results.power_final)
        ax = plt.gca()
        ax.scatter(period, val_triangles, color = 'r', marker = '^', s=20, zorder = 10)

        plt.xlim(np.min(results.period), np.max(results.period))
        for n in range(2, 10):
            ax.scatter( n*period,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
            ax.scatter(period / n,val_triangles, color = 'maroon', marker = '^', s=20, zorder = 10, alpha= 0.8)
        plt.ylabel(r'SDE')#, fontsize = 40)
        plt.xlabel('Period (days)')#, fontsize = 40)

        ax.plot(results.period, results.power_final, color = 'k', lw=0.65)

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

    stats = model.compute_stats(period, duration, t0)
    transit_times_all = stats["transit_times"][np.where(stats["per_transit_count"]>0)[0]]
    single = False
    if len(transit_times_all)<2:
        single = True

    repeat = False
    if len(periods)>0:# and len(factor_indxs)>0:

        new_period, keep_factor = checking_BLS_periodicity(period, periods, t0, T0)

        if keep_factor>0:
            repeat=False
            intransit = np.logical_or(intransit, transit_mask(time, new_period, duration, t0))
            period = new_period

        if keep_factor < -50 :

            intransit = np.logical_or(intransit,  transit_mask(time, new_period/2, duration, t0))
        else:
            intransit = np.logical_or(intransit, transit_mask(time, new_period, duration, t0))

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

#     delta_BIC_box = (bic_wide_box + bic_long_box)/2 - bic_transit
    
    if verbose:
        print(f"ΔBIC={delta_BIC:.2f}, (threshold={min_delta_BIC})")

#     if use_AIC:
#         aic_transit = compute_AIC(logL_transit, k)
#         print(f"AIC={aic_transit:.2f}")
        
        
            
            
#     if (delta_BIC < min_delta_BIC) or (single) or (repeat):
    if (single) or (repeat):
        print('masking this single transit or repeat detection and continuing')
#         if delta_BIC < min_delta_BIC and not first:
#             print("Candidate rejected: insufficient BIC improvement.")
#             intransit = np.logical_or(intransit, transit_mask(time, period, duration, t0))

#             return using_BLS_recursive(time, flux, intransit=intransit,flux_err=flux_err,verbose=verbose, plot=plot, max_planets=max_planets,min_SDE=min_SDE, min_delta_BIC=min_delta_BIC, use_AIC=use_AIC,periods=periods, T0=T0, Tdur=Tdur, depths=depths, first=False)

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

        return using_BLS_recursive(time, flux, intransit=intransit,flux_err=flux_err,verbose=verbose, plot=plot, max_planets=max_planets,min_SDE=min_SDE, min_delta_BIC=min_delta_BIC, use_AIC=use_AIC,periods=periods, T0=T0, Tdur=Tdur, depths=depths, first=False)

    # Accept candidate
    periods.append(period)
    T0.append(t0)
    Tdur.append(duration)
    depths.append(depth)
    print('depths all', depths)

    if verbose:
        print(f"Accepted planet: P={period:.4f} d")

    # Mask transits using your transit_mask
    intransit = np.logical_or(intransit, transit_mask(time, period, duration, t0))

    if len(periods) >= max_planets:
        print("Reached max planets.")
        return np.array(periods), np.array(T0), np.array(Tdur), np.array(depths), intransit

    # Recurse
    return using_BLS_recursive(time, flux, intransit=intransit,flux_err=flux_err,verbose=verbose, plot=plot, max_planets=max_planets,min_SDE=min_SDE, min_delta_BIC=min_delta_BIC, use_AIC=use_AIC,periods=periods, T0=T0, Tdur=Tdur, depths=depths, first=False)

    
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
    for iii in range(len(t0s)): 

        params_df = pymc_new_general_function(time, flux, flux_err, t0s[iii], [pers[iii], ab, depths[iii]], 'Periodic')
        if len(params_df)>0:

            T0_, period_, depth_, tdur_, SNR = params_df.loc['t0', 'mean'], params_df.loc['Per', 'mean'], params_df.loc['depth', 'mean'], params_df.loc['dur', 'mean'], params_df.loc['SNR', 'mean']
            print('params df', params_df)

        else:
            T0_, period_, tdur_, depth_, SNR = np.nan, np.nan, np.nan, np.nan, 0

        if not np.isnan(T0_):
            periods.append(period_)
            T0_vals.append(T0_)
            Tdur.append(tdur_)
            depth.append(depth_)
            SNR_vals.append(SNR)

        gc.collect()

        intransit = np.logical_or(intransit, transit_mask(total_time, period_, float((2*tdur_/24)), float(T0_)))

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
#     print('t_samp', np.median(np.diff(time)), 'length time', len(time), 'time', time, 'time differences', np.diff(time), 'flux differences', np.diff(flux))
#     print('checking time unique', len(np.unique(time)), 'checking flux unique', len(np.unique(flux)))

#     time_init      = np.array(total_time_flux_df['TIME'].astype(float))
#     flux_init      = np.array(total_time_flux_df['FLUX'].astype(float))
#     flux_err_init  = np.array(total_time_flux_df['FLUX_ERR'].astype(float))
#     intransit_init = np.full(len(time_init), False)
    

#     indexes_split_unorganize_start = breaking_up_data(time_init, 0.75)   
#     diff_ary = np.array([max(np.array(time_init)[x])-min(np.array(time_init)[x]) for x in indexes_split_unorganize_start])
    
#     all_good_indxs =np.concatenate(list(itertools.compress(indexes_split_unorganize_start, diff_ary>0.25))).ravel()

#     time      = time_init[all_good_indxs]
#     flux      = flux_init[all_good_indxs]
#     flux_err  = flux_err_init[all_good_indxs]
#     intransit = np.full(len(time), False)
#     intransit_init[~all_good_indxs] = True


    print('running search on all data')

    ###running search on whole dataset
    if TLS:
        periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_per = using_TLS_to_find_periodic_signals(time, flux, u = ab, intransit =intransit, verbose = verbose)

    else:

        periods_multis, T0_multis, Tdur_multis, depth_multis, intransit_per = using_BLS_recursive(time, flux, intransit = intransit, verbose = verbose, first=True)
        
#     print('depths', depth_multis)
    
    if len(T0_multis)>0:
        nt0_vals, nperiods, ndepth, nTdur, nSNR_vals, intransit, params_df = fitting_periodic_planets(time, flux, flux_err, periods_multis, T0_multis, depth_multis, ab, intransit_per, verbose, save_phaseFold, data_file = data_file)

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
        indexes_split_unorganize = breaking_up_data(time, 2.)   
        indexes_split = sorted(indexes_split_unorganize, key=lambda x: len(x), reverse=True)


#     print('split indexes lengths: ', [len(x) for x in indexes_split])
    for iii in range(len(indexes_split)):
        if len(indexes_split) == 1:
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


        print('periods to not f again', np.array(periods_multis)[toss_bool])
        
        if len(np.array(periods_multis[~toss_bool]))>0:
        

            nt0_vals_split, nperiods_split, ndepth_split, nTdur_split, nSNR_vals_split, intransit, nparams_df = fitting_periodic_planets(split_time, split_flux, split_flux_err, list(np.array(periods_multis)[~toss_bool]), list(np.array(T0_multis)[~toss_bool]), list(np.array(depth_multis)[~toss_bool]), ab, intransit, verbose, save_phaseFold=False, total_time = time, data_file=data_file)
        
            nparams_df.loc[len(nparams_df)] = [None] * len(nparams_df.columns) 

            params.append(nparams_df)
    
            periods.extend(nperiods_split)
            T0_vals.extend(nt0_vals_split)
            Tdur.extend(nTdur_split)
            depth.extend(ndepth_split)
            SNR_vals.extend(nSNR_vals_split)
            
            

        # intransit[indexes_split[iii]] = np.logical_or(intransit_split, intransit_small_fit)
        print('done with multis ', iii+1, ':', len(indexes_split))

#     intransit_init[all_good_indxs] = intransit
#     intransit_indexes =  np.where(intransit)

#     total_time_flux_df['INTRANS_P'] = ~intransit_init
#     if save_file:
#         total_time_flux_df.to_csv(data_file, index =False)

#         print('split indexes lengths: ', [len(x) for x in indexes_split])
    
    only_per_intransit = np.full(len(time), False)

    for iii in range(len(periods)):
        print('period is', periods[iii])
        new_transit_planet = transit_mask(time, periods[iii], (Tdur[iii]/24)+(1/3), T0_vals[iii])
        print('intransit '+str((iii+1)*7), new_transit_planet, len(np.where(new_transit_planet)[0]), len(new_transit_planet))

        only_per_intransit = np.logical_or(only_per_intransit,new_transit_planet)
        

    intransit_indexes =  np.where(only_per_intransit)
    
#     print('checking differences in intransit arrays')
#     differences = (intransit != only_per_intransit)

#     # Count the number of True values (differences)
#     num_differences = np.sum(differences)
#     print(f'there are {num_differences} between the two')
    
#     final_intransit = np.full(len(time), False)
#     final_intransit[all_good_indxs] = only_per_intransit
    
    return periods, T0_vals, Tdur, depth, only_per_intransit, SNR_vals,intransit_indexes, params


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
        planet_df.loc[len(planet_df.index)] = [int(ticid), planet_name, periods_multi[jjj], T0_multi[jjj], Tdur_multi[jjj],1-depth_multi[jjj], SNR[jjj]]
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


def approximate_common_denominator(float1, float2, precision=10**3):
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
                    if len(rrr[rrr>12.])>0:
                        new_rrr.append(list(rrr[rrr>12.]))
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






def singles_search(ticid, data_total, intransit = [], catalog_df = False, confidence = 0.55,  verbose = True, run_1 = True, data_file = ''):
    
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

    indexes_split_unorganize = breaking_up_data(total_time, 1.5)  
    all_good_indxs = []

    if len(indexes_split_unorganize)>1:
        diff_ary = np.array([max(np.array(total_time)[x])-min(np.array(total_time)[x]) for x in indexes_split_unorganize])
#         for iii in range(len(diff_ary)):
#             if diff_ary[iii]>1:
#                 all_good_indxs.extend(indexes_split_unorganize[iii])
# #         for 
        all_good_indxs =np.concatenate(list(itertools.compress(indexes_split_unorganize, diff_ary>1))).ravel()
    if len(all_good_indxs) == 0: 
        all_good_indxs = list(range(len(total_time)))
        
    print('all good indexes (i.e., itertools result: ', type(all_good_indxs), len(total_time))
    list_1 = list(all_good_indxs)
    list_2 = []
    for index, value in enumerate(total_time):
        list_2.append(index)
    print('check differences in good indexes and indexes: ',  [item for item in list_1 if item not in list_2])
    print('type of time array, ', type(total_time))
    total_time = total_time[all_good_indxs]
    total_flux = total_flux[all_good_indxs]
    total_flux_err =total_flux_err[all_good_indxs]

    ab, smass, smass_min, smass_max, sradius, sradius_min, sradius_max = get_catalog_info(ticid, df = catalog_df)
#     print('val', ab)
    
    bboxes = DT_analysis(total_time, total_flux, total_flux_err, confidence)
#     print('ran bboxes', bboxes)

    print('number singles found', len(bboxes))
    t0_singles, dur_singles, depth_singles = [],[],[]
    if len(bboxes)>0:
        n_events = len(bboxes)
        for j, boxes in enumerate(bboxes):
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

#         if verbose: 
        
# #             indexes = breaking_up_data(total_time, 27)
# #             print(len(indexes))
#             for iii in range(len(indexes)):

#                 new_time = total_time[np.array(indexes[iii])]
#                 new_flux = total_flux[p.array(indexes[iii])]
#                 new_flux_err = total_flux_err[p.array(indexes[iii])]


#                 gc.collect()

            #         print(detrended_lc)

#                 fig = plt.figure(figsize = (24, 8))

#                 ax = fig.add_subplot(111)
#                 ax.xaxis.set_major_locator(ticker.MultipleLocator(25)) 
#                 for spine in ['top', 'right']:
#                     ax.spines[spine].set_visible(False)
#                 ax.tick_params(color='k', labelcolor='k')

#                 plot_lc_with_bboxes(detrended_lc, bboxes, ms=3, marker='.', lw=0, ax = ax)

#                 ax.set_ylim(1-3.5*np.std(new_flux), 1+3.5*np.std(new_flux))
#                 plt.show()

#     if run_1:
#         init_periodic_singles = []
#     else:
#         init_periodic_singles = check_if_singles_are_periodic(t0_singles)


    t0_singles = np.array(t0_singles)
    dur_singles = np.array(dur_singles)
    depth_singles = np.array(depth_singles)

#     if run_1:: 
#         t0_singles = np.array(t0_singles)
#         dur_singles = np.array(dur_singles)
#         depth_singles = np.array(depth_singles)
#     else: 
#         t0s_ = []
#         dur_singles_
        
#         for iii in range(len(t0_singles)):
#             model_flux = build_box_model(total_time, duration, t0)
#             model_long_box  = build_box_model(total_time, t0, duration/2, depth*1.5)
#             model_wide_box  = build_box_model(total_time, period, t0, duration*2, depth/1.5)

#             null_flux = np.ones_like(total_flux)

#             logL_transit = compute_log_likelihood(total_flux, model_flux, total_flux_err)
#             logL_long_box = compute_log_likelihood(total_flux, model_long_box, total_flux_err)
#             logL_wide_box = compute_log_likelihood(total_flux, model_wide_box, total_flux_err)

#             logL_null = compute_log_likelihood(total_flux, null_flux, total_flux_err)

#             n, k = len(time_new), 3
#             bic_transit = compute_BIC(logL_transit, n, 3)
#             bic_long_box  = compute_BIC(logL_long_box, n, 3)
#             bic_wide_box  = compute_BIC(logL_wide_box, n, 3)

#             bic_null = compute_BIC(logL_null, n, 1)
#             delta_BIC = bic_null - bic_transit

#             delta_BIC_box = (bic_wide_box + bic_long_box)/2 - bic_transit

#             if verbose:
#                 print(f"ΔBIC={delta_BIC:.2f}, ΔBIC_Box={delta_BIC_box:.2f}  (threshold={min_delta_BIC})")

#             if use_AIC:
#                 aic_transit = compute_AIC(logL_transit, k)
#                 print(f"AIC={aic_transit:.2f}")





#     print('depths: ',depth_singles)
    
   

    new_T0_periodic = []
    planet_name = 0
    params_df = []
    
#     if len(init_periodic_singles)>0:
#         indxs =np.unique([round(x[1], 3) for x in init_periodic_singles], return_index=True)[1]
# #         print(indxs)
#         periodic_singles = np.array(init_periodic_singles)[indxs]
#         print('FOUND PERIODIC SINGLES', periodic_singles)
#         for ppp in periodic_singles:
#             T0_min, per, T0_all = ppp
#             tdur_, depth_, T0_, period_ = run_mcmc_code_for_tdur_and_depth(np.array(total_time), total_flux, total_flux_err, T0_min, per, ab)
#             planet_name +=1
        
#             planet_df.loc[len(planet_df.index)] = [int(ticid), planet_name, period_, T0_, tdur_,depth_]
#     singles_indx = np.where(np.isin(t0_singles, init_periodic_singles, invert = Trfe))
#     t0_singles = t0_singles[singles_indx]
#     dur_singles = dur_singles[singles_indx]
#     depth_singles = depth_singles[singles_indx]
    for sss in range(len(t0_singles)):
        print(sss)
        planet_name+=1

        if run_1:
            planet_df.loc[len(planet_df.index)] = [int(ticid), int(planet_name), np.inf, t0_singles[sss], dur_singles[sss],1 -depth_singles[sss]]
        else:
#             print('data file 2', data_file)
            new_params = pymc_new_general_function(np.array(total_time), total_flux, total_flux_err, t0_singles[sss], [dur_singles[sss], ab, 1.-depth_singles[sss]], 'Single')
            params_df.append(new_params)
            
            t0_, period_, depth_, tdur_, q = new_params.loc['t0', 'mean'], new_params.loc['Per', 'mean'], new_params.loc['depth', 'mean'], new_params.loc['dur', 'mean'], new_params.loc['SNR', 'mean']
            if not np.isnan(t0_):
    
                planet_df.loc[len(planet_df.index)] = [int(ticid), int(planet_name), np.inf, t0_, tdur_,depth_, q]
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



def MCMC_for_planet_parameters(time, flux, uncertainty, period, t0, ab, verbose = True, dur = 0):
    

    period_unc = 0.01
    if period == np.inf:
        period = 365.
        period_unc = 1.
   
    ini_names = ['t0','per','rp','cosi','a','u1','u2','norm']
    ini  = np.array([t0,period,0.01,0.,period,.3,.3,1.0],dtype=float)
    ndim = len(ini)
    err  = np.array([0.001,period_unc, 0.001,0.01,0.01, 0.1, 0.1,0.01])
    pos  = [ini+5e-3*err*np.random.randn(ndim) for rrr in range(NUM_WALKERS)]

    ff=flux
    uu=uncertainty
    tt=time
    PP=period
    cad= np.min(np.diff(time))
    u1, u2 = ab
    global PARAMS
    
    if dur == 0:

        PARAMS = [flux,uncertainty,time,cad,t0, u1, u2]

        
    else:
        use_flux = flux[np.abs(time-t0)<(10*dur/24.)]
        use_unc  = uncertainty[np.abs(time-t0)<(10*dur/24.)]
        use_time = time[np.abs(time-t0)<(10*dur/24.)]
        PARAMS   = [use_flux,use_unc,use_time,cad,t0, u1, u2]

    with Pool(8) as pool:
        
        sampler = emcee.EnsembleSampler(NUM_WALKERS,ndim,lnprob_MCMC_global,pool=pool)

        sampler.run_mcmc(pos,NCHAIN,progress=True)

    del globals()['PARAMS']
    
    samples = sampler.chain[:, BCHAIN:, :].reshape((-1,ndim))
    
    if verbose: 
        fig = corner.corner(samples, labels = ini_names)
        plt.show()

    report = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0))))
#     print('making corner plot')
#     fig = corner.corner(samples, labels=ini_names)
#     plt.show()
#     print('done with corner plot')
    t0 = report[0][0]
    per = report[1][0]
    rp = report[2][0]
    cosi = report[3][0]
    v=np.percentile(np.arccos(samples[:,3])*180/np.pi, [50, 16, 84])
    i = (v[1], v[2]-v[1], np.abs(v[1]-v[0])) 
    i = i[0]
    u1 = u1
    u2 = u2
    a = report[4][0]
    norm = report[5][0]
    b = np.abs(a*cosi)
    depth = rp**2.
    T0 = (per*24.)/(a*np.pi)
    tau = rp*T0/np.sqrt(1.-(b**2.))
    dur = np.sqrt(1.-(b**2.))*T0+tau
    win = dur*2
    intran = np.where(np.abs(time-t0)<(dur/2./24.))
    outran = np.where(time[~intran])
    uq = np.full(len(flux),np.std(flux[outran]))
    sigs  = np.mean(uq[intran])
    Q = np.sqrt(len(intran))*depth/sigs
#     Q = np.sqrt(np.sum((1.-flux[intran])/uq[intran]))
    result = [report[0],rp,per,cosi,i, a,u1,u2,norm,b,depth,dur,win,tau,Q]
    if verbose: 
        result_name = ['t0','rp','per','cosi','i', 'a','u1','u2','norm','b','depth','dur','win','tau','Q']
#         for iii in range(len(result)):
#             print(result_name[iii],result[iii])
    return [t0, per, depth, dur, rp, cosi, a, b, u1, u2, norm, win, tau, Q]


# In[ ]:

def calculating_posterior_distributions(chains, time, sample_size, u1, u2, cad, have_per = True):
    a,b,c = chains.shape
    param_rand = np.zeros((sample_size, c))
    models = np.zeros((sample_size, len(time)))

    param_sample = chains.reshape((a*b, c))

    np.random.seed(42)
    rand_indx = np.random.randint(0, a*b, sample_size)

    for iii in range(sample_size):
        if have_per:

            t0,per,rp,cosi,a,norm = param_sample[rand_indx[iii]]
        else:
            t0,rp,cosi,a,norm = param_sample[rand_indx[iii]]
            per = max(time) - min(time)
        models[iii, :] =  predict_lc(time,t0,per,rp,cosi,a,u1,u2,cad)
    
    med_flux = np.zeros(len(time))
    up_flux  = np.zeros(len(time))
    dwn_flux = np.zeros(len(time))

    for jjj in range(len(time)):
        med_flux[jjj], up_flux[jjj], dwn_flux[jjj] = np.percentile(models[:, jjj], [50, 84, 16])

    return med_flux, up_flux, dwn_flux

    
    

def addChain(*args):
    temp = [arg for arg in args if arg is not None]
    if len(temp) > 1:
        return np.concatenate(temp, axis=0)
    else:
        return np.array(temp[0])
    
    


def general_mcmc_function(time, flux, unc, t0, other_pars, type_fn, data_file = "", verbose = True, chain_diff = 0):
    # np.random.seed(42)
        
#     time_here = tm.time()
#     print('original data cadence: ', 24*60*np.median(np.diff(time)))
#     time, flux, unc = bin_data_with_diff_cadences_many_args(time, flux = flux, unc = unc, min_cad = 10/60/24)
#     print('time to bin: ', (tm.time()-time_here)/60, 'minutes' )
#     print('used data cadence: ', 24*60*np.median(np.diff(time)))
    always_keep = False
    if chain_diff>0:
        always_keep = True
    

    global PARAMS
#     print('data file 3', data_file)
    if len(data_file)>0:
        target = str(data_file.split('/')[-2])
    
    mask = np.logical_or(np.logical_or(np.isnan(flux),  np.isnan(time)),  np.isnan(unc))
    time = time[~mask]
    unc  = unc[~mask]
    flux = flux[~mask]

#     else:
#         target = 'no_name_'

    cad = np.min(np.diff(time))*60.*24.
    
#     filename = target[6:-10]+'_'+type_fn+'_'+"backend.h5"
    
    if type_fn == 'Single':
        dur, ab, depth = other_pars
#         print('single; depth going into MCMC (should be SMALL)', depth)

        per = np.max(time)-np.min(time)
        u1, u2 = ab

        ini_names = ['t0','rp','cosi','a', 'norm']
        ini = np.array([t0,np.sqrt(depth),0.,per, 1.0],dtype=float)
        ndim = len(ini)
        
        err = np.array([0.001,0.001,0.05,0.1, 0.01])
        pos = [ini+5e-3*err*np.random.randn(ndim) for rrr in range(NUM_WALKERS)]
        
        indxes = np.where(np.abs(time-t0)<(1.+dur/24.))
        use_flux = np.array(flux)[indxes]
        use_unc  = np.array(unc)[indxes]
        use_time = np.array(time)[indxes]
        cad = np.min(np.diff(use_time))*60.*24.

        PARAMS   = [use_flux,use_unc,use_time,per,cad,t0, u1, u2]
        mcmc_func = lnprob_MCMC_global_single

    elif type_fn == 'Periodic':
        per, ab, depth = other_pars
        print('periodic; starting period', per)

        ini_names = ['t0','per','rp','cosi','a','norm']
        ini  = np.array([t0,per,np.sqrt(depth),0.,per,1.0],dtype=float)
        ndim = len(ini)
        
        err  = np.array([0.001,0.001, 0.001,0.01,0.1,0.01])
        pos  = [ini+5e-3*err*np.random.randn(ndim) for rrr in range(NUM_WALKERS)]
        u1, u2 = ab
        use_flux = np.array(flux)
        use_unc  = np.array(unc)
        use_time = np.array(time)
        
        PARAMS   = [use_flux,use_unc,use_time,cad,t0, u1, u2]
        mcmc_func = lnprob_MCMC_global

#     backend = emcee.backends.HDFBackend(filename)
#     backend.reset(NUM_WALKERS, ndim)


    # with Pool(8) as pool:
        
        # sampler = emcee.EnsembleSampler(NUM_WALKERS,ndim,mcmc_func,pool=pool, backend=backend)

    #     sampler.run_mcmc(pos,max_n,progress=True)

    # samples = sampler.chain[:, BCHAIN:, :].reshape((-1,ndim))
    sampler = emcee.EnsembleSampler(NUM_WALKERS,ndim,mcmc_func)#, backend=backend)

    # index = 0

    # autocorr = np.empty(max_n)

    # # This will be useful to testing convergence
    # old_tau = np.inf
    
    # # Now we'll sample for up to max_n steps
    # for sample in sampler.sample(coords, iterations=max_n, progress=True):
    #     # Only check convergence every 100 steps
    #     if sampler.iteration % 100:
    #         continue
    
    #     # Compute the autocorrelation time so far
    #     # Using tol=0 means that we'll always get an estimate even
    #     # if it isn't trustworthy
    #     tau = sampler.get_autocorr_time(tol=0)
    #     autocorr[index] = np.mean(tau)
    #     index += 1
    
    #     # Check convergence
    #     converged = np.all(tau * 100 < sampler.iteration)
    #     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    #     if converged:
    #         break
    #     old_tau = tau

    index = 0

    finished = False
    step = 0
    tol = 20
    state = pos
    max_n = 2E4 - chain_diff
    print('max steps', max_n)
    pbar = tqdm(total=max_n)
    
    while ((not finished) and (step<max_n+1)):
        state = next(sampler.sample(state, iterations=1))
    
        p = state.coords
        ln_prob = state.log_prob

        if step == 0:
            chain = addChain(p[np.newaxis,:,:])
            prob = addChain(ln_prob[:,np.newaxis])
            
        else:
            chain = addChain(chain, p[np.newaxis,:,:])
            prob = addChain(prob, ln_prob[:,np.newaxis])

        if step % 500 == 0:
            try:
                result_params = chain[500:, :, :]
                testing = [np.percentile(result_params[:, :, x], [50, 16, 84]) for x in range(ndim)]
                
            except Exception as err:
#                 print(err)
                result_params = chain[-500:,:,:]
                
            t0_try,per_try,rp_try,cosi_try,a_try,norm_try = 0, 0, 0, 0, 0, 0
            
            model_time = np.arange(min(time),max(time),1/48)
            
            if type_fn == 'Periodic':
                
                t0_try,per_try,rp_try,cosi_try,a_try,norm_try = list(map(lambda v:(v[0], np.abs(v[0]-v[1]), np.abs(v[2]-v[0])),[np.percentile(result_params[:, :, x], [50, 16, 84]) for x in range(ndim)]))

                med_model_flux, u_lim_model_flux, dwn_lim_model_flux = calculating_posterior_distributions(result_params, model_time, 750, u1, u2, cad)
                
            elif type_fn == 'Single':
                
                t0_try,rp_try,cosi_try,a_try,norm_try =list(map(lambda v: (v[0], np.abs(v[0]-v[1]), np.abs(v[2]-v[0])), [np.percentile(result_params[:, :, x], [50, 16, 84]) for x in range(ndim)]))
                per_try = np.array([per, per/2, per/2])

                med_model_flux, u_lim_model_flux, dwn_lim_model_flux = calculating_posterior_distributions(result_params, model_time, 750, u1, u2, cad, have_per = False)
            
            # flux_theo_try = predict_lc(model_time,t0_try[0],per_try[0],rp_try[0],cosi_try[0],a_try[0],u1,u2,cad)

            try:
                
                tau = sampler.get_autocorr_time(tol=tol)
                if step==max_n:
                    flat_step_taus = step_taus.flatten()

#                     print('mean of all step_taus ', np.nanmean(flat_step_taus))

                    # print('mean of step_taus by param: ', [ini_names[x]+': mean: '+str(np.nanmean(step_taus[:, x]))+' max: '+str(np.nanmax(step_taus[:, x])) for x in range(ndim)])
#                     print('t0 properties: ', ini_names[0]+': mean: '+str(np.nanmean(step_taus[:, 0]))+', median: '+str(np.nanmedian(step_taus[:, 0]))+' max: '+str(np.nanmax(step_taus[:, 0])))

#                     print('last step_tau: ', step_taus[-1])
                    
                if not np.any(np.isnan(tau)):
                    print(f"step {step} progress {step/(tau*tol)} current effective number of samples {step/tau}")

                    finished = True 
                    print('hurray!')

                    
            except emcee.autocorr.AutocorrError as err:

                tau = err.tau

#                 print(f"step {step} progress {step/(tau*tol)} current effective number of samples {step/tau}")

                
                if step == max_n: 
                    print(step)
                    last_step_taus = step_taus[-1]
                    mean_not_norm = np.mean(last_step_taus[:-1])
                    
                    check_for_singles = np.all(last_step_taus[1:-1]<mean_not_norm) and last_step_taus[0]>mean_not_norm
                    
                    if (len(last_step_taus[last_step_taus > tol]) > 1 and not (type_fn == 'Single' and check_for_singles)):
                        print('tolerance', tol, 'med', np.median(last_step_taus))

                        print(f'current number of effective steps {step/(tau)}', 'good enough')
                        finished = True

        if step == 0:
                        
            taus = addChain(tau[:,np.newaxis])
            step_taus = np.array([step/tau])

        else:
            taus = addChain(taus, tau[:,np.newaxis])

            step_tau = step/tau
            step_taus = np.concatenate((step_taus, np.array([step_tau])))

        if finished or step == max_n:
            # print('step ', step)
            # print('step taus', step_taus)
#             print('was this ever good enough? ', len(np.where(step_taus>tol)[0])>0)
#             print('mean of all step_taus ', np.nanmean(step_taus.flatten()))
            print('mean last step tau', np.nanmean(step_taus[-1]))
#             print('t0 properties: ', ini_names[0]+': mean: '+str(np.nanmean(step_taus[:, 0]))+', median: '+str(np.nanmedian(step_taus[:, 0]))+' max: '+str(np.nanmax(step_taus[:, 0])))
            print('last step_tau: ', step_taus[-1])
            
            
#             CSV_tau_files
#             with PdfPages('./DV_reports/'+os.path.dirname(data_file).split('/')[-1]+'.pdf') as pdf: #+str(planet_num)+'.pdf') as pdf:

#             fig0 = plt.figure(figsize=(8.5, 11),constrained_layout=True,dpi=100)
#             gs = fig0.add_gridspec(2,1,height_ratios=[1, 1.5], hspace = 0.1) #create grid for subplots - makes it easier to assign where each plot goes

#             gs0 = gs[0].subgridspec(1, 2)
#             ax1 = [fig0.add_subplot(gs0[0,0]), fig0.add_subplot(gs0[0,1])]

#             gs1 = gs[1].subgridspec(2, 3)   
#             ax_new = [fig0.add_subplot(gs1[x,y]) for x in range(2) for y in range(3)]


# #             ax1 = plt.subplots(gs0)
            
#             ax1[0].scatter(time, flux/np.nanmedian(flux), color = 'k', s = 3, zorder = 10, alpha= 0.75)
#             ax1[0].plot(model_time , med_model_flux, color = 'r', lw = 0.8, alpha = 0.8, zorder = 40)
#             ymi, yma= ax1[0].get_ylim()
#             xmi, xma= ax1[0].get_xlim()
#             n_up_per = np.round(np.abs(xma - t0_try[0])/per_try[0])
#             n_dwn_per = np.round(np.abs(xmi - t0_try[0])/per_try[0])
#             all_t0 = t0_try[0] + per_try[0]*np.arange(-1*n_dwn_per, n_up_per)
#             ax1[0].vlines(t0_try[0], ymi, yma, alpha = 0.5, zorder=0, lw = 2.5, color = 'orange')
#             ax1[0].vlines(all_t0, ymi, yma, alpha = 0.5, zorder=0, lw = 0.75, color = 'orange')
#             ax1[0].vlines(t0,     ymi, yma, alpha = 0.5, zorder=0, lw = 1.5, color = 'green')
#             ax1[0].set_xlim(xmi, xma)

#             x = ((time - t0_try[0] + 0.5*per_try[0]) % per_try[0]) -( 0.5*per_try[0])
#             x_model = ((model_time - t0_try[0] + 0.5*per_try[0]) % per_try[0]) -( 0.5*per_try[0])
#             m = np.abs(x) < 1.
#             m_model = np.abs(x_model) <1.

#             phase_x_model = np.linspace(-1., 1., 1000)
#             phase_flux_model= predict_lc(phase_x_model,0,per_try[0],rp_try[0],cosi_try[0],a_try[0], u1, u2, cad)


#             t0_diff = ((t0 - t0_try[0] + 0.5*per_try[0]) % per_try[0]) -( 0.5*per_try[0])

#             ax1[1].scatter(x[m], flux[m]/np.nanmedian(flux), color = 'k', s = 3, zorder = 10, alpha = 0.75)
#             ax1[1].plot(phase_x_model, phase_flux_model, color = 'r',lw = 2, alpha = 0.8, zorder = 40)
#             ax1[1].fill_between(x_model[m_model], u_lim_model_flux[m_model], dwn_lim_model_flux[m_model], color='r', alpha=0.25)

# #                 axes[1].plot(x_model[m_model], med_model_flux[m_model],      color = 'r',lw = 2, alpha = 0.8, zorder = 40)
# #                 axes[1].scatter(x_model[m_model],  u_lim_model_flux[m_model], color = 'purple', alpha=0.5, s=3)
# #                 axes[1].scatter(x_model[m_model], dwn_lim_model_flux[m_model], color = 'brown', alpha=0.5, s = 3)

# #                 ymi, yma= axes[1].get_ylim()

#             ax1[1].vlines(0, ymi, yma, alpha = 0.6, zorder=0, lw = 1.5, color = 'orange')
#             ax1[1].vlines(t0_diff, ymi, yma, alpha = 0.6, zorder=0, lw = 1.5, color = 'green')
#             ax1[1].set_xlim(-0.75, 0.75)
# #             plt.show()
# #             plt.close()
# #             ax_new = plt.subplots(gs1).reshape(1,6)[0]

#             for iii in range(ndim):
#                 param = ini_names[iii]
#                 ax_new[iii].plot(list(range(len(chain[:,:,iii]))), chain[:,:,iii])

#                 ax_new[iii].set_ylabel(param)
#                 # print('num chains', np.shape(chain))

            
#             else:
#                 print('these are my result values: ', [t0_try,rp_try,cosi_try,a_try,norm_try])



#             plt.show()

        step = step + 1
        pbar.update(1)
    pbar.close()

    print('step num ', step, 'finished? ', finished)


    
    print("done processing")
       
        
    burn = int(np.ceil(max_n*2/3))
#     burn = int(np.ceil(np.nanmax(taus)) * 2)
    
#     burn_chain = chain[:,:burn,:]
#     burn_prTT prob[:,:burn]
    
    if step<burn+10:
        burn = int(np.ceil(step/2))
    new_chain = chain[:,burn:,:]
    new_prob = prob[:,burn:]
    
#     print("finished")

    samples = sampler.chain[:, burn:, :].reshape((-1,ndim))
    xxx =  str(np.nanmean(step_taus[-1, 0])).split('.')
    if not np.abs(step_taus[-1, 0])>0:
        xxx = ['bad', 'nan']
        
        
#     print('this is number', xxx)
    
#     print('data file 4', data_file.split('/'))
    target_source = os.path.dirname(data_file).split('/')
    if (not finished) and (step>=max_n):
        add_str = 'N'
        kept = False
    else:
        add_str = 'Y'
        kept = True

    if len(target_source)<3:
        target_source = data_file.split('/')
    fileName = '../saving_MCMC_plots/'+target_source[-1][:-10]+xxx[0]+'_'+xxx[1][:3]+add_str+'.pdf'

    ticid = target_source[-1].split('tic-')[1].split('_')[0]
    print('saving all to csv')
    names = ['TICID','Nsteps','kept?', 'P/S', 'mean_last_st', 'med_last_st']+ini_names
    vals = [ticid,step, kept, type_fn, np.mean(step_taus[-1]), np.median(step_taus[-1])]+list(step_taus[-1])

    print('vals look like? ', vals)
    saving_vals_dict = {key: [value] for key, value in zip(names, vals)}
    pd.DataFrame(saving_vals_dict).to_csv('./CSV_tau_files/ticid_'+ticid+'-savingVals4Comparison_'+xxx[0]+'_'+xxx[1][:3]+'.csv')
#     print('file name', fileName)
    with PdfPages(fileName) as pdf: #+str(planet_num)+'.pdf') as pdf:

        fig0 = plt.figure(figsize=(8.5, 11),constrained_layout=True,dpi=100)
        gs = fig0.add_gridspec(2,1 ,height_ratios=[1.5, 2], hspace = 0.1) #create grid for subplots - makes it easier to assign where each plot goes

        gs0 = gs[0].subgridspec(2, 2)
        ax1 = [fig0.add_subplot(gs0[0,0]), fig0.add_subplot(gs0[0,1]), fig0.add_subplot(gs0[1,0])]

        gs1 = gs[1].subgridspec(3,2)   
        ax_new = [fig0.add_subplot(gs1[x,y]) for x in range(3) for y in range(2)]


#             ax1 = plt.subplots(gs0)

        ax1[0].scatter(time, flux/np.nanmedian(flux), color = 'k', s = 3, zorder = 10, alpha= 0.75)
        ax1[0].plot(model_time , med_model_flux, color = 'r', lw = 0.8, alpha = 0.8, zorder = 40)
        ymi, yma= ax1[0].get_ylim()
        xmi, xma= ax1[0].get_xlim()
        n_up_per = np.round(np.abs(xma - t0_try[0])/per_try[0])
        n_dwn_per = np.round(np.abs(xmi - t0_try[0])/per_try[0])
        all_t0 = t0_try[0] + per_try[0]*np.arange(-1*n_dwn_per, n_up_per)
        ax1[0].vlines(t0_try[0], ymi, yma, alpha = 0.5, zorder=0, lw = 2.5, color = 'orange')
        ax1[0].vlines(all_t0, ymi, yma, alpha = 0.5, zorder=0, lw = 0.75, color = 'orange')
        ax1[0].vlines(t0,     ymi, yma, alpha = 0.5, zorder=0, lw = 1.5, color = 'green')
        ax1[0].set_xlim(xmi, xma)

        x = ((time - t0_try[0] + 0.5*per_try[0]) % per_try[0]) -( 0.5*per_try[0])
        x_model = ((model_time - t0_try[0] + 0.5*per_try[0]) % per_try[0]) -( 0.5*per_try[0])
        m = np.abs(x) < 1.
        m_model = np.abs(x_model) <1.

        phase_x_model = np.linspace(-1., 1., 1000)
        phase_flux_model= predict_lc(phase_x_model,0,per_try[0],rp_try[0],cosi_try[0],a_try[0], u1, u2, cad)


        t0_diff = ((t0 - t0_try[0] + 0.5*per_try[0]) % per_try[0]) -( 0.5*per_try[0])

        ax1[1].scatter(x[m], flux[m]/np.nanmedian(flux), color = 'k', s = 3, zorder = 10, alpha = 0.75)
        ax1[1].plot(phase_x_model, phase_flux_model, color = 'r',lw = 2, alpha = 0.6, zorder = 40)
        ax1[1].fill_between(x_model[m_model], u_lim_model_flux[m_model], dwn_lim_model_flux[m_model], color='r', alpha=0.25, zorder = 20)

#                 axes[1].plot(x_model[m_model], med_model_flux[m_model],      color = 'r',lw = 2, alpha = 0.8, zorder = 40)
#                 axes[1].scatter(x_model[m_model],  u_lim_model_flux[m_model], color = 'purple', alpha=0.5, s=3)
#                 axes[1].scatter(x_model[m_model], dwn_lim_model_flux[m_model], color = 'brown', alpha=0.5, s = 3)

#                 ymi, yma= axes[1].get_ylim()

        ax1[1].vlines(0, ymi, yma, alpha = 0.6, zorder=0, lw = 1.5, color = 'orange')
        ax1[1].vlines(t0_diff, ymi, yma, alpha = 0.6, zorder=0, lw = 1.5, color = 'green')
        ax1[1].set_xlim(-0.75, 0.75)
#             plt.show()
#             plt.close()
#             ax_new = plt.subplots(gs1).reshape(1,6)[0]

        try:
            tdur = dur
        except NameError:
            tdur = (per_try[0]*24.)/(a_try[0]*np.pi)
        print('transit duration after fitting (hours): ', tdur)
        ax1[2].scatter(time[abs(time -  t0_try[0])<(tdur+1/3)], flux[abs(time -  t0_try[0])<(tdur+1/3)]/np.nanmedian(flux), color = 'k', s = 3, zorder = 10, alpha= 0.75)
        xmin, xmax = ax1[2].get_xlim()
        ax1[2].plot(model_time[abs(model_time -  t0_try[0])<(tdur+1/3)] , med_model_flux[abs(model_time -  t0_try[0])<(tdur+1/3)], color = 'r', lw = 0.8, alpha = 0.6, zorder = 40)
        ax1[2].vlines(t0_try[0], ymi, yma, alpha = 0.5, zorder=0, lw = 2.5, color = 'orange')
        ax1[2].vlines(t0,     ymi, yma, alpha = 0.5, zorder=0, lw = 1.5, color = 'green')
        ax1[2].vlines([t0_try[0]+tdur/2, t0_try[0] - tdur/2],     ymi, yma, alpha = 0.5, zorder=0, lw = 2, color = 'dodgerblue')
        ax1[2].set_xlim(xmin, xmax)
        
        

        for iii in range(ndim):
            param = ini_names[iii]
            ax_new[iii].plot(list(range(len(chain[:,:,iii]))), chain[:,:,iii])

            ax_new[iii].set_ylabel(param)


        
        fig1 = corner.corner(samples, labels = ini_names)
        
        pdf.savefig(fig0)
        pdf.savefig(fig1)

        plt.show()

#     if verbose: 
        # tau = sampler.get_autocorr_time()
        # print(tau)
        # burnin = int(2 * np.nanmax(tau))
        # thin = int(0.5 * np.nanmin(tau))
        # log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
        # log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)
        
        # print("burn-in: {0}".format(burnin))
        # print("thin: {0}".format(thin))
        # print("flat chain shape: {0}".format(samples.shape))
        # print("flat log prob shape: {0}".format(log_prob_samples.shape))
        # print("flat log prior shape: {0}".format(log_prior_samples.shape))

        # txt = "burn-in: {0}".format(burnin)+"\nthin: {0}".format(thin) +"\nflat chain shape: {0}".format(samples.shape)+"\nflat log prob shape: {0}".format(log_prob_samples.shape)+"\nflat log prior shape: {0}".format(log_prior_samples.shape)

        # plt.text(0.01, 0.95, txt, transform=plt.gca().transAxes,   horizontalalignment='left', verticalalignment='top')
        
        # plt.savefig("./corner_plots/"+target[:-5]+'_corner_plot')
#         samples = sampler.get_chain()
        # for param in range(len(ini_names)):
        #     new_fig, ax = plt.subplots(1, figsize=(12, 12))
        #     ax.plot(list(range(len(chain[:,:,param]))), chain[:,:,param])
        #     ymin, ymax = ax.get_ylim()
        #     delta_y = ymax-ymin
        #     # delta_chain = max(chain[:,:,param])-min(chain[:,:,param])
        #     ax.vlines([100, 500, 1000, 2000, 3000], ymin-0.1*delta_y, ymax+0.1*delta_y)
        #     # ax.title(ini_names[param])
        #     ax.set_ylim(ymin-0.07*delta_y, ymax+0.07*delta_y)
        #     plt.savefig('./corner_plots/'+data_file.split('/')[-2][:-5]+'_'+ini_names[param]+'_converging')
    
#     samples = sampler.get_chain(discard=BCHAIN)#, flat=True, thin=thin)

    report = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0))))

    t0 = report[0][0]
    u1, u2 = ab

    if type_fn == 'Single':
        
        rp = report[1][0]
        cosi = report[2][0]
        a = report[3][0]
        norm = report[4][0]
        v=np.percentile(np.arccos(samples[:,2])*180/np.pi, [50, 16, 84])

    elif type_fn == 'Periodic':
        per = report[1][0]
        rp = report[2][0]
        cosi = report[3][0]
        a = report[4][0]
        norm = report[5][0]

        v=np.percentile(np.arccos(samples[:,3])*180/np.pi, [50, 16, 84])
        
    i = (v[1], v[2]-v[1], np.abs(v[1]-v[0])) 
    i = i[0]

    b = np.abs(a*cosi)
    depth = rp**2.
    T0 = (per*24.)/(a*np.pi)
    tau = rp*T0/np.sqrt(1.-(b**2.))
    dur = np.sqrt(1.-(b**2.))*T0+tau 
    win = dur*2
#     print('t0', t0, 'dur', dur)
    if type_fn == 'Periodic':
        intran = np.where(transit_mask(time, per, dur/24, t0))[0]
#         print('intran', len(np.where(intran)[0]))
    elif type_fn == 'Single':  
        intran = np.abs(time-t0)<(dur/2./24.)
    outran =np.abs(time-t0)>=(dur/24.)
    
    uq = np.full(len(flux),np.std(flux[outran]))
    sigs  = np.mean(uq[intran])
    
    print('len intran', len(intran), np.sqrt(len(intran)), 'depth', depth, 'sigs', sigs)
    Q = np.sqrt(len(intran))*depth/sigs
#     Q = np.sqrt(np.sum((1.-flux[intran])/uq[intran]))
    result = [report[0][0],rp,per,cosi,i, a,u1,u2,norm,b,depth,dur,win,tau,Q]
    if verbose: 
        result_name = ['t0','rp','per','cosi','i', 'a','u1','u2','norm','b','depth','dur','win','tau','Q']
        for iii in range(len(result_name)):
            print(result_name[iii],result[iii])
    if (not finished) and (step>=max_n) and (not always_keep):
        print('NOT GOING TO KEEP THIS ONE')
        return np.full(len([t0, per, depth, dur, rp, cosi, a, b, u1, u2, norm, win, tau, Q]), np.nan)


    return [t0, per, depth, dur, rp, cosi, a, b, u1, u2, norm, win, tau, Q]



def run_mcmc_code_for_tdur_and_depth(time, flux, unc, t0, period, ab, verbose = False, dur = 0):

    period_unc = 0.01
    if period == np.inf:
        period = 100.
        period_unc = 1.
        
    
    ini_names = ['t0','per','rp','cosi','a','u1','u2','norm']
    ini  = np.array([t0,period,0.01,0.,period,.3,.3,1.0],dtype=float)
    ndim = len(ini)
    err  = np.array([0.01,period_unc,0.01, 0.001, 0.1, 0.01, 0.01, 0.1])
    pos  = [ini+5e-3*err*np.random.randn(ndim) for rrr in range(NUM_WALKERS)]

    ff=flux
    uu=unc
    tt=time
    PP=period
    cad= np.min(np.diff(time))
    u1, u2 = ab
    global PARAMS
    
    if dur == 0:

        PARAMS = [flux,unc,time,cad,t0, u1, u2]

    else:
        indxes = np.where(np.abs(time-t0)<(5*dur/24.))
        use_flux = np.array(flux)[indxes]
        use_unc  = np.array(unc)[indxes]
        use_time = np.array(time)[indxes]
        PARAMS   = [use_flux,use_unc,use_time,cad,t0, u1, u2]

    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(NUM_WALKERS,ndim,lnprob_MCMC_global,pool=pool)

        sampler.run_mcmc(pos,NCHAIN,progress=True)

    del globals()['PARAMS']
    samples = sampler.chain[:, BCHAIN:, :].reshape((-1,ndim))
    
    if verbose: 
        fig = corner.corner(samples, labels = ini_names)
        plt.show()

    report = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0))))
#     print('making corner plot')
#     fig = corner.corner(samples, labels=ini_names)
#     plt.show()
#     print('done with corner plot')
    t0 = report[0][0]
    per = report[1][0]
    rp = report[2][0]
    cosi = report[3][0]
    a = report[4][0]
    b = np.abs(a*cosi)
    depth = rp**2.
    T0 = (per*24.)/(a*np.pi)
    tau = rp*T0/np.sqrt(1.-(b**2.))
    dur = np.sqrt(1.-(b**2.))*T0+tau
    

    if verbose: 
        result_name = ['t0','rp','per','cosi','i', 'a','u1','u2','norm','b','depth','dur','win','tau','Q']
#         for iii in range(len(report)):
#             print(result_name[iii],report[iii])
    return dur, depth, t0, per # [report[0], per, depth, dur, rp, cosi, a, b, u1, u2, norm, win, tau, Q]



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


# def sample_until_converged(model, max_attempts=3, rhat_threshold=1.2):
#     for attempt in range(1, max_attempts + 1):
#         print(f"Sampling attempt {attempt}...")
        
#         if attempt == 1:
#             trace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.9)
#         else:
#             trace = pm.sample(draws=2000*attempt, tune=1000*attempt, chains=4, target_accept=0.95)
        
#         summary = az.summary(trace)
#         if (summary['r_hat'] < rhat_threshold).all():
#             print(f"Converged on attempt {attempt}")
#             return trace
        
#         print(f"Attempt {attempt} failed to converge. Retrying...")

#     raise RuntimeError("Model did not converge after multiple attempts.")


# def sample_until_converged(model, max_attempts=3, rhat_threshold=1.2):
#     # Separate variables dynamically
#     transit_vars = [var for var in model.free_RVs if var.name not in ['unc']]  # exclude unc since it's fixed
#     # If you want to split by name pattern:
#     # transit_vars = [var for var in model.free_RVs if var.name.lower() in ['t0','per','rp_rs','a_rs','cosi','u1','u2','norm']]
    
#     # Use Metropolis for all free RVs (since BatmanOp kills gradients)
#     step = pm.Metropolis(vars=transit_vars)
    
#     for attempt in range(1, max_attempts + 1):
#         print(f"Sampling attempt {attempt}...")
#         trace = pm.sample(step=step, draws=2000*attempt, tune=1000*attempt, chains=4, target_accept=0.9)
        
#         summary = az.summary(trace)
#         if (summary['r_hat'] < rhat_threshold).all():
#             print(f"Converged on attempt {attempt}")
#             return trace
        
#         print(f"Attempt {attempt} failed to converge. Retrying...")
    
#     raise RuntimeError("Model did not converge after multiple attempts.")
    


def sample_until_converged(model, max_attempts=5, rhat_threshold=1.1):
    # Get all free random variables in the model
    free_vars = model.free_RVs
    if not free_vars:
        raise ValueError("No free random variables found for sampling.")
    print('free vars', free_vars)
    # Use Metropolis for all free RVs
    step = pm.Metropolis(vars=free_vars)

    for attempt in range(1, max_attempts + 1):
        print(f"Sampling attempt {attempt}...")
        trace = pm.sample(step=step, draws=2000*attempt, tune=1000*attempt, chains=4, cores = 1)

        summary = az.summary(trace)
        if (summary['r_hat'] < rhat_threshold).all():
            print(f"Converged on attempt {attempt}")
            return trace
#         print('checking nans trace', trace.posterior['SNR'])
        print('checking nanas summary', az.summary(trace))
        print(f"Attempt {attempt} failed to converge. Retrying...")

    raise RuntimeError("Model did not converge after multiple attempts.")

# def predict_lc_pymc(time_lc,t0,P,rp_rs,a,inc,e,omega,u1,u2,cad):
#     oversample = 4
#     params = batman.TransitParams()
#     params.t0  = t0
#     params.per = P
#     params.rp  = rp_rs
#     params.a   = a
#     params.inc = inc
#     params.ecc = e
#     params.w = omega*180./np.pi
#     params.u = [u1,u2]
#     params.limb_dark = "quadratic"
        
#     cad = pt.switch(cad > 0, cad, 30.)
    
#     cad_val = cad if isinstance(cad, float) else 30.  # fallback if needed
#     cad_val = cad_val / 24. / 60.

#     m = batman.TransitModel(params, time_lc ,supersample_factor = oversample, exp_time = cad/24./60.)

#     flux_theo = m.light_curve(params)

#     return flux_theo

# def predict_lc_wrapper(pars):
#     time_lc, cad, t0, P, rp_rs, inc, a_rs, u1, u2 = pars
#     e = 0.
#     omega = 90.
     
#     return predict_lc_pymc(time_lc, t0, P, rp_rs, a_rs, inc, e, omega, u1,u2, cad)


# def pymc_new_general_function(time, flux, unc, t0, other_pars, type_fn, verbose = True, keep_ld_fixed = True, phase_fold = True):

#     mask = np.logical_or(np.logical_or(np.isnan(flux),  np.isnan(time)),  np.isnan(unc))
#     time = time[~mask]
#     unc  = unc[~mask]
#     flux = flux[~mask]

#     cad = np.min(np.diff(time))*60.*24.
    
#     if type_fn == 'Single':
#         dur, ab, depth = other_pars
#         per = np.max(time)-np.min(time)
#         per_unc = 10
        
#         indxes = np.where(np.abs(time-t0)<(1.+dur/24.))
        
#         use_time = np.array(time)[indxes]
#         use_flux = np.array(flux)[indxes]
#         use_unc  = np.array(unc)[indxes]
#     elif type_fn == 'Periodic':
#         per, ab, depth = other_pars
#         per_unc = 0.1
#         use_time = np.array(time)
#         use_flux = np.array(flux)
#         use_unc  = np.array(unc)
        
#     u1, u2 = ab
        
#     with pm.Model() as model:


#         time_lc = pm.Data("time_lc", use_time, mutable = False)
#         cad = pm.Data("cad", cad, mutable = False)


#         t0    = pm.TruncatedNormal("t0", mu=t0, sigma=0.01, lower=t0-0.25, upper=t0+0.25)
#         Per   = pm.Normal("Per", mu=per, sigma=per_unc)
#         rp_rs = pm.TruncatedNormal("rp_rs", mu=pt.sqrt(depth), sigma=0.01, lower=0, upper=1)
#         a_rs  = pm.TruncatedNormal("a_rs", mu=per, sigma=0.1, lower=1)
#         cosi  = pm.TruncatedNormal("cosi", mu=0., sigma=0.01, lower=0, upper=1)
#         norm  = pm.Normal("norm", mu=1.0, sigma=0.01)

#         inc = pm.Deterministic('inclination', pt.arccos(cosi)*180./np.pi)

#         if keep_ld_fixed:
#             u1 = pm.Data("u1", u1, mutable=False)
#             u2 = pm.Data("u2", u2, mutable=False)
#         else:
#             u1 = pm.TruncatedNormal("u1", mu=u1, sigma=0.01, lower=0, upper=1)
#             u2 = pm.TruncatedNormal("u2", mu=u2, sigma=0.01, lower=0, upper=1)
            

        
#         #not phase folded
#         if phase_fold:
#             print('phase fold ingredients: time: ', time_lc, '\nt0: ', t0, '\nPer: ', Per)
#             folded_phase = ((time_lc - t0 + 0.5*Per) % Per) -( 0.5*Per)
#             print('folded phase', folded_phase)
#             sort_indx    = pt.argsort(folded_phase)
#             print('sorted phase index', sort_indx)

#             phase = folded_phase[sort_indx]
#             print('sorted phase', phase)
#             p_cad = pt.median(phase[1:] - phase[:-1]) * 60 * 24
            
#             p_pars = [phase+t0, p_cad, t0, Per, rp_rs, inc, a_rs, u1, u2]
#             p_flux_model = predict_lc_wrapper(p_pars)
        
#             pm.Normal("obs", mu=p_flux_model * norm, sigma=use_unc, observed=use_flux[sort_indx])


        
#         #phase folded
#         else:
#             pars = [time_lc, cad, t0, Per, rp_rs, inc, a_rs, u1, u2]

#             flux_model = predict_lc_wrapper(pars)

#             pm.Normal("obs", mu=flux_model * norm, sigma=use_unc, observed=use+flux)
            
            
            
#         #Defining all non-used deterministic variables to be saved later
#         b      = pm.Deterministic('b', a_rs*cosi) #
#         depth  = pm.Deterministic('depth', rp_rs**2) #
#         T_dur0 = (per*24.)/(a*np.pi)
#         tau    = pm.Deterministic('tau', rp_rs*T_dur0/pt.sqrt(1.-(b**2.))) #
#         dur    = pm.Deterministic('dur', pt.sqrt(1.-(b**2.))*T0+tau ) #
#         win    = dur*2 
#         if type_fn == 'Periodic':
#             intran = np.where(transit_mask(time, per, dur/24, t0))[0]
#     #         print('intran', len(np.where(intran)[0]))
#         elif type_fn == 'Single':  
#             intran = np.abs(time_lc-t0)<(dur/2./24.)
#         outran = np.abs(time_lc-t0)>=(dur/24.)
#         uq     = np.full(len(flux),np.std(flux[outran]))
#         sigs   = np.mean(uq[intran])
                        
#         SNR    = pm.Deterministic('SNR', pt.sqrt(len(intran))*depth/sigs ) #
# #         SNR    = pm.Deterministic('SNR', pt.sqrt(np.sum((1.-use_flux[intran])/use_unc[intran]))) #

                    

#     with model:
#         try:
#             trace = sample_until_converged(model)
#             summary = extract_summary_dataframe(trace)

#         except RuntimeError as error:
#             return pd.DataFrame(np.nan, index=summary.index, columns=summary.columns)
            

#     # Summary statistics
#     print(summary)

#     if verbose:

#         # Trace plots
#         az.plot_trace(trace)

#         # Posterior plots
#         az.plot_posterior(trace)

#         # Save results
# #         az.to_netcdf(trace, "trace_output.nc")
#         plt.show()
    
#     return summary
                                


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
        return [pt.zeros_like(i) for i in inputs]

    
def set_up_variables_for_pymc_fit(time, flux, unc, t0, other_pars, type_fn):
    mask = np.logical_or(np.logical_or(np.isnan(flux),  np.isnan(time)),  np.isnan(unc))
    time = time[~mask]
    unc  = unc[~mask]
    flux = flux[~mask]

    cad = np.min(np.diff(time))*60.*24.
    
    if type_fn == 'Single':
        dur, ab, depth = other_pars
        per1 = np.max(time)-np.min(time)
        per2 = dur*10*np.pi/24
        per = np.min([per1, per2])
        print('time difference', np.max(time)-np.min(time))
        if per<10:
            per = 27.8
        
        indxes = np.where(np.abs(time-t0)<(1.+dur/24.))
        
        use_time = np.array(time)[indxes]
        use_flux = np.array(flux)[indxes]
        use_unc  = np.array(unc)[indxes]
    elif type_fn == 'Periodic':
        per, ab, depth = other_pars
        use_time = np.array(time)
        use_flux = np.array(flux)
        use_unc  = np.array(unc)
        
    u1, u2 = ab
        
    return use_time, use_flux, use_unc, per, u1, u2, depth, cad


def median_pytensor(x):
    sorted_x = pt.sort(x)
    n = x.shape[0]
    mid = n // 2
    return pt.switch(
        pt.eq(n % 2, 0),
        (sorted_x[mid - 1] + sorted_x[mid]) / 2.0,
        sorted_x[mid]
    )

    
def pymc_new_general_function(time, flux, unc, T0, other_pars, type_fn, verbose = True, keep_ld_fixed = True, phase_fold = False):

    time, flux, unc, Per, U1, U2, Depth, cad = set_up_variables_for_pymc_fit(time, flux, unc, T0, other_pars, type_fn)
    batman_op = BatmanOp()

    
    
    G = 2941.18330364 #R_s^3 M_s^-1 day^−2
    
    with pm.Model() as model:

        t0    = pm.TruncatedNormal("t0", mu=T0, sigma=0.01, lower=T0-0.25, upper=T0+0.25)

        if type_fn == 'Single':
            print('single')
            per   = pm.TruncatedNormal("Per", mu=Per, sigma=10., lower= 0)
            ecc   = pm.TruncatedNormal("Eccen", mu=0, sigma=0.25, lower = 0, upper = 1)
            sigma_a = 5.
            a_smaj = 215/10

        elif type_fn == 'Periodic':
            print('periodic, period =', Per, 'days')

            per   = pm.TruncatedNormal("Per", mu=Per, sigma=0.1, lower= 0.25)
            ecc   = 0
            sigma_a = 1.
#             a_smaj = Per
            a_smaj = ((Per/365)**(2/3))*215/2
            print('a_smaj_guess', a_smaj)
#             norm  = pm.Normal("norm", mu=1.0, sigma=0.01)


        rp_rs = pm.TruncatedNormal("rp_rs", mu=pt.sqrt(Depth), sigma=0.1, lower=0, upper=1)
        a_rs  = pm.TruncatedNormal("a_rs", mu=a_smaj, sigma=sigma_a, lower=1)
        cosi  = pm.TruncatedNormal("cosi", mu=0., sigma=0.01, lower=0, upper=1/a_rs)

        inc = pm.Deterministic('inclination', pt.arccos(cosi)*180./np.pi)

        if keep_ld_fixed:
            u1 = U1
            u2 = U2
        else:
            
            u1 = pm.TruncatedNormal("u1", mu=U1, sigma=0.01, lower=0, upper=1)
            u2 = pm.TruncatedNormal("u2", mu=U2, sigma=0.01, lower=0, upper=1)
          
            
        cad = pt.switch(cad > 0, cad, 30.)

            
        #Defining all non-used deterministic variables to be saved later
        b      = pm.Deterministic('b', pt.abs(a_rs*cosi)) #
        depth  = pm.Deterministic('depth', rp_rs**2) #
        T_dur0 = (per*24.)/(a_rs*np.pi)
        tau    = pm.Deterministic('tau', rp_rs*T_dur0/pt.sqrt(1.-(b**2.))) #
        dur    = pm.Deterministic('dur', pt.sqrt(1.-(b**2.))*T_dur0 + tau ) #
        win    = pm.Deterministic('win', dur*2 ) #
        

        # Masks (symbolic)
        if type_fn == 'Periodic':
            intran_mask = transit_mask_tensors(time, per, dur / 24., t0)  # must return PyTensor boolean
            intran_idx = pt.nonzero(intran_mask)[0]
        elif type_fn == 'Single':
            intran_mask = pt.abs(time - t0) < (dur / 2. / 24.)
            intran_idx = pt.nonzero(intran_mask)[0]
            per_scaled = pm.Deterministic('scaled_P', pt.sqrt(3 * np.pi / G * a_rs**3)) #
            
        outran_mask = pt.invert(intran_mask)

        # Compute uq and sigs symbolically

        out_flux = flux * outran_mask  # zeros elsewhere
        count = pt.maximum(pt.sum(outran_mask), 1)  # avoid zero
        mean_out = pt.sum(out_flux) / count
        std_out = pt.sqrt(pt.sum(outran_mask * (flux - mean_out)**2) / count)

        N_tran = pt.sum(intran_mask)
        uq = pt.ones_like(flux) * std_out
#         sigs = pt.mean(pt.where(intran_mask, uq, 0))
        

        sigs = pt.switch(N_tran > 0,pt.mean(pt.where(intran_mask, uq, 0)), 1e6) 
        print('N_intran', pm.draw(N_tran), 'depth', pm.draw(depth), 'sig', pm.draw(sigs))
        # SNR


        SNR_val = pt.switch(pt.gt(N_tran, 0), pt.sqrt(N_tran) * depth / sigs, 0)
        SNR_clipped = pt.clip(SNR_val, 0, 1e4)
        SNR_final = pt.where(pt.eq(SNR_clipped, 1e4), 1, SNR_clipped)
        
        if not phase_fold:
            SNR = pm.Deterministic("SNR", SNR_final)
        
#         if type_fn =='Single':
            
        norm = pm.Deterministic("norm", median_pytensor(out_flux))
        #not phase folded
        if phase_fold:
            folded_phase = ((time - T0 + 0.5*Per) % Per) -( 0.5*Per)
            sort_indx    = np.argsort(folded_phase)
            phase = folded_phase[sort_indx]
            p_cad = np.median(phase[1:] - phase[:-1]) * 60 * 24
            
            p_flux_model = batman_op(phase+T0, t0, per, rp_rs, a_rs, inc, u1, u2, ecc, p_cad)
            intran_mask = transit_mask_tensors(phase+t0, per, dur / 24., t0)  # must return PyTensor boolean
            outran_mask = pt.invert(intran_mask)
            out_flux = flux * outran_mask  # zeros elsewhere
            count = pt.maximum(pt.sum(outran_mask), 1)  # avoid zero
            mean_out = pt.sum(out_flux) / count
            std_out = pt.sqrt(pt.sum(outran_mask * (flux - mean_out)**2) / count)

            N_tran = pt.sum(intran_mask)
            uq = pt.ones_like(flux) * std_out
    #         sigs = pt.mean(pt.where(intran_mask, uq, 0))


            sigs = pt.switch(N_tran > 0,pt.mean(pt.where(intran_mask, uq, 0)), 1e6) 
            print('N_intran', pm.draw(N_tran), 'depth', pm.draw(depth), 'sig', pm.draw(sigs))
            # SNR


            SNR_val = pt.switch(pt.gt(N_tran, 0), pt.sqrt(N_tran) * depth / sigs, 0)
            SNR_clipped = pt.clip(SNR_val, 0, 1e4)
            SNR_final = pt.where(pt.eq(SNR_clipped, 1e4), 1, SNR_clipped)

            SNR = pm.Deterministic("SNR", SNR_final)
            
            
            pm.Normal("obs", mu=p_flux_model * norm, sigma=unc, observed=flux[sort_indx])


        
        #phase folded
        else:

            flux_model = batman_op(time,t0, per, rp_rs, a_rs, inc, u1, u2, ecc, cad)

            pm.Normal("obs", mu=flux_model * norm, sigma=unc, observed=flux)
            
            

    with model:
        try:

            trace = sample_until_converged(model)
            summary = extract_summary_dataframe(trace)

        except RuntimeError as error:
            return pd.DataFrame(columns=['mean', 'median', 'sd', 'hdi_16%', 'hdi_84%', 'r_hat'])
            

    # Summary statistics
#     print(summary)

    if verbose:

        # Trace plots
        az.plot_trace(trace)

        # Posterior plots
        az.plot_posterior(trace)

        # Save results
#         az.to_netcdf(trace, "trace_output.nc")
        plt.show()
    print('summary', summary)
    return summary

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
    
    indexes_split_unorganize = breaking_up_data(time, 2.)   
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
        

        


# In[ ]:





# In[ ]:





