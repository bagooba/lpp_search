import batman
import emcee
import glob
import os
import shutil
import math
import corner
import numba
import itertools

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



import warnings
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))

from Functions_all import *



all_tois = pd.read_csv('/users/malharris/lpp_search/data/dr2_dr3_ticid_final.csv')

gaia_ids = [all_tois['gaia_dr3_id'][x] for x in range(len(all_tois)) if type(all_tois['gaia_dr3_id'][x]) == str]


def mk_target_dir_mv_fits_file(fits_file_with_GAIAid, sector_df):
    gaia_ID = fits_file_with_GAIAid.split('-')[2]
    file_gaia = 'Gaia DR3 '+gaia_ID
    

#     print(gaia_ID, sector_df[sector_df['gaia_id'].astype(str)==gaia_ID]['tic_id'])
    tic_id_index = sector_df[sector_df['gaia_dr3_id'].astype(str)==file_gaia]['tic_id'].index
    if len(tic_id_index)>1:
        print(tic_id_index)
    ticid = int(sector_df['tic_id'][tic_id_index[0]][4:])
    mkdir_if_doesnt_exist('/carc//', 'target_tic-'+str(ticid)+'_gaiaID-'+str(gaia_ID))
    os.rename(fits_file_with_GAIAid, '../new_toi_data/target_tic-'+str(ticid)+'_gaiaID-'+str(gaia_ID)+'/'+fits_file_with_GAIAid.split('/')[-1])
    
    
iii=0
for file in glob.glob('/easley/scratch/users/malharris/*/fits/*.fits'):
    gaia_id_check = set([g_id for g_id in gaia_ids if str(g_id) in file])
    if len(gaia_id_check)==1:
        mk_target_dir_mv_fits_file(file, all_tois)
        iii+=1
        if iii%10 == 0:
            print(iii)
    elif len(gaia_id_check)>1:
        print('weird', gaia_id_check)
