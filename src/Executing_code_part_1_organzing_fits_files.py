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

gaia_ids = [all_tois['dr3_source_id'][x] for x in range(len(all_tois))]


def mk_target_dir_mv_fits_file(fits_file_with_GAIAid, id_df):
    gaia_ID = fits_file_with_GAIAid.split('-')[1]
    print('checking gaia id', gaia_ID)
    

#     print(gaia_ID, sector_df[sector_df['gaia_id'].astype(str)==gaia_ID]['tic_id'])
    tic_id_index = id_df[id_df['dr3_source_id'].astype(str)==gaia_ID]['tic_id'].index

    ticid = str(id_df['tic_id'][tic_id_index[0]])
    mkdir_if_doesnt_exist('/carc/scratch/projects/dragomir/dragomir2016394/search_files/', 'target_tic-'+str(ticid)+'_gaiaID-'+str(gaia_ID))
    os.rename(fits_file_with_GAIAid, '/carc/scratch/projects/dragomir/dragomir2016394/search_files/target_tic-'+str(ticid)+'_gaiaID-'+str(gaia_ID)+'/'+fits_file_with_GAIAid.split('/')[-1])
    

def main(gaia_ids):

    for g_id in gaia_ids:
        file_lst = glob.glob('/carc/scratch/projects/dragomir/dragomir2016394/mdwarfs/*/fits/*.fits')
        good_files = [file for file in file_lst if str(g_id) in file]
        print('checking files', good_files)
        for file in good_files:
            mk_target_dir_mv_fits_file(file, all_tois)



if __name__ == "__main__":
    # total = len(gaia_ids)
    # lst_len = np.ceil(total/5000)
    idx_str = os.environ.get("SLURM_ARRAY_TASK_ID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    print('indx string', idx_str)
    if idx_str is None:
        print("Usage: python E*organizing_fits_files.py <index>  # or SLURM_ARRAY_TASK_ID")
        sys.exit(1)
    else:
        run = int(7261*float(idx_str))
        main(gaia_ids[run:run+7261])