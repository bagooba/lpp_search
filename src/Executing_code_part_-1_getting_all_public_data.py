#!/usr/bin/env python
# coding: utf-8

# In[9]:


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


# import eleanor

import warnings
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))

from Functions_all import *


# In[10]:


import requests

import time
from bs4 import BeautifulSoup
import math
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import shutil
import datetime
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
    from urllib.parse import urlparse
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve
    from urlparse import urlparse
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))

from Functions_all import *


# In[11]:


import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import math
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import shutil
import datetime
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt


# In[12]:


# gaia_id_list = []
# for target in glob.glob('./Search_target_data/target_*'):
#     gaiaID = str(target.split('/')[-1].split('_')[2].split('-')[-1])
#     gaia_id_list.append(gaiaID)


# # In[13]:


# star_sector_csv = glob.glob('./mdwarfs_ticLists_southSectors/mdwarfs_s'+str(1).zfill(2)+'.csv')[0]
# star_sector_df = pd.read_csv(star_sector_csv, header = None)#
# star_sector_df.columns = star_sector_df.iloc[-1].reset_index(drop=True)

# # Drop the last row which is now the header
# star_sector_df = star_sector_df.reset_index(drop=True).drop(star_sector_df.index[-1])
# star_sector_df = star_sector_df.rename_axis(None, axis=1)


# # In[14]:


# star_sector_df


# # In[15]:


# star_sector_df_GAIA= star_sector_df[star_sector_df.GAIA_ID.astype(str).isin(gaia_id_list)].reset_index(drop=True)


# # In[16]:


# star_sector_df_GAIA = star_sector_df_GAIA[star_sector_df_GAIA['S1']==True].reset_index(drop=True)


# # In[17]:

def main(file):

#     url = 'https://archive.stsci.edu/hlsp/tglc'
#     response = requests.get(url)


#     # In[18]:


#     import subprocess
#     import requests

#     soup = BeautifulSoup(response.text, 'html.parser')
#     # soup_list = soup.findAll('a', href=True, class_=False, target=False)[14:-2:2]

#     # soup = BeautifulSoup(response.text, 'html.parser')
#     soup_list = soup.findAll('a', href=True, class_=False, target=False)[14:-2:2]#[8:-2:2]

    # sectors_south = list(range(11, 14))#+list(range(36, 40))#+list(range(61, 70))
    sectors_all = list(range(1, 56))

    # In[25]:

    all_tois = pd.read_csv('../data/PS_2025.11.17_10.16.43.csv', skiprows = 105)#.dropna()

    gaia_ids = [all_tois['gaia_dr3_id'][x][9:] for x in range(len(all_tois)) if type(x) == str]

    # In[26]:


    vvv = 0
    import sys
    import urllib.request  # the lib that handles the url stuff
    with open ('run_tois_.sh', 'w') as rsh:
        sh_file_str_list = []

        sector = int(str(file).split('_s')[-1][:4])
        print('sector ', sector )
#         gaia_ids = list(star_sector_df_GAIA[star_sector_df_GAIA['S'+str(sector)]==True]['GAIA_ID'].astype(str))

        for line in urllib.request.urlopen(file['href']):
            vvv+=1
            line_str = str(line.decode('utf-8')) #utf-8 or iso8859-1 or whatever the page encoding scheme is

            if vvv==1:
                sh_file_str_list.append(line_str)
                continue
            else:
                if vvv%1E5 == 0:
                    print(vvv/1E5)
                try:
                    sector = int(line_str.split(' ')[4].split('/')[0][2:])
                except:
                    print('error', line_str)
                    continue
                gaia_str = line_str.split('gaiaid-')[-1].split('-')[0]
                if gaia_str in gaia_ids:
                    making_new_line_str = "../known_toi_data/"+line_str.split(' ')[4][1:]
                    lst_line_str = line_str.split(' ')
                    new_line_str = lst_line_str[0]+' '+lst_line_str[1]+' '+lst_line_str[2]+' '+lst_line_str[3]+' '+making_new_line_str+' '+lst_line_str[5]
                    print('abs', new_line_str)
                    sh_file_str_list.append(new_line_str)

        rsh.writelines(sh_file_str_list)

    #     else:
    #         break


# In[ ]:





    
if __name__ == "__main__":
    try:
        file_num = int(sys.argv[1])
    except ValueError:
        
        sys.exit(1)
    # file_num +=1000

    url = 'https://archive.stsci.edu/hlsp/tglc'
    response = requests.get(url)


    # In[18]:


    import subprocess
    import requests

    soup = BeautifulSoup(response.text, 'html.parser')
    # soup_list = soup.findAll('a', href=True, class_=False, target=False)[14:-2:2]

    # soup = BeautifulSoup(response.text, 'html.parser')
    soup_list = soup.findAll('a', href=True, class_=False, target=False)[14:-2:2]#[8:-2:2]

    # factor_files = target_files[factor_files_min:factor_files_max]

    main(soup_list[file_num])
    time_end = tm.time()
    print('time it took: ', (time_end-time1)/60, ' minutes')
    gc.collect()





