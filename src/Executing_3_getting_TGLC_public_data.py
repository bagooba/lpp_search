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



import warnings
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))

from Functions_all import *


# In[10]:



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
import requests


# In[12]:

def main(file_num):
    gaia_id_list = []

    # In[13]:


    sectors_south = list(range(1, 14))+list(range(27, 27+13))


    import requests

    # In[17]:


    url = 'https://archive.stsci.edu/hlsp/tglc'
    response = requests.get(url)


    # In[18]:


    import subprocess
    import requests

    soup = BeautifulSoup(response.text, 'html.parser')
    # soup_list = soup.findAll('a', href=True, class_=False, target=False)[14:-2:2]

    soup_list = soup.findAll('a', href=True, class_=False, target=False)[8:-2:2]


    # sectors_south = list(range(11, 14))#+list(range(36, 40))#+list(range(61, 70))
    sectors_all = list(range(1, 56))

    # In[25]:

    all_tois = pd.read_csv('../data/PS_2025.08.05_10.28.29.csv', skiprows = 104 )

    gaia_ids = np.unique([all_tois['gaia_id'][x][9:] for x in range(len(all_tois))])

    # In[26]:


    vvv = 0
    import sys
    import urllib.request  # the lib that handles the url stuff
    file = soup_list[3+file_num]
    sh_file_str_list = []
    print('line1', file)
    try:
        sector = int(str(file).split('_s')[-1][:4])
    except: 
        sector = -1
    print('sector ', sector )
    if sector in sectors_south:
#         gaia_ids = list(star_sector_df_GAIA[star_sector_df_GAIA['S'+str(sector)]==True]['GAIA_ID'].astype(str))
        with open(f'../sector_executable_files/run_tois_new_s{sector}.sh', 'w') as rsh:

            for line in urllib.request.urlopen(file['href']):
                vvv+=1
                line_str = str(line.decode('utf-8')) #utf-8 or iso8859-1 or whatever the page encoding scheme is

    #             print('line', line_str)
                if vvv==1:
                    sh_file_str_list.append(line_str)
                    continue
                else:
                    try:
                        sector = int(line_str.split(' ')[4].split('/')[0][2:])
                    except:
                        print(line_str)
                        continue
                    gaia_str = line_str.split('gaiaid-')[-1].split('-')[0]
                    if gaia_str in gaia_ids:
                        making_new_line_str = "../known_toi_data/"+line_str.split(' ')[4][1:]
                        lst_line_str = line_str.split(' ')
                        new_line_str = lst_line_str[0]+' '+lst_line_str[1]+' '+lst_line_str[2]+' '+lst_line_str[3]+' '+making_new_line_str+' '+lst_line_str[5]
                        print('new line', new_line_str)
                        sh_file_str_list.append(new_line_str)

            rsh.writelines(sh_file_str_list)
    
    else:
        print('not south')


    # In[ ]:


if __name__ == "__main__":
    # from schwimmbad import JoblibPool

    try:
        file_num = int(sys.argv[1])
    except ValueError:
        
        sys.exit(1)
    # file_num +=1000

    main(file_num)
    print('done')







