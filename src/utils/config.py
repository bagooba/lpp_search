import pandas as pd

G = 2941.18330364  # R_s^3 M_s^-1 day^−2
MODEL_PATH = '/users/malharris/lpp_search/model_TESS.pth'
MDWARF_CATALOG = '/users/malharris/lpp_search/data/final_mdwarf_params.csv'

LDC_for_quadratic = pd.read_csv('/users/malharris/lpp_search/data/LDC_params/table15.dat', 
                                header = None, 
                                sep="\s+", index_col=None,
                               names = ['logg', 'Teff', 'z','L/HP', 'aLSM', 'bLSM',
                                       'aFCM', 'bFCM', 'SQRT(CHI2)', 'qsr', 'PC'])

LDC_PARAMS_MDWARF = LDC_for_quadratic[LDC_for_quadratic['Teff']<4300]
                  
