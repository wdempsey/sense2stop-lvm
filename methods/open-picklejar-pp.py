#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt

exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']

#%%
###############################################################################
# Create a data frame with records of start & end of day timestamps
# for each participant-day
###############################################################################
dir_code_methods = os.environ['dir_code_methods']

# output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'setup-day-limits.py')).read())


###############################################################################
# Dictionaries for latent variable models
###############################################################################

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_selfreport')
infile = open(filename,'rb')
dict_selfreport = pickle.load(infile)
infile.close()

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_random_ema')
infile = open(filename,'rb')
dict_random_ema = pickle.load(infile)
infile.close()

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_puffmarker')
infile = open(filename,'rb')
dict_puffmarker = pickle.load(infile)
infile.close()

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_knitted')
infile = open(filename,'rb')
dict_knitted = pickle.load(infile)
infile.close()

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_knitted_with_puffmarker')
infile = open(filename,'rb')
dict_knitted_with_puffmarker = pickle.load(infile)
infile.close()

