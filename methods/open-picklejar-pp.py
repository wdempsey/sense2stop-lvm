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
# Dictionaries for latent variable models
###############################################################################

filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_random_ema')
infile = open(filename,'rb')
collect_all = pickle.load(infile)
infile.close()





