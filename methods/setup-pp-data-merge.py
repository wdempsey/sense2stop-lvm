#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']

#%%
###############################################################################
# Knit together dictionaries from various data streams
###############################################################################



