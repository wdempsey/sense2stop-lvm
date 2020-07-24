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
# Read in data
data_dates = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'participant-dates.csv'))
data_hq_episodes = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'hq-episodes-final.csv'))
data_selfreport = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'self-report-smoking-final.csv'))

#%%

