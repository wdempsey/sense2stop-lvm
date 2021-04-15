#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt

exec(open('../../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']
dir_code_methods = os.environ['dir_code_methods']

###############################################################################
# Create a data frame with records of start & end of day timestamps
# for each participant-day
###############################################################################

#%%
# Read in data
data_hq_episodes = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'hq-episodes-final.csv'))

# data_hq_episodes contains the start & end timestampfor each day, but also contains other info we do not need for now
data_day_limits = (
    data_hq_episodes
    .drop_duplicates(subset=['id','study_day','start_time','end_time'])
    .loc[:,['id','date','study_day','prequit','start_time','end_time']]
)

#%%
def CheckFormat(x):
    try: 
        out = datetime.strptime(x,'%Y-%m-%d %H:%M:%S%z') 
    except:
        out = datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f%z')
    return out

# the time variables below are in human-readable format
# set utc=True so that timestamp remains fixed across all machines running these scripts
# although local time of participants is in GMT-XX
data_day_limits['date'] = (
    data_day_limits['date']
    .apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
    .apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d", utc=True))
    .apply(lambda x: np.datetime64(x))
)

data_day_limits['start_time'] = (
    data_day_limits['start_time']
    .apply(lambda x: CheckFormat(x))
    .apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d %H:%M:%S", utc=True))
    .apply(lambda x: np.datetime64(x))
    .apply(lambda x: x - np.timedelta64(10,'m'))  # adjust start of day to be 10 minutes earlier
)

data_day_limits['end_time'] = (
    data_day_limits['end_time']
    .apply(lambda x: CheckFormat(x))
    .apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d %H:%M:%S", utc=True))
    .apply(lambda x: np.datetime64(x))
)

# Some end times fall right after midnight but dates have not been adjusted to fall on the next day
# here, we adjust those dates to fall on the next calendar day
data_day_limits['end_time'] = np.where(data_day_limits['start_time'] > data_day_limits['end_time'], data_day_limits['end_time']+np.timedelta64(1,'D'), data_day_limits['end_time'])

# Now, calculate the length (in number of hours) of each participant day
data_day_limits['day_length'] = (data_day_limits['end_time'] - data_day_limits['start_time'])/np.timedelta64(1,'h')

# Make adjustment for rows with day_length==0
data_day_limits['end_time'] = np.where(data_day_limits['day_length']==0, data_day_limits['end_time']+np.timedelta64(1,'D'), data_day_limits['end_time'])
data_day_limits['day_length'] = (data_day_limits['end_time'] - data_day_limits['start_time'])/np.timedelta64(1,'h')

# Rename columns to prepare for merging with other data frame
data_day_limits = data_day_limits.rename(columns = {"id":"participant_id"})
data_day_limits = data_day_limits.loc[:, ['participant_id', 'date', 'study_day','day_length','start_time','end_time']]


# %%
filename = os.path.join(os.path.realpath(dir_picklejar), 'data_day_limits')
outfile = open(filename, 'wb')
pickle.dump(data_day_limits, outfile)
outfile.close()

