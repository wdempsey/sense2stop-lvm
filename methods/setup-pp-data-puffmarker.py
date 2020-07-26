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
dir_code_methods = os.environ['dir_code_methods']

#%%
# Read in data
data_dates = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'participant-dates.csv'))
data_hq_episodes = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'hq-episodes-final.csv'))
data_puffmarker = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'puff-episode-final.csv'))

###############################################################################
# Create a data frame with records of start & end of day timestamps
# for each participant-day
###############################################################################

# output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'setup-day-limits.py')).read())

#%%

###############################################################################
# Begin to work with the puffmarker data
# First, ensure dates are properly formatted
# Then, merge data frame containing start & end of day timestamps with
# the data frame containing puffmarker timestamps
###############################################################################

# the date variable in data_puffmarker is in human-readable format
# set utc=True so that timestamp remains fixed across all machines running these scripts
# although local time of participants is in GMT-XX
data_puffmarker['puffmarker_time'] = (
    data_puffmarker['date']
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d %H:%M:%S", utc=True))
    .apply(lambda x: np.datetime64(x))
)

#%%
data_puffmarker['date'] = (
    data_puffmarker['date']
    .apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    .apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d", utc=True))
    .apply(lambda x: np.datetime64(x))
)

#%%
data_puffmarker = data_puffmarker.loc[:, ['participant_id', 'date', 'puffmarker_time']]
data_puffmarker = pd.merge(left = data_puffmarker, right = data_day_limits, how = 'left', on = ['participant_id','date'])

#%%
# Exclude rows corresponding to puffmarker timestamps before start_time and after end_time
data_puffmarker = data_puffmarker[data_puffmarker['puffmarker_time'] >= data_puffmarker['start_time']]
data_puffmarker = data_puffmarker[data_puffmarker['puffmarker_time'] <= data_puffmarker['end_time']]
data_puffmarker['hours_since_start_day'] = (data_puffmarker['puffmarker_time'] - data_puffmarker['start_time'])/np.timedelta64(1,'h')
data_puffmarker = data_puffmarker.loc[:, ['participant_id','date','puffmarker_time','hours_since_start_day']]
data_puffmarker.index = np.array(range(0, len(data_puffmarker.index)))

#%%

this_data_stream = data_puffmarker
all_participant_id = data_hq_episodes['id'].drop_duplicates()
all_participant_id.index = np.array(range(0,len(all_participant_id.index)))
all_dict = {}

#%%
for i in range(0, len(all_participant_id)):
    current_dict = {}
    current_participant = all_participant_id[i]
    current_participant_day_limits = data_day_limits[data_day_limits['participant_id'] == current_participant]
    current_participant_data_stream = this_data_stream[this_data_stream['participant_id'] == current_participant]

    # Within a participant, go through each day
    for j in range(0, len(current_participant_day_limits.index)):
        # Grab rows corresponding to data observed only on this_date
        this_date = current_participant_day_limits['date'].iloc[j]
        dat = current_participant_data_stream[current_participant_data_stream['date']==this_date]

        # Other information
        this_study_day = current_participant_day_limits['study_day'].iloc[j]
        this_day_length = current_participant_day_limits['day_length'].iloc[j]

        # Combine information into a dictionary
        new_dict = {this_study_day:
                    {'participant_id':current_participant,
                     'study_day':this_study_day, 
                     'day_length': this_day_length, 
                     'hours_since_start_day':dat['hours_since_start_day']
                     }
                    }
        current_dict.update(new_dict)
    
    # Update participant
    all_dict.update({current_participant:current_dict})

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_puffmarker')
outfile = open(filename, 'wb')
pickle.dump(all_dict, outfile)
outfile.close()

