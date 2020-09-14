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
data_selfreport = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'self-report-smoking-final.csv'))

#%%
# Drop duplicate records from data_selfreport
data_selfreport = data_selfreport.drop_duplicates()

#%%

###############################################################################
# Create a data frame with records of start & end of day timestamps
# for each participant-day
###############################################################################

# output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'setup-day-limits.py')).read())

#%%

###############################################################################
# Begin to work with the Random EMA data
# First, ensure dates are properly formatted
# Then, merge data frame containing start & end of day timestamps with
# the data frame containing Random EMA timestamps
###############################################################################

# the date variable in data_selfreport is in human-readable format
# set utc=True so that timestamp remains fixed across all machines running these scripts
# although local time of participants is in GMT-XX
data_selfreport['selfreport_time'] = (
    data_selfreport['date']
    .apply(lambda x: pd.to_datetime(x, format = "%m/%d/%Y %H:%M", utc=True))  # note difference between format of selfreport timestamp and random ema & puffmarker timestamp
    .apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    .apply(lambda x: np.datetime64(x))
)

data_selfreport['date'] = (
    data_selfreport['date']
    .apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))
    .apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d", utc=True))
    .apply(lambda x: np.datetime64(x))
)

#%%
data_selfreport = data_selfreport.loc[:, ['participant_id', 'date', 'selfreport_time','message']]
data_selfreport = pd.merge(left = data_selfreport, right = data_day_limits, how = 'left', on = ['participant_id','date'])

# Exclude rows corresponding to Random EMAs before start_time and after end_time
data_selfreport = data_selfreport[data_selfreport['selfreport_time'] >= data_selfreport['start_time']]
data_selfreport = data_selfreport[data_selfreport['selfreport_time'] <= data_selfreport['end_time']]
data_selfreport['hours_since_start_day'] = (data_selfreport['selfreport_time'] - data_selfreport['start_time'])/np.timedelta64(1,'h')

#%%
# SANITY CHECK: are there Self-Reports with duplicate hours_since_start_day?
# If there are duplicates, only retain the very first record
for participant in (data_selfreport['participant_id'].unique()):
    for days in range(1,15):
        current_data_selfreport = data_selfreport[data_selfreport['participant_id']==participant]
        current_data_selfreport = current_data_selfreport[current_data_selfreport['study_day']==days]
        if len(current_data_selfreport.index>0):
            which_duplicated = current_data_selfreport['hours_since_start_day'].duplicated()
            if np.sum(which_duplicated)>0:
                #print((participant, days, which_duplicated))
                current_data_selfreport = current_data_selfreport[~which_duplicated]
                data_selfreport[(data_selfreport['participant_id']==participant) & (data_selfreport['study_day']==days)] = current_data_selfreport

# SANITY CHECK:
#data_selfreport['message'].value_counts()
#%%
###############################################################################
# Recode variable that captures timing of smoking
###############################################################################

def recode_message(message):
    accept_response = ['Smoking Event(less than 5 minutes ago)', 
                        'Smoking Event(5 - 15 minutes ago)', 
                        'Smoking Event(15 - 30 minutes ago)',
                        'Smoking Event(more than 30 minutes ago)']

    recode_dictionary = {'Smoking Event(less than 5 minutes ago)': 1, 
                     'Smoking Event(5 - 15 minutes ago)': 2,
                     'Smoking Event(15 - 30 minutes ago)': 3,
                     'Smoking Event(more than 30 minutes ago)': 4}

    if message in accept_response:
        use_value = recode_dictionary[message] 
    else:
        use_value = np.nan
    return use_value

data_selfreport['message'] = data_selfreport['message'].apply(lambda x: recode_message(x))

#%%
def calculate_delta(message):
    accept_response = [1,2,3,4]
    # delta is in hours
    #use_this_delta = {1: np.mean([0,5])/60, 2: np.mean([5,15])/60, 3: np.mean([15,30])/60, 4: np.nan} 
    use_this_delta = {1: np.mean([0,5])/60, 2: np.mean([5,15])/60, 3: np.mean([15,30])/60, 4: 30/60} 

    if pd.isna(message):
        use_value = message
    elif message in accept_response:
        use_value = use_this_delta[message] 
    else:
        use_value = pd.NA  
    return use_value

data_selfreport['delta'] = data_selfreport['message'].apply(lambda x: calculate_delta(x))

data_selfreport.index = np.array(range(0, len(data_selfreport.index)))

#%%

this_data_stream = data_selfreport
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
                     'hours_since_start_day':dat['hours_since_start_day'],
                     'message':dat['message'],
                     'delta':dat['delta']
                     }
                   }
        current_dict.update(new_dict)
    
    # Update participant
    all_dict.update({current_participant:current_dict})

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_selfreport')
outfile = open(filename, 'wb')
pickle.dump(all_dict, outfile)
outfile.close()


