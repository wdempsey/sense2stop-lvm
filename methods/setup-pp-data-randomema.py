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
data_random_ema = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'random-ema-final.csv'))

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

# the date variable in data_random_ema is in human-readable format
# set utc=True so that timestamp remains fixed across all machines running these scripts
# although local time of participants is in GMT-XX
data_random_ema['random_ema_time'] = (
    data_random_ema['date']
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d %H:%M:%S", utc=True))
    .apply(lambda x: np.datetime64(x))
)

#%%
data_random_ema['date'] = (
    data_random_ema['date']
    .apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    .apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d", utc=True))
    .apply(lambda x: np.datetime64(x))
)

#%%
data_random_ema = data_random_ema.loc[:, ['participant_id', 'date', 'random_ema_time','smoke','when_smoke']]
data_random_ema = pd.merge(left = data_random_ema, right = data_day_limits, how = 'left', on = ['participant_id','date'])

###############################################################################
# Exclude rows corresponding to Random EMAs delivered before start_time or
# after end_time. Additionally, exclude Random EMAs that do not have any 
# response recorded in the 'smoke' variable
###############################################################################

# Exclude rows corresponding to Random EMAs before start_time and after end_time
data_random_ema = data_random_ema[data_random_ema['random_ema_time'] >= data_random_ema['start_time']]
data_random_ema = data_random_ema[data_random_ema['random_ema_time'] <= data_random_ema['end_time']]
data_random_ema['hours_since_start_day'] = (data_random_ema['random_ema_time'] - data_random_ema['start_time'])/np.timedelta64(1,'h')

#%%
# SANITY CHECK: are there Random EMA with duplicate hours_since_start_day?
# If there are duplicates, only retain the very first record
for participant in (data_random_ema['participant_id'].unique()):
    for days in range(1,15):
        current_data_random_ema = data_random_ema[data_random_ema['participant_id']==participant]
        current_data_random_ema = current_data_random_ema[current_data_random_ema['study_day']==days]
        if len(current_data_random_ema.index>0):
            which_duplicated = current_data_random_ema['hours_since_start_day'].duplicated()
            if np.sum(which_duplicated)>0:
                print((participant, days, which_duplicated))  # no duplicates detected
                #current_data_random_ema = current_data_random_ema[~which_duplicated]
                #data_random_ema[(data_random_ema['participant_id']==participant) & (data_random_ema['study_day']==days)] = current_data_random_ema

#%%

# Retain selected columns
data_random_ema = data_random_ema.loc[:, ['participant_id','date','random_ema_time','hours_since_start_day','smoke','when_smoke']]

# Exclude rows with missing values in the variable smoke
data_random_ema = data_random_ema[~pd.isna(data_random_ema['smoke'])]

# Exclude rows having no response recorded in the 'smoke' variable 
data_random_ema = data_random_ema[~(data_random_ema['smoke']=='None')]
data_random_ema = data_random_ema[~((data_random_ema['smoke']=='Yes') & (data_random_ema['when_smoke']=='None'))]

# SANITY CHECK:
#data_random_ema.loc[:, ['participant_id','smoke','when_smoke']].groupby(['smoke','when_smoke']).count()




#%%

###############################################################################
# Recode variable that captures timing of smoking
###############################################################################

def recode_when_smoke(message):
    accept_response = ['1 - 19 Minutes', '20 - 39 Minutes', '40 - 59 Minutes','60 - 79 Minutes', '80 - 100 Minutes', '> 100 Minutes']
    recode_dictionary = {'1 - 19 Minutes': 1, '20 - 39 Minutes': 2, '40 - 59 Minutes': 3, '60 - 79 Minutes': 4, '80 - 100 Minutes': 5, '> 100 Minutes': 6} 

    if message in accept_response:
        use_value = recode_dictionary[message] 
    else:
        use_value = np.nan
    return use_value

data_random_ema['when_smoke'] = data_random_ema['when_smoke'].apply(lambda x: recode_when_smoke(x))

#%%
def calculate_delta(message):
    accept_response = [1,2,3,4,5,6]
    # delta is in hours
    #use_this_delta = {1: np.mean([1,19])/60, 2: np.mean([20,39])/60, 3: np.mean([40,59])/60, 4: np.mean([60,79])/60, 5: np.mean([80,100])/60, 6: np.nan} 
    use_this_delta = {1: np.mean([1,19])/60, 2: np.mean([20,39])/60, 3: np.mean([40,59])/60, 4: np.mean([60,79])/60, 5: np.mean([80,100])/60, 6: 100/60} 

    if pd.isna(message):
        use_value = message
    elif message in accept_response:
        use_value = use_this_delta[message] 
    else:
        use_value = pd.NA  
    return use_value

data_random_ema['delta'] = data_random_ema['when_smoke'].apply(lambda x: calculate_delta(x))

data_random_ema.index = np.array(range(0, len(data_random_ema.index)))

#%%
this_data_stream = data_random_ema
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
                     'smoke':dat['smoke'],
                     'when_smoke':dat['when_smoke'],
                     'delta':dat['delta']
                     }
                   }
        current_dict.update(new_dict)
    
    # Update participant
    all_dict.update({current_participant:current_dict})

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_random_ema')
outfile = open(filename, 'wb')
pickle.dump(all_dict, outfile)
outfile.close()

