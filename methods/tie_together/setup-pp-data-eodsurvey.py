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

#%%
# Read in data
data_dates = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'participant-dates.csv'))
data_hq_episodes = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'hq-episodes-final.csv'))
data_eodsurvey = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'eod-ema-final.csv'))

#%%
# output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'tie_together', 'setup-day-limits.py')).read())

data_day_limits['start_time_hour_of_day'] = data_day_limits['start_time'].apply(lambda x: x.hour + (x.minute)/60 + (x.second)/3600) 
data_day_limits['end_time_hour_of_day'] = data_day_limits['end_time'].apply(lambda x: x.hour + (x.minute)/60 + (x.second)/3600)


# %%
data_eodsurvey['eod_survey_time'] = (
    data_eodsurvey['date']
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d %H:%M:%S", utc=True))  
    .apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    .apply(lambda x: np.datetime64(x))
)

# %%
data_eodsurvey['date'] = (
    data_eodsurvey['date']
    .apply(lambda x: pd.to_datetime(x, format = "%Y-%m-%d %H:%M:%S", utc=True))  
    .apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
    .apply(lambda x: np.datetime64(x))
)

# %%
data_eodsurvey = pd.merge(left = data_eodsurvey, right = data_day_limits, how = 'left', on = ['participant_id','date'])

# %%
data_eodsurvey['rescaled_8to9'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=8, minute=0, second=0))   
data_eodsurvey['rescaled_8to9'] = (data_eodsurvey['rescaled_8to9'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_8to9'] = np.where(data_eodsurvey['8to9'] != 1.0, np.nan, data_eodsurvey['rescaled_8to9'])

# %%
data_eodsurvey['rescaled_9to10'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=9, minute=0, second=0))   
data_eodsurvey['rescaled_9to10'] = (data_eodsurvey['rescaled_9to10'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_9to10'] = np.where(data_eodsurvey['9to10'] != 1.0, np.nan, data_eodsurvey['rescaled_9to10'])

# %%
data_eodsurvey['rescaled_10to11'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=10, minute=0, second=0))   
data_eodsurvey['rescaled_10to11'] = (data_eodsurvey['rescaled_10to11'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_10to11'] = np.where(data_eodsurvey['10to11'] != 1.0, np.nan, data_eodsurvey['rescaled_10to11'])

# %%
data_eodsurvey['rescaled_11to12'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=11, minute=0, second=0))   
data_eodsurvey['rescaled_11to12'] = (data_eodsurvey['rescaled_11to12'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_11to12'] = np.where(data_eodsurvey['11to12'] != 1.0, np.nan, data_eodsurvey['rescaled_11to12'])

# %%
data_eodsurvey['rescaled_12to13'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=12, minute=0, second=0))   
data_eodsurvey['rescaled_12to13'] = (data_eodsurvey['rescaled_12to13'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_12to13'] = np.where(data_eodsurvey['12to13'] != 1.0, np.nan, data_eodsurvey['rescaled_12to13'])

# %%
data_eodsurvey['rescaled_13to14'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=13, minute=0, second=0))   
data_eodsurvey['rescaled_13to14'] = (data_eodsurvey['rescaled_13to14'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_13to14'] = np.where(data_eodsurvey['13to14'] != 1.0, np.nan, data_eodsurvey['rescaled_13to14'])

# %%
data_eodsurvey['rescaled_14to15'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=14, minute=0, second=0))   
data_eodsurvey['rescaled_14to15'] = (data_eodsurvey['rescaled_14to15'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_14to15'] = np.where(data_eodsurvey['14to15'] != 1.0, np.nan, data_eodsurvey['rescaled_14to15'])

# %%
data_eodsurvey['rescaled_15to16'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=15, minute=0, second=0))   
data_eodsurvey['rescaled_15to16'] = (data_eodsurvey['rescaled_15to16'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_15to16'] = np.where(data_eodsurvey['15to16'] != 1.0, np.nan, data_eodsurvey['rescaled_15to16'])

# %%
data_eodsurvey['rescaled_16to17'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=16, minute=0, second=0))   
data_eodsurvey['rescaled_16to17'] = (data_eodsurvey['rescaled_16to17'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_16to17'] = np.where(data_eodsurvey['16to17'] != 1.0, np.nan, data_eodsurvey['rescaled_16to17'])

# %%
data_eodsurvey['rescaled_17to18'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=17, minute=0, second=0))   
data_eodsurvey['rescaled_17to18'] = (data_eodsurvey['rescaled_17to18'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_17to18'] = np.where(data_eodsurvey['17to18'] != 1.0, np.nan, data_eodsurvey['rescaled_17to18'])

# %%
data_eodsurvey['rescaled_18to19'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=18, minute=0, second=0))   
data_eodsurvey['rescaled_18to19'] = (data_eodsurvey['rescaled_18to19'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_18to19'] = np.where(data_eodsurvey['18to19'] != 1.0, np.nan, data_eodsurvey['rescaled_18to19'])

# %%
data_eodsurvey['rescaled_19to20'] = data_eodsurvey['start_time'].apply(lambda x: x.replace(hour=19, minute=0, second=0))   
data_eodsurvey['rescaled_19to20'] = (data_eodsurvey['rescaled_19to20'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey['rescaled_19to20'] = np.where(data_eodsurvey['19to20'] != 1.0, np.nan, data_eodsurvey['rescaled_19to20'])

# %%
eod_tot_rows = data_eodsurvey.shape[0]

# %%
data_eodsurvey['ticked_box_raw'] = np.repeat({'ticked_box_raw':np.nan}, repeats = eod_tot_rows)
data_eodsurvey['ticked_box_scaled'] = np.repeat({'ticked_box_scaled':np.nan}, repeats = eod_tot_rows)

# %%

for idx_row in range(0, eod_tot_rows):
    curr_array_raw = []
    curr_array = []

    if data_eodsurvey['status'].iloc[idx_row] == 'COMPLETED':
        curr_box = data_eodsurvey['rescaled_8to9'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([8])

        curr_box = data_eodsurvey['rescaled_9to10'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([9])

        curr_box = data_eodsurvey['rescaled_10to11'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([10])

        curr_box = data_eodsurvey['rescaled_11to12'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([11])

        curr_box = data_eodsurvey['rescaled_12to13'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([12])

        curr_box = data_eodsurvey['rescaled_13to14'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([13])

        curr_box = data_eodsurvey['rescaled_14to15'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([14])

        curr_box = data_eodsurvey['rescaled_15to16'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([15])

        curr_box = data_eodsurvey['rescaled_16to17'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([16])

        curr_box = data_eodsurvey['rescaled_17to18'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([17])

        curr_box = data_eodsurvey['rescaled_18to19'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([18])

        curr_box = data_eodsurvey['rescaled_19to20'].iloc[idx_row]
        if (~np.isnan(curr_box)) and (curr_box>=0):
            curr_array.extend([curr_box])
            curr_array_raw.extend([19])
    
    data_eodsurvey['ticked_box_raw'].iloc[idx_row] = np.array(curr_array_raw)
    data_eodsurvey['ticked_box_scaled'].iloc[idx_row] = np.array(curr_array)

# %%
data_eodsurvey['hours_since_start_day'] = (data_eodsurvey['eod_survey_time'] - data_eodsurvey['start_time'])/np.timedelta64(1,'h')
data_eodsurvey = data_eodsurvey.loc[:, ['participant_id', 'date', 'hours_since_start_day', 'ticked_box_raw','ticked_box_scaled']]

# %%
data_day_limits = pd.merge(left = data_day_limits, right = data_eodsurvey, how = 'left', on = ['participant_id','date'])

this_data_stream = data_day_limits
all_participant_id = data_day_limits['participant_id'].drop_duplicates()
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

        if np.isnan(dat['hours_since_start_day'].iloc[0]):
            # Combine information into a dictionary
            new_dict = {this_study_day: 
                        {'participant_id':current_participant,
                        'study_day':this_study_day, 
                        'day_length': this_day_length, 
                        'start_time_hour_of_day':current_participant_day_limits['start_time_hour_of_day'].iloc[j],
                        #'end_time_hour_of_day':current_participant_day_limits['end_time_hour_of_day'].iloc[j],
                        'assessment_begin':np.array([]),
                        'ticked_box_scaled':np.array([]),
                        'ticked_box_raw':np.array([])
                        }
                    }
        else:
            # Combine information into a dictionary
            new_dict = {this_study_day: 
                        {'participant_id':current_participant,
                        'study_day':this_study_day, 
                        'day_length': this_day_length, 
                        'start_time_hour_of_day':current_participant_day_limits['start_time_hour_of_day'].iloc[j],
                        #'end_time_hour_of_day':current_participant_day_limits['end_time_hour_of_day'].iloc[j],
                        'assessment_begin':np.array([dat['hours_since_start_day'].iloc[0]]),
                        'ticked_box_scaled':dat['ticked_box_scaled'].iloc[0],
                        'ticked_box_raw':dat['ticked_box_raw'].iloc[0]
                        }
                    }

        current_dict.update(new_dict)
    
    # Update participant
    all_dict.update({current_participant:current_dict})

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_eod_survey')
outfile = open(filename, 'wb')
pickle.dump(all_dict, outfile)
outfile.close()

