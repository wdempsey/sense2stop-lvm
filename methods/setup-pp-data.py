#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

# List down file paths
#dir_data = "../smoking-lvm-cleaned-data/final"

exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']

# Read in data
data_dates = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'participant-dates.csv'))
data_selfreport = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'self-report-smoking-final.csv'))

#%%

###############################################################################
# Data preparation: data_dates data frame
###############################################################################
# Create unix timestamps corresponding to 12AM of a given human-readable date
data_dates["start_date_unixts"] = (
data_dates["start_date"]
    .apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    .apply(lambda x: datetime.timestamp(x))
)

data_dates["quit_date_unixts"] = (
data_dates["quit_date"]
    .apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    .apply(lambda x: datetime.timestamp(x))
)

data_dates["expected_end_date_unixts"] = (
data_dates["expected_end_date"]
    .apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    .apply(lambda x: datetime.timestamp(x))
)

data_dates["actual_end_date_unixts"] = (
data_dates["actual_end_date"]
    .apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    .apply(lambda x: datetime.timestamp(x))
)

# More tidying up
data_dates = (
    data_dates
        .rename(columns={"participant": "participant_id", 
                         "quit_date": "quit_date_hrts",
                         "start_date": "start_date_hrts",
                         "actual_end_date": "actual_end_date_hrts",
                         "expected_end_date": "expected_end_date_hrts"})
        .loc[:, ["participant_id", 
                 "start_date_hrts","quit_date_hrts",
                 "expected_end_date_hrts", "actual_end_date_hrts",
                 "start_date_unixts", "quit_date_unixts",
                 "expected_end_date_unixts","actual_end_date_unixts"]]
)

#%%

###############################################################################
# Merge data_selfreport with data_dates
###############################################################################
data_selfreport = data_dates.merge(data_selfreport, 
                                   how = 'left', 
                                   on = 'participant_id')

#%%

###############################################################################
# Data preparation: data_selfreport data frame
###############################################################################
# Drop the participants labelled 10X as they are pilot individuals
data_selfreport = data_selfreport.dropna(how = 'any', subset=['hour'])

def calculate_delta(message):
    sr_accptresponse = ['Smoking Event(less than 5 minutes ago)', 
                        'Smoking Event(5 - 15 minutes ago)', 
                        'Smoking Event(15 - 30 minutes ago)',
                        'Smoking Event(more than 30 minutes ago)']
    sr_dictionary = {'Smoking Event(less than 5 minutes ago)': 1, 
                     'Smoking Event(5 - 15 minutes ago)': 2,
                     'Smoking Event(15 - 30 minutes ago)': 3,
                     'Smoking Event(more than 30 minutes ago)': 4} 

    if message in sr_accptresponse:
        # Convert time from minutes to seconds
        use_delta = sr_dictionary[message] 
    else:
        # If participant reported smoking more than 30 minutes ago,
        # then we consider time s/he smoked as missing
        use_delta = pd.NA  
    return use_delta

def round_day(raw_day):
    if pd.isna(raw_day):
        # Missing values for raw_day can occur
        # if participant reported smoking more than 30 minutes ago
        out_day = pd.NA
    else:
        # This takes care of the instances when participant reported to smoke 
        # less than 30 minutes ago
        if raw_day >= 0:
            # If on or after Quit Date, round down to the nearest integer
            # e.g., floor(2.7)=2
            out_day = np.floor(raw_day)
        else:
            # If before Quit Date, round up to the nearest integer
            # e.g., ceil(-2.7)=-2
            out_day = np.ceil(raw_day)
        
    return out_day

#%%
data_selfreport['date'] = pd.to_datetime(data_selfreport.date)
data_selfreport['start_date'] = pd.to_datetime(data_selfreport.start_date_hrts)
data_selfreport['quit_date'] = pd.to_datetime(data_selfreport.quit_date_hrts)
data_selfreport["delta"] = data_selfreport["message"].apply(lambda x: calculate_delta(x))

# Create a new variable, study_day: number of days since participant entered
# the study
data_selfreport['study_day'] = (data_selfreport['date'] - data_selfreport['start_date']).dt.days

# Create a new variable, day_since_quit: number of days before or after 
# 12AM on Quit Date
data_selfreport['day_since_quit'] = (data_selfreport['date'] - data_selfreport['quit_date']).dt.days

# Create a new variable, is_post_quit: whether a given day falls before or on/after 12AM on Quit Date
data_selfreport["is_post_quit"] = data_selfreport["day_since_quit"].apply(lambda x: 0 if x < 0 else 1)

# Create a new variable, day_within_period: 
# if is_post_quit<0, number of days after 12AM on start of study
# if is_post_quit>=0, number of days after 12AM on Quit Date
# hence day_within_period is a count variable with ZERO as minimum value
data_selfreport["day_within_period"] = np.where(data_selfreport["is_post_quit"]==0,
                                                data_selfreport["study_day"], 
                                                data_selfreport["day_since_quit"])

# Number of hours elapsed since the beginning of the study
data_selfreport['hours_since_start_of_study'] = (data_selfreport['date'] - data_selfreport['start_date'])/np.timedelta64(1,'h')


#%%
# Get number of hours elapsed between two self-reported smoking events
data_selfreport['actual_end_date_hrts'] = pd.to_datetime(data_selfreport['actual_end_date_hrts'])
data_selfreport['time_to_actual_end_date'] = (data_selfreport.actual_end_date_hrts - data_selfreport.date) / np.timedelta64(1,'m') + 720 # Add 720 minutes to deal with quit date you can provide data still.
data_selfreport = data_selfreport.sort_values(['participant_id','date'])
data_selfreport['time_to_next_event'] = data_selfreport.groupby("participant_id").date.diff().shift(-1)/np.timedelta64(1,'m')

#%%
# For NaN, time_to_next_event is the time until actual quit date.
# These should be treated as censored times  
data_selfreport["censored"] = data_selfreport["time_to_next_event"].isnull()

for index in np.where(data_selfreport.censored==True):
    temp = data_selfreport['time_to_actual_end_date'].iloc[index]
    data_selfreport['time_to_next_event'].iloc[index] = temp

#%%
# Create features
data_selfreport['date'] = pd.to_datetime(data_selfreport['date'])
data_selfreport['start_date_hrts'] = pd.to_datetime(data_selfreport['start_date_hrts'])
data_selfreport['quit_date_hrts'] = pd.to_datetime(data_selfreport['quit_date_hrts'])
data_selfreport['ones']=1

#%%

# History of smoking variables: within day ------------------------------------
# Whether the current SR is the first within a given day
data_selfreport["order_within_day"] = (
    data_selfreport
        .groupby(["participant_id","study_day"])["ones"]
        .cumsum()
)
data_selfreport["is_first_sr_within_day"] = np.where(data_selfreport["order_within_day"]==1,1,0)

# The number of hours elapsed since beginning of the pre- or post-quit period
data_selfreport["hours_since_previous_sr_within_day"] = (
    data_selfreport
        .groupby(["participant_id","study_day"])["date"]
        .diff()
        .shift(0)/np.timedelta64(1,'h')
)

# History of smoking variables: within period ---------------------------------
# Whether the current SR is the first within pre- or post-quit period
data_selfreport["ones"]=1
data_selfreport["order_within_period"] = (
    data_selfreport
        .groupby(["participant_id","is_post_quit"])["ones"]
        .cumsum()
)
data_selfreport["is_first_sr_within_period"] = np.where(data_selfreport["order_within_period"]==1,1,0)

# The number of hours elapsed since previous SR within pre- or post-quit periods
data_selfreport["hours_since_previous_sr_within_period"] = (
    data_selfreport
        .groupby(["participant_id","is_post_quit"])["date"]
        .diff()
        .shift(0)/np.timedelta64(1,'h')
)

#%%

# Time variables: relative to quit day ----------------------------------------
data_selfreport['hours_relative_quit'] = np.where(data_selfreport["is_post_quit"]==0,
                                                  data_selfreport["quit_date_hrts"] - data_selfreport["date"],
                                                  data_selfreport["date"] - data_selfreport["quit_date_hrts"])
# This variable captures the period around (just before or after) quit date
data_selfreport['hours_relative_quit'] = data_selfreport['hours_relative_quit']/np.timedelta64(1,'h')

data_selfreport['is_within24hours_quit'] = np.where(data_selfreport['hours_relative_quit']<=24,1,0)
data_selfreport['is_within48hours_quit'] = np.where(data_selfreport['hours_relative_quit']<=48,1,0)
data_selfreport['is_within72hours_quit'] = np.where(data_selfreport['hours_relative_quit']<=72,1,0)

# This variable: within-day temporal dynamics ---------------------------------
data_selfreport["hour_of_day"] = data_selfreport["hour"] + data_selfreport["minute"]/60
data_selfreport["sleep"] = np.where((data_selfreport['hour_of_day']>=1).bool and (data_selfreport['hour_of_day']<=6).bool,1,0)

#%%
# Time variables: relative to beginning of pre- or post-quit periods ----------
# The number of hours elapsed since beginning of the pre- or post-quit period
data_selfreport['hours_since_start_of_period'] = np.where(data_selfreport["is_post_quit"]==0,
                                                          data_selfreport["date"] - data_selfreport["start_date_hrts"],
                                                          data_selfreport["date"] - data_selfreport["quit_date_hrts"])

data_selfreport['hours_since_start_of_period'] = data_selfreport['hours_since_start_of_period']/np.timedelta64(1,'h')

#%%
# Clean up time elapsed variables: set time elapsed for first SR within period/day
data_selfreport["hours_since_previous_sr_within_day"] = np.where(data_selfreport["order_within_day"]==1, 
                                                                 data_selfreport["hour_of_day"], 
                                                                 data_selfreport["hours_since_previous_sr_within_day"])

                                                                 
data_selfreport["hours_since_previous_sr_within_period"] = np.where(data_selfreport["order_within_period"]==1, 
                                                                    data_selfreport["hours_since_start_of_period"], 
                                                                    data_selfreport["hours_since_previous_sr_within_period"])

#%%
# Finally, select subset of columns
use_these_columns = ["participant_id",
                     "start_date_hrts", "quit_date_hrts",
                     "expected_end_date_hrts","actual_end_date_hrts", 
                     "date","hours_since_start_of_study",
                     "is_post_quit", "study_day", "day_since_quit", "day_within_period",
                     "message", "delta", "time_to_next_event","censored",
                     "order_within_day", "order_within_period",
                     "is_first_sr_within_day", "is_first_sr_within_period", 
                     "hours_since_previous_sr_within_day", "hours_since_previous_sr_within_period",
                     "hours_since_start_of_period",
                     "is_within24hours_quit","is_within48hours_quit","is_within72hours_quit",
                     "hour_of_day", "sleep"]
data_selfreport = data_selfreport.loc[:, use_these_columns]

data_selfreport.to_csv(os.path.join(os.path.realpath(dir_data), 'work_with_datapoints.csv'), index=False)

