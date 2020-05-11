#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os

# List down file paths
dir_data = "../Processed Data/smoking-lvm-cleaned-data/final"

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
        .rename(columns={"participant": "participant_id"})
        .loc[:, ["participant_id", 
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
data_selfreport["begin_unixts"] = data_selfreport["timestamp"]/1000

def calculate_delta(message):
    sr_accptresponse = ['Smoking Event(less than 5 minutes ago)', 
                        'Smoking Event(5 - 15 minutes ago)', 
                        'Smoking Event(15 - 30 minutes ago)']
    sr_dictionary = {'Smoking Event(less than 5 minutes ago)': 2.5, 
                     'Smoking Event(5 - 15 minutes ago)': 10,
                     'Smoking Event(15 - 30 minutes ago)': 17.5} 

    if message in sr_accptresponse:
        # Convert time from minutes to seconds
        use_delta = sr_dictionary[message]*60  
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

data_selfreport["delta"] = data_selfreport["message"].apply(lambda x: calculate_delta(x))
data_selfreport["smoked_unixts"] = data_selfreport["begin_unixts"] - data_selfreport["delta"]

# Create a new variable, study_day: number of days since participant entered
# the study
data_selfreport["study_day"] = (
        data_selfreport
        .loc[:, ["start_date_unixts","smoked_unixts"]]
        .pipe(lambda x: (x["smoked_unixts"]-x["start_date_unixts"])/(60*60*24))
        .apply(lambda x: round_day(x))
)

# Create a new variable, day_since_quit: number of days before or after 
# 12AM on Quit Date
data_selfreport["day_since_quit"] = (
    data_selfreport
        .loc[:, ["quit_date_unixts","smoked_unixts"]]
        .pipe(lambda x: (x["smoked_unixts"]-x["quit_date_unixts"])/(60*60*24))
        .apply(lambda x: round_day(x))
)

# Drop columns with missing values in the smoked_unixts variable
data_selfreport = data_selfreport.dropna(how = 'any', subset=['smoked_unixts'])
data_selfreport["study_day"] = data_selfreport["study_day"].apply(lambda x: np.int(x))
data_selfreport["day_since_quit"] = data_selfreport["day_since_quit"].apply(lambda x: np.int(x))

# Create a new variable, is_post_quit: whether a given day falls before or on/after 12AM on Quit Date
data_selfreport["is_post_quit"] = data_selfreport["day_since_quit"].apply(lambda x: -1 if x < 0 else 1)

# Create a new variable, day_within_period: 
# if is_post_quit<0, number of days after 12AM on start of study
# if is_post_quit>=0, number of days after 12AM on Quit Date
# hence day_within_period is a count variable with ZERO as minimum value
data_selfreport["day_within_period"] = np.where(data_selfreport["is_post_quit"]<0,
                                                data_selfreport["study_day"], 
                                                data_selfreport["day_since_quit"])

# Finally, select subset of columns
use_these_columns = ["participant_id", "start_date_unixts", "quit_date_unixts",
                     "expected_end_date_unixts","actual_end_date_unixts",
                     "is_post_quit", "study_day", "day_since_quit", "day_within_period",
                     "begin_unixts", "message", "delta", "smoked_unixts"]
data_selfreport = data_selfreport.loc[:, use_these_columns]

#%%
###############################################################################
# Write out csv file for prepared data if write_out==True
###############################################################################
write_out = True

if write_out:
    data_selfreport.to_csv(os.path.join(os.path.realpath(dir_data), 'work_with_datapoints.csv'), index=False)

# %%
