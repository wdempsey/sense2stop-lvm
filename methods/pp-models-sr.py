#%%
import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

# List down file paths
#dir_data = "../smoking-lvm-cleaned-data/final"
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

# Number of hours elapsed since the beginning of the study
data_selfreport["smoked_unixts_scaled"] = (
    data_selfreport
        .loc[:, ["start_date_unixts","smoked_unixts"]]
        .pipe(lambda x: (x["smoked_unixts"]-x["start_date_unixts"])/(60*60))
)

#%%
# Get number of hours elapsed between two self-reported smoking events
data_selfreport = data_selfreport.sort_values(['participant_id','smoked_unixts'])

tmpdf = (
    data_selfreport
        .groupby("participant_id")
        .apply(lambda x: np.append([np.nan],[x["smoked_unixts_scaled"].head(-1)]))
        .reset_index()
        .rename(columns={"participant_id": "participant_id", 0:"smoked_unixts_scaled_lagminus01"})
)

tmparray = []
for i in range(0, len(tmpdf.index)):
    current_participant_id = tmpdf["participant_id"].loc[i]
    current_array = tmpdf[tmpdf["participant_id"]==current_participant_id]["smoked_unixts_scaled_lagminus01"]
    idx = current_array.index[0]
    current_array = current_array[idx]
    tmparray = np.append(tmparray, current_array)

data_selfreport = data_selfreport.assign(smoked_unixts_scaled_lagminus01 = tmparray)

data_selfreport["hours_between"] = (
    data_selfreport
        .loc[:, ["smoked_unixts_scaled","smoked_unixts_scaled_lagminus01"]]
        .pipe(lambda x: x["smoked_unixts_scaled"]-x["smoked_unixts_scaled_lagminus01"])
)

#%%
# Finally, select subset of columns
use_these_columns = ["participant_id", "start_date_unixts", "quit_date_unixts",
                     "expected_end_date_unixts","actual_end_date_unixts",
                     "is_post_quit", "study_day", "day_since_quit", "day_within_period",
                     "begin_unixts", "message", "delta", "smoked_unixts",
                     "smoked_unixts_scaled", "smoked_unixts_scaled_lagminus01", "hours_between"]
data_selfreport = data_selfreport.loc[:, use_these_columns]

#%%
###############################################################################
# Data preparation: Create data to be used as input to pymc3
###############################################################################

# Collect data to be used in analyses in a dictionary
collect_data_analysis = {}
collect_data_analysis['df_datapoints'] = (
    data_selfreport
        .loc[:,["participant_id", "hours_between"]]
        .dropna()
)

#%%
# collect_results is a dictionary that will collect results across all models
collect_results={}

#%%

###############################################################################
# Estimation using pymc3
###############################################################################

use_this_data = collect_data_analysis['df_datapoints']

# Create new participant id's
participant_names = use_this_data['participant_id'].unique()
n_participants = len(participant_names)
d = {'participant_id':participant_names, 'participant_idx':np.array(range(0,n_participants))}
reference_df = pd.DataFrame(d)
use_this_data = use_this_data.merge(reference_df, how = 'left', on = 'participant_id')
participant_idx = use_this_data['participant_idx'].values

with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    # Outcome Data
    Y_observed = pm.Data('hours_between', use_this_data['hours_between'].values)

    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    gamma = pm.Normal('gamma', mu=0, sd=10, shape=n_participants)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb = gamma[participant_idx]
    lamb = np.exp(loglamb)
    Y_hat = pm.Exponential('Y_hat', lam = lamb, observed=Y_observed)

#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=3000, tune=7000, cores=1)

#%%
# Calculate 95% credible interval
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]

# Produce trace plots
#pm.traceplot(posterior_samples)

# Transform coefficients and recover mu value
model_summary_meanscale = np.exp(model_summary_logscale)
model_summary_meanscale = np.power(model_summary_meanscale, -1)
model_summary_meanscale = model_summary_meanscale.rename(index=lambda x: '1/exp('+x+')') 

# Round up to 3 decimal places
model_summary_meanscale = model_summary_meanscale.round(3)
model_summary_meanscale = model_summary_meanscale.round(3)

# Collect results
collect_results['0'] = {'model':model, 
                        'posterior_samples':posterior_samples,
                        'model_summary_logscale':model_summary_logscale,
                        'model_summary_meanscale':model_summary_meanscale}
#%%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale, model_summary_meanscale

#%%

###############################################################################
# Print results from all models
###############################################################################
import matplotlib.pyplot as plt

# Model 0
pm.traceplot(collect_results['0']['posterior_samples'])
print(collect_results['0']['model_summary_logscale'])
print(collect_results['0']['model_summary_meanscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['0']['posterior_samples'], var_names=['gamma'], credible_interval=0.95)

# %%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_pp_models')
outfile = open(filename, 'wb')
pickle.dump(collect_results, outfile)
outfile.close()

# %%
