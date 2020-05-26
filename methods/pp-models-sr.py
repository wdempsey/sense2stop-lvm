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
data_selfreport["is_post_quit"] = data_selfreport["day_since_quit"].apply(lambda x: 0 if x < 0 else 1)

# Create a new variable, day_within_period: 
# if is_post_quit<0, number of days after 12AM on start of study
# if is_post_quit>=0, number of days after 12AM on Quit Date
# hence day_within_period is a count variable with ZERO as minimum value
data_selfreport["day_within_period"] = np.where(data_selfreport["is_post_quit"]==0,
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
        .apply(lambda x: np.append([x["smoked_unixts_scaled"].tail(-1)], [np.nan]))
        .reset_index()
        .rename(columns={"participant_id": "participant_id", 0:"smoked_unixts_scaled_lagplus01"})
)

tmparray = []
for i in range(0, len(tmpdf.index)):
    current_participant_id = tmpdf["participant_id"].loc[i]
    current_array = tmpdf[tmpdf["participant_id"]==current_participant_id]["smoked_unixts_scaled_lagplus01"]
    idx = current_array.index[0]
    current_array = current_array[idx]
    tmparray = np.append(tmparray, current_array)

data_selfreport = data_selfreport.assign(smoked_unixts_scaled_lagplus01 = tmparray)

data_selfreport["hours_between"] = (
    data_selfreport
        .loc[:, ["smoked_unixts_scaled","smoked_unixts_scaled_lagplus01"]]
        .pipe(lambda x: x["smoked_unixts_scaled_lagplus01"] -  x["smoked_unixts_scaled"])
)

#%%
# For each participant, count number of timestamps they have
data_selfreport["ones"]=1

data_selfreport["order_in_sequence"] = (
    data_selfreport
        .groupby("participant_id")["ones"]
        .cumsum()
)

#%%
data_selfreport = (
    data_selfreport
        .groupby("participant_id")["order_in_sequence"]
        .max()
        .reset_index()
        .rename(columns={"participant_id":"participant_id","order_in_sequence":"max_order_in_sequence"})
        .merge(data_selfreport, how="right", on="participant_id")
)

data_selfreport["censored"] = np.where(data_selfreport["order_in_sequence"]==data_selfreport["max_order_in_sequence"],1,0)

#%%
# Finally, select subset of columns
use_these_columns = ["participant_id",
                     "start_date_hrts", "quit_date_hrts",
                     "expected_end_date_hrts","actual_end_date_hrts", 
                     "start_date_unixts", "quit_date_unixts",
                     "expected_end_date_unixts","actual_end_date_unixts",
                     "is_post_quit", "study_day", "day_since_quit", "day_within_period",
                     "begin_unixts", "message", "delta", "smoked_unixts",
                     "smoked_unixts_scaled", "smoked_unixts_scaled_lagplus01", 
                     "hours_between","censored"]
data_selfreport = data_selfreport.loc[:, use_these_columns]

#%%
###############################################################################
# Data preparation: Create data to be used as input to pymc3
###############################################################################

# Collect data to be used in analyses in a dictionary
collect_data_analysis = {}
collect_data_analysis['df_datapoints'] = (
    data_selfreport
        .loc[:,["participant_id", "is_post_quit", "smoked_unixts_scaled", "hours_between","censored"]]
)

#%%
# collect_results is a dictionary that will collect results across all models
collect_results={}

#%%

###############################################################################
# Estimation using pymc3
###############################################################################

use_this_data = collect_data_analysis['df_datapoints']

def exponential_log_complementary_cdf(x, lam):
    ''' log complementary CDF of exponential distribution '''
    return -lam*x

censored = use_this_data['censored'].values.astype(bool)
hours_between_observed = use_this_data['hours_between'].values.astype(float)
hours_since_start_censored = use_this_data['smoked_unixts_scaled'].values.astype(float)

#%%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta = pm.Normal('beta', mu=0, sd=10)
    #alpha = pm.Normal('alpha', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = beta
    lamb_observed = np.exp(loglamb_observed)
    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=hours_between_observed[~censored])

    loglamb_censored = beta # Switched model to 1 parameter for both censored/uncensored (makes sense if final obs is "real")
    # loglamb_censored = alpha # Model makes more sense if the final censored obs if due to dropout
    lamb_censored = np.exp(loglamb_censored)
    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = hours_since_start_censored[censored], lam = lamb_censored))


#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=3000, tune=2000, cores=1, target_accept=0.90)

#%%
# Calculate 95% credible interval
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]

# Produce trace plots
pm.traceplot(posterior_samples)

# Collect results
collect_results['0'] = {'model':model, 
                        'posterior_samples':posterior_samples,
                        'model_summary_logscale':model_summary_logscale}
#%%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale


#%%

###############################################################################
# Estimation using pymc3
###############################################################################

use_this_data = collect_data_analysis['df_datapoints']

def exponential_log_complementary_cdf(x, lam):
    ''' log complementary CDF of exponential distribution '''
    return -lam*x

censored = use_this_data['censored'].values.astype(bool)
hours_between_observed = use_this_data['hours_between'].values.astype(float)
hours_since_start_censored = use_this_data['smoked_unixts_scaled'].values.astype(float)
is_post_quit = use_this_data['is_post_quit'].values.astype(float)

#%%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
#    alpha = pm.Normal('alpha', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = beta_prequit*(1-is_post_quit[~censored]) + beta_postquit*is_post_quit[~censored]
    lamb_observed = np.exp(loglamb_observed)
    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=hours_between_observed[~censored])

    loglamb_censored = beta_prequit*(1-is_post_quit[censored]) + beta_postquit*is_post_quit[censored] # Model if no dropout
#    loglamb_censored = alpha # Model if final window is drop-out
    lamb_censored = np.exp(loglamb_censored)
    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = hours_since_start_censored[censored], lam = lamb_censored))


#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=3000, tune=2000, cores=1, target_accept=0.90)

#%%
# Calculate 95% credible interval
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]

# Produce trace plots
pm.traceplot(posterior_samples)

# Collect results
collect_results['1'] = {'model':model, 
                        'posterior_samples':posterior_samples,
                        'model_summary_logscale':model_summary_logscale}
#%%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale

#%%

###############################################################################
# Estimation using pymc3
###############################################################################

use_this_data = collect_data_analysis['df_datapoints']

def exponential_log_complementary_cdf(x, lam):
    ''' log complementary CDF of exponential distribution '''
    return -lam*x

censored = use_this_data['censored'].values.astype(bool)
hours_between_observed = use_this_data['hours_between'].values.astype(float)
hours_since_start_censored = use_this_data['smoked_unixts_scaled'].values.astype(float)
is_post_quit = use_this_data['is_post_quit'].values.astype(float)

# Create new participant id's
participant_names = use_this_data['participant_id'].unique()
n_participants = len(participant_names)
d = {'participant_id':participant_names, 'participant_idx':np.array(range(0,n_participants))}
reference_df = pd.DataFrame(d)
use_this_data = use_this_data.merge(reference_df, how = 'left', on = 'participant_id')
participant_idx = use_this_data['participant_idx'].values

#%%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    gamma_prequit = pm.Normal('gamma_prequit', mu=0, sd=10, shape=n_participants)
    gamma_postquit = pm.Normal('gamma_postquit', mu=0, sd=10, shape=n_participants)

#    alpha = pm.Normal('alpha', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = (beta_prequit + gamma_prequit[participant_idx][~censored])*(1-is_post_quit[~censored]) + (beta_postquit + gamma_postquit[participant_idx][~censored])*is_post_quit[~censored]
    lamb_observed = np.exp(loglamb_observed)
    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=hours_between_observed[~censored])

#    loglamb_censored = alpha
    loglamb_censored = (beta_prequit + gamma_prequit[participant_idx][censored])*(1-is_post_quit[censored]) + (beta_postquit + gamma_postquit[participant_idx][censored])*is_post_quit[censored]
    lamb_censored = np.exp(loglamb_censored)
    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = hours_since_start_censored[censored], lam = lamb_censored))


#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=5000, tune=5000, cores=1, target_accept=0.90)

#%%
# Calculate 95% credible interval
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]

# Produce trace plots
pm.traceplot(posterior_samples)

# Collect results
collect_results['2'] = {'model':model, 
                        'posterior_samples':posterior_samples,
                        'model_summary_logscale':model_summary_logscale}
#%%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale

#%%

###############################################################################
# Print results from all models
###############################################################################
import matplotlib.pyplot as plt

# Model 0
pm.traceplot(collect_results['0']['posterior_samples'])
print(collect_results['0']['model_summary_logscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['0']['posterior_samples'], var_names=['beta'], credible_interval=0.95)
pm.forestplot(collect_results['0']['posterior_samples'], var_names=['alpha'], credible_interval=0.95)

# Model 1
pm.traceplot(collect_results['1']['posterior_samples'])
print(collect_results['1']['model_summary_logscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_prequit'], credible_interval=0.95)
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_postquit'], credible_interval=0.95)
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['alpha'], credible_interval=0.95)

# Model 2
pm.traceplot(collect_results['2']['posterior_samples'])
print(collect_results['2']['model_summary_logscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['beta_prequit'], credible_interval=0.95)
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['beta_postquit'], credible_interval=0.95)
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['gamma_prequit'], credible_interval=0.95)
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['gamma_postquit'], credible_interval=0.95)
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['alpha'], credible_interval=0.95)

# %%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_pp_models')
outfile = open(filename, 'wb')
pickle.dump(collect_results, outfile)
outfile.close()

# %%
