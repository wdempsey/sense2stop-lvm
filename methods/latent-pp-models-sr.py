#%%
import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import os
import pickle
import theano.tensor as tt
from scipy import special

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
data_selfreport['hours_since_start'] = (data_selfreport['date'] - data_selfreport['start_date'])/np.timedelta64(1,'h')


#%%
# Get number of hours elapsed between two self-reported smoking events
data_selfreport['actual_end_date_hrts'] = pd.to_datetime(data_selfreport['actual_end_date_hrts'])
data_selfreport['time_to_quit'] = (data_selfreport.actual_end_date_hrts - data_selfreport.date) / np.timedelta64(1,'m') + 720 # Add 720 minutes to deal with quit date you can provide data still.
data_selfreport = data_selfreport.sort_values(['participant_id','date'])
data_selfreport['time_to_next_event'] = data_selfreport.groupby("participant_id").date.diff().shift(-1)/np.timedelta64(1,'m')

#%%
# For NaN, time_to_next_event is the time until actual quit date.
# These should be treated as censored times  
data_selfreport["censored"] = data_selfreport["time_to_next_event"].isnull()

for index in np.where(data_selfreport.censored==True):
    temp = data_selfreport['time_to_quit'].iloc[index]
    data_selfreport['time_to_next_event'].iloc[index] = temp


#%%
# Finally, select subset of columns
use_these_columns = ["participant_id",
                     "start_date_hrts", "quit_date_hrts",
                     "expected_end_date_hrts","actual_end_date_hrts", 
                     "is_post_quit", "study_day", "day_since_quit", "day_within_period",
                     "message", "delta", "time_to_next_event","censored"]
data_selfreport = data_selfreport.loc[:, use_these_columns]

#%%
###############################################################################
# Data preparation: Create data to be used as input to pymc3
###############################################################################

# Collect data to be used in analyses in a dictionary
collect_data_analysis = {}
collect_data_analysis['df_datapoints'] = (
    data_selfreport
        .loc[:,["participant_id", "is_post_quit", "time_to_next_event","censored", 
                "day_within_period", "delta"]]
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

def exponential_log_pdf(x, lam):
    ''' log complementary CDF of exponential distribution '''
    return np.log(lam)-lam*x

def convert_windowtag(windowtag):
    if windowtag == 1:
        window_max = 5; window_min = 0
    elif windowtag == 2:
        window_max = 15; window_min = 5
    elif windowtag == 3:
        window_max = 30; window_min = 15
    else:
        window_max = 60; window_min = 30
    return window_min, window_max

def normal_cdf(x, mu=0, sd=1):
    '''Use theano to compute cdf'''
    z = (x-mu)/sd
    return (tt.erf(z/np.sqrt(2))+1)/2 

def selfreport_mem(x, t, winmin, winmax):
    ''' Measurement model for self-report '''
    gap = t - x
    mem_scale = 10
    upper = normal_cdf(winmax, mu = gap, sd = mem_scale)
    lower = normal_cdf(winmin, mu = gap, sd = mem_scale)
    return tt.log(upper-lower)

censored = use_this_data['censored'].values.astype(bool)
time_to_next_event = use_this_data['time_to_next_event'].values.astype(float)
day_within_period = use_this_data['day_within_period'].values.astype(float)
is_post_quit = use_this_data['is_post_quit'].values.astype(float)
windowtag = use_this_data['delta'].values.astype(float)
temp = np.array(list(map(convert_windowtag,windowtag)))
windowmin = temp[:,0]
windowmax = temp[:,1]
midpoint = (windowmin+windowmax)/2
test_obs = time_to_next_event-midpoint
test_obs[test_obs <= 0] = 1.
negativetimes = time_to_next_event <= 1
remove_obs = censored | negativetimes 
num_ok = time_to_next_event[~remove_obs].size
test_obs1 = test_obs[~remove_obs]
time_to_next_event1 = time_to_next_event[~remove_obs]
windowmin1=windowmin[~remove_obs]
windowmax1=windowmax[~remove_obs]
day_within_period1=day_within_period[~remove_obs]
is_post_quit1 = is_post_quit[~remove_obs]

#%%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta = pm.Normal('beta', mu=0, sd=10)
    beta_day = pm.Normal('beta_day', mu=0, sd=10)
           
    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = beta + beta_day*day_within_period1
    lamb_observed = np.exp(loglamb_observed)
#    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed = time_to_next_event[~censored])
    Y_latent = pm.Exponential('Y_latent', lam = lamb_observed, shape=len(test_obs1), testval=test_obs1)
    Y_observed = pm.Potential('Y_observed', selfreport_mem(Y_latent, time_to_next_event1, windowmin1, windowmax1))
    
    loglamb_censored = beta + beta_day*day_within_period[censored] # Switched model to 1 parameter for both censored/uncensored (makes sense if final obs is "real")
    lamb_censored = np.exp(loglamb_censored)
    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))


#%%
# Sample from posterior distribution
with model:
#    posterior_samples = pm.sample(draws=3000, tune=2000, cores=1, target_accept=0.80)
    posterior_samples = pm.sample(draws = 3000, tune=2000, init='adapt_diag', cores = 1)
    
#%%
# Calculate 95% credible interval
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]

# Produce trace plots
pm.traceplot(posterior_samples, var_names = ['beta', 'beta_day'])

# Collect results
collect_results['0'] = {'model':model, 
                        'posterior_samples':posterior_samples,
                        'model_summary_logscale':model_summary_logscale}
#%%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale


#%%

###############################################################################
# Estimation using pymc3; pre-/post- quit split.
###############################################################################

with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    beta_prequit_day = pm.Normal('beta_prequit_day', mu=0, sd=10)
    beta_postquit_day = pm.Normal('beta_postquit_day', mu=0, sd=10)
#    alpha = pm.Normal('alpha', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = beta_prequit*(1-is_post_quit1) + beta_prequit_day*day_within_period1*(1-is_post_quit1) + beta_postquit*is_post_quit1 + beta_postquit_day*day_within_period1*is_post_quit1
    lamb_observed = np.exp(loglamb_observed)
    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event1)

    loglamb_censored = beta_prequit*(1-is_post_quit[censored]) + beta_prequit_day*day_within_period[censored]*(1-is_post_quit[censored]) + beta_postquit*is_post_quit[censored] + beta_postquit_day*day_within_period[censored]*is_post_quit[censored] # Model if no dropout
#    loglamb_censored = alpha # Model if final window is drop-out
    lamb_censored = np.exp(loglamb_censored)
    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))


#%%
# Sample from posterior distribution
with model:
#    posterior_samples = pm.sample(draws=3000, tune=2000, cores=1, target_accept=0.80)
    posterior_samples = pm.sample(draws = 3000, tune=2000, init='adapt_diag', cores = 1)

#%%
# Calculate 95% credible interval
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]

# Produce trace plots
pm.traceplot(posterior_samples, var_names = ['beta_prequit', 'beta_prequit_day' , 'beta_postquit', 'beta_postquit_day'])

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
time_to_next_event = use_this_data['time_to_next_event'].values.astype(float)
day_within_period = use_this_data['day_within_period'].values.astype(float)
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
    beta_prequit_day = pm.Normal('beta_prequit_day', mu=0, sd=10)
    beta_postquit_day = pm.Normal('beta_postquit_day', mu=0, sd=10)
    gamma_prequit = pm.Normal('gamma_prequit', mu=0, sd=10, shape=n_participants)
    gamma_postquit = pm.Normal('gamma_postquit', mu=0, sd=10, shape=n_participants)

#    alpha = pm.Normal('alpha', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = (beta_prequit + gamma_prequit[participant_idx][~censored])*(1-is_post_quit[~censored]) + beta_prequit_day*day_within_period[~censored]*(1-is_post_quit[~censored]) + (beta_postquit + gamma_postquit[participant_idx][~censored])*is_post_quit[~censored] + beta_postquit_day*day_within_period[~censored]*is_post_quit[~censored]
    lamb_observed = np.exp(loglamb_observed)
    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

#    loglamb_censored = alpha
    loglamb_censored = (beta_prequit + gamma_prequit[participant_idx][censored])*(1-is_post_quit[censored]) + beta_prequit_day*day_within_period[censored]*(1-is_post_quit[censored]) + (beta_postquit + gamma_postquit[participant_idx][censored])*is_post_quit[censored] + beta_postquit_day*day_within_period[censored]*is_post_quit[censored]
    lamb_censored = np.exp(loglamb_censored)
    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))


#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=5000, tune=5000, cores=1, target_accept=0.80)

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
pm.forestplot(collect_results['0']['posterior_samples'], var_names=['beta_day'], credible_interval=0.95)
#pm.forestplot(collect_results['0']['posterior_samples'], var_names=['alpha'], credible_interval=0.95)

# Model 1
pm.traceplot(collect_results['1']['posterior_samples'])
print(collect_results['1']['model_summary_logscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_prequit'], credible_interval=0.95)
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_prequit_day'], credible_interval=0.95)
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_postquit'], credible_interval=0.95)
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_postquit_day'], credible_interval=0.95)
#pm.forestplot(collect_results['1']['posterior_samples'], var_names=['alpha'], credible_interval=0.95)

# Model 2
pm.traceplot(collect_results['2']['posterior_samples'])
print(collect_results['2']['model_summary_logscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['beta_prequit'], credible_interval=0.95)
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['beta_prequit_day'], credible_interval=0.95)
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['beta_postquit'], credible_interval=0.95)
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['beta_postquit_day'], credible_interval=0.95)
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['gamma_prequit'], credible_interval=0.95)
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['gamma_postquit'], credible_interval=0.95)
#pm.forestplot(collect_results['2']['posterior_samples'], var_names=['alpha'], credible_interval=0.95)

# %%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_pp_models_linear_datetimes')
outfile = open(filename, 'wb')
pickle.dump(collect_results, outfile)
outfile.close()

# %%
