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

exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']

# Read in data
data_selfreport = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'work_with_datapoints.csv'))

#%%
###############################################################################
# Data preparation: Create data to be used as input to pymc3
###############################################################################

# Collect data to be used in analyses in a dictionary
collect_data_analysis = {}
collect_data_analysis['df_datapoints'] = data_selfreport

#%%

###############################################################################
# Define functions
###############################################################################

def exponential_log_complementary_cdf(x, lam):
    ''' log complementary CDF of exponential distribution '''
    return -lam*x

#%%
# collect_results is a dictionary that will collect results across all models
collect_results={}

#%%

###############################################################################
# Estimation using pymc3
###############################################################################

use_this_data = collect_data_analysis['df_datapoints']

censored = use_this_data['censored'].values.astype(bool)
time_to_next_event = use_this_data['time_to_next_event'].values.astype(float)
day_within_period = use_this_data['day_within_period'].values.astype(float)

#%%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta = pm.Normal('beta', mu=0, sd=10)
    beta_day = pm.Normal('beta_day', mu=0, sd=10)
    #alpha = pm.Normal('alpha', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = beta + beta_day*day_within_period[~censored]
    lamb_observed = np.exp(loglamb_observed)
    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

    loglamb_censored = beta + beta_day*day_within_period[censored] # Switched model to 1 parameter for both censored/uncensored (makes sense if final obs is "real")
    # loglamb_censored = alpha # Model makes more sense if the final censored obs if due to dropout
    lamb_censored = np.exp(loglamb_censored)
    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))


#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=5000, tune=5000, cores=1, init='adapt_diag', target_accept=0.90, max_treedepth=50)

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

censored = use_this_data['censored'].values.astype(bool)
day_within_period = use_this_data['day_within_period'].values.astype(float)
time_to_next_event = use_this_data['time_to_next_event'].values.astype(float)
is_post_quit = use_this_data['is_post_quit'].values.astype(float)

#%%
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
    loglamb_observed = beta_prequit*(1-is_post_quit[~censored]) + beta_prequit_day*day_within_period[~censored]*(1-is_post_quit[~censored]) + beta_postquit*is_post_quit[~censored] + beta_postquit_day*day_within_period[~censored]*is_post_quit[~censored]
    lamb_observed = np.exp(loglamb_observed)
    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

    loglamb_censored = beta_prequit*(1-is_post_quit[censored]) + beta_prequit_day*day_within_period[censored]*(1-is_post_quit[censored]) + beta_postquit*is_post_quit[censored] + beta_postquit_day*day_within_period[censored]*is_post_quit[censored] # Model if no dropout
#    loglamb_censored = alpha # Model if final window is drop-out
    lamb_censored = np.exp(loglamb_censored)
    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))


#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=5000, tune=5000, cores=1, init='adapt_diag', target_accept=0.90, max_treedepth=50)

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

censored = use_this_data['censored'].values.astype(bool)
day_within_period = use_this_data['day_within_period'].values.astype(float)
time_to_next_event = use_this_data['time_to_next_event'].values.astype(float)
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
    posterior_samples = pm.sample(draws=5000, tune=5000, cores=1, init='adapt_diag', target_accept=0.90, max_treedepth=50)

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

# %%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_pp_models_sr')
outfile = open(filename, 'wb')
pickle.dump(collect_results, outfile)
outfile.close()

# %%
