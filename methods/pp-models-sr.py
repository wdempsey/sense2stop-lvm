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

#%%

###############################################################################
# Estimation using pymc3: Add features
###############################################################################
use_this_data = collect_data_analysis['df_datapoints']

censored = use_this_data['censored'].values.astype(bool)
day_within_period = use_this_data['day_within_period'].values.astype(float)
time_to_next_event = use_this_data['time_to_next_event'].values.astype(float)
is_post_quit = use_this_data['is_post_quit'].values.astype(float)

#%%
is_first_sr_within_day = use_this_data['is_first_sr_within_day'].values.astype(float)
is_first_sr_within_period = use_this_data['is_first_sr_within_period'].values.astype(float)

#order_within_day = use_this_data['order_within_day'].values.astype(float)
#order_within_period = use_this_data['order_within_period'].values.astype(float)
#hours_since_start_of_study = use_this_data['hours_since_start_of_study'].values.astype(float)
#hours_since_previous_sr_within_day = use_this_data['hours_since_previous_sr_within_day'].values.astype(float)
#hours_since_previous_sr_within_period = use_this_data['hours_since_previous_sr_within_period'].values.astype(float)
#hours_since_start_of_period = use_this_data['hours_since_start_of_period'].values.astype(float)
#hours_relative_quit = use_this_data['hours_relative_quit'].values.astype(float)
#is_within48hours_quit = use_this_data['is_within48hours_quit'].values.astype(float)
#hour_of_day = use_this_data['hour_of_day'].values.astype(float)

feature1 = is_first_sr_within_day
feature2 = is_first_sr_within_period

#%%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    beta_prequit_day = pm.Normal('beta_prequit_day', mu=0, sd=10)
    beta_postquit_day = pm.Normal('beta_postquit_day', mu=0, sd=10)
    beta_prequit_feature1 = pm.Normal('beta_prequit_feature1', mu=0, sd=10)
    beta_postquit_feature1 = pm.Normal('beta_postquit_feature1', mu=0, sd=10)
    beta_prequit_feature2 = pm.Normal('beta_prequit_feature2', mu=0, sd=10)
    beta_postquit_feature2 = pm.Normal('beta_postquit_feature2', mu=0, sd=10)
#    alpha = pm.Normal('alpha', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = beta_prequit*(1-is_post_quit[~censored]) + beta_prequit_day*day_within_period[~censored]*(1-is_post_quit[~censored]) + beta_postquit*is_post_quit[~censored] + beta_postquit_day*day_within_period[~censored]*is_post_quit[~censored]
    loglamb_observed_features1 = beta_prequit_feature1*(1-feature1[~censored]) + beta_postquit_feature1*feature1[~censored]
    loglamb_observed_features2 = beta_prequit_feature2*(1-feature2[~censored]) + beta_postquit_feature2*feature2[~censored]
    lamb_observed = np.exp(loglamb_observed + loglamb_observed_features1 + loglamb_observed_features2)
    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

    loglamb_censored = beta_prequit*(1-is_post_quit[censored]) + beta_prequit_day*day_within_period[censored]*(1-is_post_quit[censored]) + beta_postquit*is_post_quit[censored] + beta_postquit_day*day_within_period[censored]*is_post_quit[censored] # Model if no dropout
    loglamb_censored_features1 = beta_prequit_feature1*(1-feature1[censored]) + beta_postquit_feature1*feature1[censored]
    loglamb_censored_features2 = beta_prequit_feature2*(1-feature2[censored]) + beta_postquit_feature2*feature2[censored]
    lamb_censored = np.exp(loglamb_censored + loglamb_censored_features1 + loglamb_censored_features2)
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
collect_results['withfeatures0'] = {'model':model, 
                                    'posterior_samples':posterior_samples,
                                    'model_summary_logscale':model_summary_logscale}
#%%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale

#%%
###############################################################################
# Estimation using pymc3: Add features
###############################################################################
use_this_data = collect_data_analysis['df_datapoints']

censored = use_this_data['censored'].values.astype(bool)
day_within_period = use_this_data['day_within_period'].values.astype(float)
time_to_next_event = use_this_data['time_to_next_event'].values.astype(float)
is_post_quit = use_this_data['is_post_quit'].values.astype(float)

#%%
hours_since_previous_sr_within_day = use_this_data['hours_since_previous_sr_within_day'].values.astype(float)
hours_since_previous_sr_within_period = use_this_data['hours_since_previous_sr_within_period'].values.astype(float)
#is_first_sr_within_day = use_this_data['is_first_sr_within_day'].values.astype(float)
#is_first_sr_within_period = use_this_data['is_first_sr_within_period'].values.astype(float)
#order_within_day = use_this_data['order_within_day'].values.astype(float)
#order_within_period = use_this_data['order_within_period'].values.astype(float)
#hours_since_start_of_study = use_this_data['hours_since_start_of_study'].values.astype(float)
#hours_since_start_of_period = use_this_data['hours_since_start_of_period'].values.astype(float)
#hours_relative_quit = use_this_data['hours_relative_quit'].values.astype(float)
#is_within48hours_quit = use_this_data['is_within48hours_quit'].values.astype(float)
#hour_of_day = use_this_data['hour_of_day'].values.astype(float)

feature1 = hours_since_previous_sr_within_day
feature2 = hours_since_previous_sr_within_period

#%%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    beta_prequit_day = pm.Normal('beta_prequit_day', mu=0, sd=10)
    beta_postquit_day = pm.Normal('beta_postquit_day', mu=0, sd=10)
    beta_prequit_feature1 = pm.Normal('beta_prequit_feature1', mu=0, sd=10)
    beta_postquit_feature1 = pm.Normal('beta_postquit_feature1', mu=0, sd=10)
    beta_prequit_feature2 = pm.Normal('beta_prequit_feature2', mu=0, sd=10)
    beta_postquit_feature2 = pm.Normal('beta_postquit_feature2', mu=0, sd=10)
#    alpha = pm.Normal('alpha', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = beta_prequit*(1-is_post_quit[~censored]) + beta_prequit_day*day_within_period[~censored]*(1-is_post_quit[~censored]) + beta_postquit*is_post_quit[~censored] + beta_postquit_day*day_within_period[~censored]*is_post_quit[~censored]
    loglamb_observed_features1 = beta_prequit_feature1*(1-feature1[~censored]) + beta_postquit_feature1*feature1[~censored]
    loglamb_observed_features2 = beta_prequit_feature2*(1-feature2[~censored]) + beta_postquit_feature2*feature2[~censored]
    lamb_observed = np.exp(loglamb_observed + loglamb_observed_features1 + loglamb_observed_features2)
    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

    loglamb_censored = beta_prequit*(1-is_post_quit[censored]) + beta_prequit_day*day_within_period[censored]*(1-is_post_quit[censored]) + beta_postquit*is_post_quit[censored] + beta_postquit_day*day_within_period[censored]*is_post_quit[censored] # Model if no dropout
    loglamb_censored_features1 = beta_prequit_feature1*(1-feature1[censored]) + beta_postquit_feature1*feature1[censored]
    loglamb_censored_features2 = beta_prequit_feature2*(1-feature2[censored]) + beta_postquit_feature2*feature2[censored]
    lamb_censored = np.exp(loglamb_censored + loglamb_censored_features1 + loglamb_censored_features2)
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
collect_results['withfeatures1'] = {'model':model, 
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
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_pp_models_sr')
outfile = open(filename, 'wb')
pickle.dump(collect_results, outfile)
outfile.close()

# %%
