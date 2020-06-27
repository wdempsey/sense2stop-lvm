#%%
#import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import os
import pickle
from scipy import special

## List down file paths
exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']

#%%

###############################################################################
# Read in preparation: data_dates data frame
###############################################################################
filename = os.path.join(os.path.realpath(dir_picklejar), 'save_all_dict')
infile = open(filename,'rb')
collect_results = pickle.load(infile)
infile.close()

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

###############################################################################
# Estimation using pymc3
###############################################################################

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
    beta = pm.Normal('beta', mu=0, sd=10)
    
    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    Y_diff_latent = pm.Exponential('Y_latent', lam = lamb_observed, shape=len(test_obs1), testval=test_obs1)
    # Y_hat_latent = pm.Determinist(of Y_diff_latent)
    # Y_observed = pm.Potential('Y_observed', selfreport_mem(Y_hat_latent))
#    Y_hat_observed is 'hours_since_start_day'
#    Given hours_since_start_day, use smartdumbRJ.py to generate a new latent event times (Y_hat_latent)
#    Given Y_hat_latent, take diff sequence and model as exponential holding times    
    loglamb_observed = beta
    lamb_observed = np.exp(loglamb_observed)
    # Define Y_hat_latent
    # Take sequence of differences, Y_diff_latent
    Y_diff_latent = pm.Exponential('Y_diff_latent', lam = lamb_observed)


#%%
# Sample from posterior distribution
with model:
#    posterior_samples = pm.sample(draws=5000, tune=5000, cores=1, target_accept=0.80)
    posterior_samples = pm.sample(draws = 3000, tune=2000, init='adapt_diag', cores = 1)    

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
