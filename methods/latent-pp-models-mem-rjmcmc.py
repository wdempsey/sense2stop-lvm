#%%
import pymc3 as pm
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
clean_data = pickle.load(infile)
infile.close()

#%% 

'''
Delete all times > 1hr before start time. 
Extend day to handle all other times and remove duplicates
Need to move this part of code to pre-processing at some point
'''

for key in clean_data.keys():
    temp = clean_data[key]
    for days in temp.keys():
        day_temp = temp[days]
        if len(day_temp['hours_since_start_day']) > 0:
            ## Check if any times < or > 1hr 
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'].iloc[np.where(day_temp['hours_since_start_day'] > -1)]
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'].iloc[np.where(day_temp['day_length'] - day_temp['hours_since_start_day'] > -1)]
            day_min = np.min(day_temp['hours_since_start_day'])
            day_min = np.min([day_min,0])
            day_max = np.max(day_temp['hours_since_start_day'])
            day_max = np.max([day_max, day_temp['day_length']])
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'] - day_min
            day_temp['hours_since_start_day'] = np.unique(day_temp['hours_since_start_day'])
            day_temp['day_length'] = day_max - day_min



#%%

###############################################################################
# Estimation using pymc3
###############################################################################

def exponential_log_complementary_cdf(x, lam):
    ''' log complementary CDF of exponential distribution '''
    return -lam*x

def exponential_log_pdf(x, lam):
    ''' log complementary CDF of exponential distribution '''
    return np.log(lam)-lam*x

def selfreport_mem(observed, latent):
    total = 1.0
    for entry in observed:
        if entry not in latent:
            total = -1000000
    for entry in latent:
        if entry in observed:
            total *= 0.9
        else:
            total *= 0.1
    return total

#%%

###############################################################################
# Estimation using pymc3
###############################################################################

#%%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta = pm.Normal('beta', mu=0, sd=10)
    loglamb_observed = beta
    lamb_observed = np.exp(loglamb_observed)
    
    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    for participants in clean_data.keys():
        for days in clean_data[participants].keys():
            if len(clean_data[participants][days]['hours_since_start_day']) > 0:
                pp_rate = lamb_observed*clean_data[participants][days]['day_length']
                smoke_length = pm.Poisson("smoke_length", mu=pp_rate, testval = len(clean_data[participants][days]['hours_since_start_day'])) # Number of Events in Day
                smoke_times = pm.Uniform("smoke_times", lower = 0.0, upper = clean_data[participants][days]['day_length'], shape = smoke_length.shape[0], testval = clean_data[participants][days]['hours_since_start_day']) # Location of Events in Day
                sr_times = pm.Potential('sr_times', selfreport_mem(observed=clean_data[participants][days]['hours_since_start_day'], latent=smoke_times))


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
collect_results = {'model':model, 
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
pm.traceplot(collect_results['posterior_samples'])
print(collect_results['model_summary_logscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['posterior_samples'], var_names=['beta'], credible_interval=0.95)
pm.forestplot(collect_results['posterior_samples'], var_names=['beta_day'], credible_interval=0.95)
#pm.forestplot(collect_results['0']['posterior_samples'], var_names=['alpha'], credible_interval=0.95)

# %%
filename = os.path.join(os.path.realpath(dir_picklejar), 'rjmcmc_models')
outfile = open(filename, 'wb')
pickle.dump(collect_results, outfile)
outfile.close()

# %% REsidual code for safekeeping

#    # Y_hat_latent = pm.Determinist(of Y_diff_latent)
#    # Y_observed = pm.Potential('Y_observed', selfreport_mem(Y_hat_latent))
##    Y_hat_observed is 'hours_since_start_day'
##    Given hours_since_start_day, use smartdumbRJ.py to generate a new latent event times (Y_hat_latent)
##    Given Y_hat_latent, take diff sequence and model as exponential holding times    
#    loglamb_observed = beta
#    lamb_observed = np.exp(loglamb_observed)
#    # Define Y_hat_latent
#    # Take sequence of differences, Y_diff_latent
#    Y_diff_latent = pm.Exponential('Y_diff_latent', lam = lamb_observed)
