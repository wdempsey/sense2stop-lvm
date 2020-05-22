#%%
import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt

# List down file paths
#dir_data = "../smoking-lvm-cleaned-data/final"
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']

###############################################################################
# Time to next event model
###############################################################################

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_pp_models')
infile = open(filename,'rb')
collect_results = pickle.load(infile)
infile.close()

# Model 0
pm.traceplot(collect_results['0']['posterior_samples'])
print(collect_results['0']['model_summary_logscale'])
print(collect_results['0']['model_summary_meanscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['0']['posterior_samples'], var_names=['gamma'], credible_interval=0.95)

###############################################################################
# Random effects model
###############################################################################

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_randeff_models_allzeros')
infile = open(filename,'rb')
collect_results = pickle.load(infile)
infile.close()

# Model 0
pm.traceplot(collect_results['0']['posterior_samples'])
print(collect_results['0']['model_summary_logscale'])
print(collect_results['0']['model_summary_expscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['0']['posterior_samples'], var_names=['gamma'], credible_interval=0.95)

#%%
# Model 1
pm.traceplot(collect_results['1']['posterior_samples'])
print(collect_results['1']['model_summary_logscale'])
print(collect_results['1']['model_summary_expscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['gamma_prequit'], credible_interval=0.95)

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['gamma_postquit'], credible_interval=0.95)

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_prequit'], credible_interval=0.95)

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_postquit'], credible_interval=0.95)

#%%
# Model 2
pm.traceplot(collect_results['2']['posterior_samples'])
print(collect_results['2']['model_summary_logscale'])
print(collect_results['2']['model_summary_expscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['gamma_prequit'], credible_interval=0.95)

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['gamma_postquit'], credible_interval=0.95)

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['beta_prequit_intercept'], credible_interval=0.95)

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['beta_prequit_slope'], credible_interval=0.95)

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['2']['posterior_samples'], var_names=['beta_postquit_slope'], credible_interval=0.95)

