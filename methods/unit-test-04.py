# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy.stats import norm
from scipy.stats import lognorm
import copy
import matplotlib.pyplot as plt  

exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']
dir_code_methods = os.environ['dir_code_methods']

exec(open(os.path.join(os.path.realpath(dir_code_methods), 'unit-test-00.py')).read())

# %%
# Test out class
tmp_latent_data = copy.deepcopy(latent_data)
tmp_clean_data = copy.deepcopy(clean_data)
#lat_pp = latent(data=tmp_latent_data, model=latent_poisson_process_ex2, params = {'lambda_prequit': 0.20, 'lambda_postquit': 0.10})
#lat_pp = latent(data=tmp_latent_data, model=latent_poisson_process_ex2, params = {'lambda_prequit': 1, 'lambda_postquit': 1})
lat_pp = latent(data=tmp_latent_data, model=latent_poisson_process_ex2, params = {'lambda_prequit': .3, 'lambda_postquit': 1.7})
sr_mem = measurement_model(data=tmp_clean_data, model=selfreport_mem_total, latent = tmp_latent_data, model_params={'p':0.9})
test_model = model(init = clean_data,  latent = lat_pp , model = sr_mem)

num_iters = 7000
use_cutpoint = 2000

# %%
###############################################################################
# Adaptive updates
###############################################################################
dict_store_params = {}
count_accept = 0
cov_init = ((.000002*2)/(2.38**2))*np.eye(2)
barX_init = np.array([0., 0.])
current_total = 0

# %%
for iter in range(1,num_iters):
    if iter == 1:
        current_out_dict = test_model.adapMH_params(adaptive = True,
                                                    covariance = cov_init, 
                                                    barX = barX_init,
                                                    covariance_init = cov_init, 
                                                    barX_init = barX_init,
                                                    iteration = iter, 
                                                    cutpoint = use_cutpoint, 
                                                    sigma = (2.38**2)/len(barX_init))
        # Store parameters
        lat_pp.update_params(new_params = {'lambda_prequit':current_out_dict['new_params']['lambda_prequit'],
                                           'lambda_postquit':current_out_dict['new_params']['lambda_postquit']})
        dict_store_params.update({iter:current_out_dict})

        if current_out_dict['rejected'] == 0:  # if not rejected
            count_accept = count_accept+1
        
        # Update params
        cov_new = current_out_dict['covariance_new']
        sigma_new = current_out_dict['sigma_new']
        barX_new = current_out_dict['barX_new']

    else:
        current_out_dict = test_model.adapMH_params(adaptive = True,
                                                    covariance = cov_new, 
                                                    barX = barX_new,
                                                    covariance_init = cov_init, 
                                                    barX_init = barX_init,
                                                    iteration = iter, 
                                                    cutpoint = use_cutpoint, 
                                                    sigma = sigma_new)
        # Store parameters
        lat_pp.update_params(new_params = {'lambda_prequit':current_out_dict['new_params']['lambda_prequit'],
                                           'lambda_postquit':current_out_dict['new_params']['lambda_postquit']})
        dict_store_params.update({iter:current_out_dict})

        if current_out_dict['rejected'] == 0:  # if not rejected
            count_accept = count_accept+1

        # Update params
        cov_new = current_out_dict['covariance_new']
        sigma_new = current_out_dict['sigma_new']

        if iter <= use_cutpoint:
            barX_new = barX_init
        else:
            current_total = current_total + current_out_dict['log_new_params']
            barX_new = current_total/(use_cutpoint - iter)

        print(current_out_dict['new_params'])


# Print out acceptance probability
accept_prob = count_accept/num_iters
print(accept_prob)


# %%
temp = np.zeros(shape = (num_iters, 2+len(lat_pp.params.keys())))

for iter in range(1,num_iters):
    temp[iter,0] = dict_store_params[iter]['new_params']['lambda_prequit']
    temp[iter,1] = dict_store_params[iter]['new_params']['lambda_postquit']
    temp[iter,2] = dict_store_params[iter]['sigma_new']
    temp[iter,3] = dict_store_params[iter]['acceptprob']

# %%
plot_cutpoint = use_cutpoint + 0

fig, axs = plt.subplots(3,2)
fig.suptitle('Adaptive MH Parameter Updates\n' + 'Acceptance Probability is '+ str(round(accept_prob*100, 1)) + str('%'), fontsize=12)
axs[0,0].hist(temp[plot_cutpoint:,0], bins = 30)
axs[0,1].plot(np.arange(temp[plot_cutpoint:,0].size),temp[plot_cutpoint:,0])
axs[1,0].hist(temp[plot_cutpoint:,1], bins = 30)
axs[1,1].plot(np.arange(temp[plot_cutpoint:,1].size),temp[plot_cutpoint:,1])

axs[2,0].plot(np.arange(temp[plot_cutpoint:,2].size),temp[plot_cutpoint:,2])
axs[2,1].plot(np.arange(temp[plot_cutpoint:,3].size),temp[plot_cutpoint:,3])
plt.show()

# %%
temp = np.zeros(shape = (num_iters, 2+len(lat_pp.params.keys())))

for iter in range(1,num_iters):
    temp[iter,0] = dict_store_params[iter]['new_params']['lambda_prequit']
    temp[iter,1] = dict_store_params[iter]['new_params']['lambda_postquit']
    temp[iter,2] = dict_store_params[iter]['sigma_new']
    temp[iter,2] = dict_store_params[iter]['acceptprob']

fig, axs = plt.subplots(3,2)
fig.suptitle('Adaptive MH Parameter Updates\n' + 'Acceptance Probability is '+ str(round(accept_prob*100, 1)) + str('%'), fontsize=12)
axs[0,0].hist(temp[:,0], bins = 30)
axs[0,1].plot(np.arange(temp[:,0].size),temp[:,0])
axs[1,0].hist(temp[:,1], bins = 30)
axs[1,1].plot(np.arange(temp[:,1].size),temp[:,1])
axs[2,0].plot(np.arange(temp[:,2].size),temp[:,2])
axs[2,1].plot(np.arange(temp[:,3].size),temp[:,3])
plt.show()
# %%
