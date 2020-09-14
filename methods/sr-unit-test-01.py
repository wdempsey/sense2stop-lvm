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

lat_pp = latent(data=tmp_latent_data, model=latent_poisson_process_ex1, params = {'lambda': 0.20})
sr_mem = measurement_model(data=tmp_clean_data, model=selfreport_mem_total, latent = tmp_latent_data, model_params={'p':0.9})
test_model = model(init = clean_data,  latent = lat_pp , model = sr_mem)

num_iters = 6000
use_cutpoint = 1000

# %%
###############################################################################
# Non-adaptive updates
###############################################################################
dict_store_params = {}
count_accept = 0

for iter in range(1,num_iters):
    current_out_dict = test_model.adapMH_params(adaptive = False,
                                                covariance = None, 
                                                barX = None,
                                                covariance_init = None, 
                                                barX_init = None,
                                                iteration = iter, 
                                                cutpoint = use_cutpoint, 
                                                sigma = None)
    
    if current_out_dict['rejected']==-1:
        print(current_out_dict['new_params'])  # detect edge case
    else:
        lat_pp.update_params(new_params = {'lambda':current_out_dict['new_params']['lambda']})
        dict_store_params.update({iter:current_out_dict})
        if current_out_dict['rejected'] == 0:  # if not rejected
            count_accept = count_accept+1

# Print out acceptance probability
accept_prob = count_accept/num_iters
print(accept_prob)

# %%
temp = np.zeros(shape = (num_iters, len(lat_pp.params.keys())))

for iter in range(1,num_iters):
    temp[iter,0] = dict_store_params[iter]['new_params']['lambda']

# %%
fig, axs = plt.subplots(1,2)
fig.suptitle('Non-Adaptive Parameter Updates\n' + 'Acceptance Probability is '+ str(round(accept_prob*100, 1)) + str('%'), fontsize=12)

axs[0].hist(temp[use_cutpoint:,0], bins = 30)
axs[1].plot(np.arange(temp[use_cutpoint:,0].size),temp[use_cutpoint:,0])
plt.show()


