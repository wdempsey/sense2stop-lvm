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

exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']
dir_code_methods = os.environ['dir_code_methods']

# %%
# Output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'setup-day-limits.py')).read())

# %%
execute_test = False

if execute_test:
# Sanity check: are there any duplicates in the hours_since_start_day column?
    for participant in dict_knitted_with_puffmarker.keys():
        for days in dict_knitted_with_puffmarker[participant].keys():
            current_data = dict_knitted_with_puffmarker[participant][days]
            if len(current_data.index) > 0:
                which_idx_dup = current_data['hours_since_start_day'].duplicated()
                which_idx_dup = np.array(which_idx_dup)
                if np.sum(which_idx_dup*1.)>0:
                    print((participant, days, np.cumsum(which_idx_dup)))  # prints those participant-days with duplicates
                    # found: 1 selfreport and 1 random ema with exactly the same hours_since_start_day
                    # the selfreport will eventually be dropped since when_smoke=4

# %%
# Test out the function
execute_test = False

if execute_test:
    use_this_id = None
    use_this_days = None
    # Test out the function latent_poisson_process_ex1
    # pre-quit
    print(latent_poisson_process_ex1(latent_dict = latent_data[use_this_id][use_this_days], params = {'lambda': 0.14}))
    # post-quit
    print(latent_poisson_process_ex1(latent_dict = latent_data[use_this_id][use_this_days], params = {'lambda': 0.14}))

# %%
# Test out the function
execute_test = False

if execute_test:
    use_this_id = None
    use_this_days = None
    # Test out the function latent_poisson_process_ex2
    # pre-quit
    print(latent_poisson_process_ex2(latent_dict = latent_data[use_this_id][use_this_days], params = {'lambda_prequit': 0.14, 'lambda_postquit': 0.75}))
    # post-quit
    print(latent_poisson_process_ex2(latent_dict = latent_data[use_this_id][use_this_days], params = {'lambda_prequit': 0.14, 'lambda_postquit': 0.75}))


# %%
# Test out the class
execute_test = False

if execute_test:
    tmp_latent_data = copy.deepcopy(latent_data)

    lat_pp_ex1 = latent(data=tmp_latent_data, model=latent_poisson_process_ex1, params = {'lambda': 0.14})
    print(lat_pp_ex1.model)
    print(lat_pp_ex1.params)
    print(lat_pp_ex1.compute_total_pp(use_params = None))

    lat_pp_ex1.update_params(new_params = {'lambda': 0.77})
    print(lat_pp_ex1.model)
    print(lat_pp_ex1.params)
    print(lat_pp_ex1.compute_total_pp(use_params = None))

# %%
# Another test on the class
execute_test = False

if execute_test:
    tmp_latent_data = copy.deepcopy(latent_data)

    lat_pp_ex2 = latent(data=tmp_latent_data, model=latent_poisson_process_ex2, params = {'lambda_prequit': 0.14, 'lambda_postquit': 0.75})
    print(lat_pp_ex2.model)
    print(lat_pp_ex2.params)
    print(lat_pp_ex2.compute_total_pp(use_params = None))

    lat_pp_ex2.update_params(new_params = {'lambda_prequit': 0.05, 'lambda_postquit': 0.25})
    print(lat_pp_ex2.model)
    print(lat_pp_ex2.params)
    print(lat_pp_ex2.compute_total_pp(use_params = None))


# %%
# Test out the function
execute_test = False

if execute_test:
    use_participant = None
    use_days = None
    tmp_clean_data = copy.deepcopy(clean_data[use_participant][use_days])  # keep clean_data[use_participant][use_days] untouched
    tmp_latent_data = copy.deepcopy(latent_data[use_participant][use_days])  # keep latent_data[use_participant][use_days] untouched
    tmp_clean_data, tmp_latent_data = matching(observed_dict = tmp_clean_data, latent_dict = tmp_latent_data)
    print(tmp_clean_data)  
    print(tmp_latent_data)
    print(clean_data[use_participant][use_days])  # Check that this object remains unmodified
    print(latent_data[use_participant][use_days])  # Check that this object remains unmodified

# %%
# Test out the function
execute_test = False

if execute_test:
    use_participant = None
    use_days = None
    tmp_clean_data = copy.deepcopy(clean_data[use_participant][use_days])  # keep clean_data[use_participant][use_days] untouched
    tmp_latent_data = copy.deepcopy(latent_data[use_participant][use_days])
    if len(tmp_latent_data['matched']) > 0:
        res = selfreport_mem(observed_dict = tmp_clean_data, latent_dict = tmp_latent_data)
        print(res)


# %%

# Test out the function
execute_test = False

if execute_test:
    use_participant = None
    use_days = None
    tmp_clean_data = copy.deepcopy(clean_data[use_participant][use_days])  # keep clean_data[use_participant][use_days] untouched
    tmp_latent_data = copy.deepcopy(latent_data[use_participant][use_days])
    res = selfreport_mem_total(observed_dict = tmp_clean_data, latent_dict = tmp_latent_data, params = {'p':0.9})
    print(res)

# %%
# Another test of the function
execute_test = False

if execute_test:
    tmp_clean_data = copy.deepcopy(clean_data)  # keep clean_data untouched
    tmp_latent_data = copy.deepcopy(latent_data)  # keep latent_data untouched

    # Sanity check: are there observed events which are NOT matched to latent events?
    all_matched = True

    for use_this_id in tmp_clean_data.keys():
        for use_this_days in tmp_clean_data[use_this_id].keys():
            observed = tmp_clean_data[use_this_id][use_this_days]
            latent = tmp_latent_data[use_this_id][use_this_days]
            res = selfreport_mem_total(observed_dict = observed, latent_dict = latent, params = {'p':0.9})
            if res== -np.inf:
                all_matched = False
                print(("NOT all matched", use_this_id, use_this_days, res))

    if all_matched:
        print("all observed events are matched to latent events")


# %% 
# Test out the class

execute_test = False

if execute_test:
    tmp_clean_data = copy.deepcopy(clean_data)  # keep clean_data untouched
    tmp_latent_data = copy.deepcopy(latent_data)  # keep latent_data untouched
    sr_mem = measurement_model(data=tmp_clean_data, model=selfreport_mem_total, latent = tmp_latent_data, model_params={'p':0.9})
    print(sr_mem.model_params)
    print(sr_mem.compute_total_mem())
    sr_mem.update_params(new_params = {'p':0.4})
    print(sr_mem.model_params)
    print(sr_mem.compute_total_mem())