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

# %%
# Read in dictionary where Self-Report, Random EMA, and puffmarker are knitted together into one data frame
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_knitted_with_puffmarker')
infile = open(filename,'rb')
dict_knitted_with_puffmarker = pickle.load(infile)
infile.close()

# Output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'setup-day-limits.py')).read())

# %%
# Create mock latent and observed data
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'create-input-data-03.py')).read())

# note:
# output of the above script is
# latent_data
# clean_data

###############################################################################################################
###############################################################################################################
###############################################################################################################

# %%
def matching(observed_dict, latent_dict):
    ''' 
    For each obs, looks backward to see if there is a matching
    latent time (that is not taken by a prior obs).  
    Reports back which latent events are matched to observed events and
    calculates delay

    Note: What if we have latent but do not have observed?
    Need to take care of that case in the function

    observed_dict: Observed dict for a given PERSON-DAY
    latent_dict: Latent dict for a given PERSON-DAY
    '''

    if (len(observed_dict['windowtag'])>0) and (len(latent_dict['hours_since_start_day'])>0):
        # This is the case when there are true latent events AND observed events
        observed_dict['matched_latent_event'] = np.array(np.repeat(np.nan, len(observed_dict['windowtag'])))
        observed_dict['delay'] = np.array(np.repeat(np.nan, len(observed_dict['windowtag'])))
        latent_dict['matched'] = np.array(np.repeat(False, len(latent_dict['hours_since_start_day'])))    
        latent_dict['matched_assessment_type'] = np.empty(len(latent_dict['hours_since_start_day']), dtype="<U30")

        for idx_assessment in range(0, len(observed_dict['assessment_order'])):
            # if 'yes' is reported in current assessment then the assessment will be 
            # matched to the latent event occurring immediately prior to it
            # if that latent event has not yet been matched to any observed event
            if observed_dict['smoke'][idx_assessment]=='No':
                next
            else:
                if observed_dict['assessment_type'][idx_assessment]=='selfreport':
                    this_scalar = observed_dict['assessment_begin'][idx_assessment]
                    which_idx = (latent_dict['hours_since_start_day'] < this_scalar)  # check: this line will work if there are no latent_dict['hours_since_start_day'] or observed_dict['assessment_begin'] that are EXACTLY zero
                    which_idx = np.where(which_idx)
                    which_idx = np.max(which_idx)
                    if latent_dict['matched'][which_idx] == False:
                        latent_dict['matched'][which_idx] =  True  # A match has been found!
                        latent_dict['matched_assessment_type'][which_idx] = 'selfreport'
                        observed_dict['matched_latent_event'][idx_assessment] = latent_dict['latent_event_order'][which_idx]
                        observed_dict['delay'][idx_assessment] = observed_dict['assessment_begin'][idx_assessment] - latent_dict['hours_since_start_day'][which_idx]
                elif observed_dict['assessment_type'][idx_assessment]=='random_ema':
                    # note: no delay variable for random ema
                    this_scalar = observed_dict['assessment_begin'][idx_assessment]
                    which_idx = (latent_dict['hours_since_start_day'] < this_scalar)  # check: this line will work if there are no latent_dict['hours_since_start_day'] or observed_dict['assessment_begin'] that are EXACTLY zero
                    which_idx = np.where(which_idx)
                    which_idx = np.max(which_idx)
                    if latent_dict['matched'][which_idx] == False:
                        latent_dict['matched'][which_idx] =  True  # A match has been found!
                        latent_dict['matched_assessment_type'][which_idx] = 'random_ema'
                        observed_dict['matched_latent_event'][idx_assessment] = latent_dict['latent_event_order'][which_idx]
                        observed_dict['delay'][idx_assessment] = observed_dict['assessment_begin'][idx_assessment] - latent_dict['hours_since_start_day'][which_idx]
                else:
                    pass

    elif (len(observed_dict['windowtag'])==0) and (len(latent_dict['hours_since_start_day'])>0):
        # This is the case when there are true latent events AND no observed events
        observed_dict['matched_latent_event'] = np.array([])
        observed_dict['delay'] = np.array([])
        latent_dict['matched'] = np.array(np.repeat(False, len(latent_dict['hours_since_start_day'])))
    elif (len(observed_dict['windowtag'])==0) and (len(latent_dict['hours_since_start_day'])==0):
        # This is the case when there are no true latent events AND no observed events
        observed_dict['matched_latent_event'] = np.array([])
        observed_dict['delay'] = np.array([])
        latent_dict['matched'] = np.array([])
    else:
        next
    
    return observed_dict, latent_dict

# %%
# Perform match for each PARTICIPANT-DAY
for participant in clean_data.keys():
    for days in clean_data[participant].keys():
        clean_data[participant][days], latent_data[participant][days] = matching(observed_dict = clean_data[participant][days], latent_dict = latent_data[participant][days])


# %%
# Calculate initial estimate of delay
tot_delay = 0
cnt_delay = 0

for participant in clean_data.keys():
    for days in clean_data[participant].keys():
        current_data = clean_data[participant][days]
        if np.sum(~(np.isnan(current_data['windowtag']) | (current_data['smoke']=='No')))>0:
            tmp_array_delay = np.array(current_data['delay'])
            tmp_array_delay = tmp_array_delay[~np.isnan(tmp_array_delay)]
            tot_delay = tot_delay + np.sum(tmp_array_delay)
            cnt_delay = cnt_delay + len(current_data['windowtag'])

mean_delay_init = tot_delay/cnt_delay
lambda_delay_init = 1/mean_delay_init

print(mean_delay_init)
print(lambda_delay_init)


# %%
# Define functions for handling windowtag by assessment type
def convert_windowtag_selfreport(windowtag):
    accept_response = [1,2,3,4]
    # windowtag is in hours
    # use_this_window_max will be based on time when prevous EMA was delivered
    use_this_window_min = {1: 0/60, 2: 5/60, 3: 15/60, 4: 30/60}
    use_this_window_max = {1: 5/60, 2: 15/60, 3: 30/60, 4: np.nan} 

    if pd.isna(windowtag):
        use_value_min = np.nan
        use_value_max = np.nan
    elif windowtag in accept_response:
        use_value_min = use_this_window_min[windowtag] 
        use_value_max = use_this_window_max[windowtag] 
    else:
        use_value_min = np.nan
        use_value_max = np.nan

    return use_value_min, use_value_max

def convert_windowtag_random_ema(windowtag):
    accept_response = [1,2,3,4,5,6]
    # windowtag is in hours
    # use_this_window_max will be based on time when prevous EMA was delivered
    use_this_window_min = {1: 1/60, 2: 20/60, 3: 40/60, 4: 60/60, 5: 80/60, 6: 100/60}
    use_this_window_max = {1: 20/60, 2: 40/60, 3: 60/60, 4: 80/60, 5: 100/60, 6: np.nan}

    if pd.isna(windowtag):
        use_value_min = np.nan
        use_value_max = np.nan
    elif windowtag in accept_response:
        use_value_min = use_this_window_min[windowtag] 
        use_value_max = use_this_window_max[windowtag] 
    else:
        use_value_min = np.nan
        use_value_max = np.nan

    return use_value_min, use_value_max


# %%
use_participant = None
use_days = None
observed_dict = copy.deepcopy(clean_data[use_participant][use_days])
latent_dict = copy.deepcopy(latent_data[use_participant][use_days])

# %%
prob_reported = []
tot_observed_smoked = np.sum(observed_dict['smoke']=='Yes')
tot_measurements = len(observed_dict['windowtag'])

# %%
if tot_observed_smoked>0:
    for idx_assessment in range(0, tot_measurements):
        if observed_dict['smoke'][idx_assessment]=='Yes' and observed_dict['assessment_type'][idx_assessment]=='selfreport':
            # Grab true latent time matched to current reported time
            idx_matched_latent_event = observed_dict['matched_latent_event'][idx_assessment]
            idx_matched_latent_event = np.int64(idx_matched_latent_event)
            curr_true_time = latent_dict['hours_since_start_day'][idx_matched_latent_event]
            # Grab current reported time
            val_min, val_max = convert_windowtag_selfreport(windowtag = observed_dict['windowtag'][idx_assessment])

            # convert val_min and val_max to number of hours since start of day
            val_min = observed_dict['assessment_begin'][idx_assessment] - val_min/60
            val_max = observed_dict['assessment_begin'][idx_assessment] - val_max/60

            if observed_dict['windowtag'][idx_assessment]==4:
                val_max = observed_dict['assessment_begin'][idx_assessment]

            # Grab current delay
            curr_delay = observed_dict['delay'][idx_assessment]
            # Calculate probabilities
            prob_upper_bound = norm.cdf(x = curr_true_time - val_min, loc = curr_true_time, scale = curr_delay)
            prob_lower_bound = norm.cdf(x = curr_true_time - val_max, loc = curr_true_time, scale = curr_delay)
            prob_positive_recall = 1-norm.cdf(x = 0, loc = curr_true_time, scale = curr_delay)
            c = (prob_upper_bound - prob_lower_bound)/prob_positive_recall
            prob_reported.extend([c])

        elif observed_dict['smoke'][idx_assessment]=='Yes' and observed_dict['assessment_type'][idx_assessment]=='random_ema':
            # Grab true latent time matched to current reported time
            idx_matched_latent_event = observed_dict['matched_latent_event'][idx_assessment]
            idx_matched_latent_event = np.int64(idx_matched_latent_event)
            curr_true_time = latent_dict['hours_since_start_day'][idx_matched_latent_event]
            # Grab current reported time
            val_min, val_max = convert_windowtag_random_ema(windowtag = observed_dict['windowtag'][idx_assessment])

            # convert val_min and val_max to number of hours since start of day
            val_min = observed_dict['assessment_begin'][idx_assessment] - val_min/60
            val_max = observed_dict['assessment_begin'][idx_assessment] - val_max/60

            if observed_dict['windowtag'][idx_assessment]==6:
                val_max = observed_dict['assessment_begin'][idx_assessment]

            # Grab current delay
            curr_delay = observed_dict['delay'][idx_assessment]
            # Calculate probabilities
            prob_upper_bound = norm.cdf(x = curr_true_time - val_min, loc = curr_true_time, scale = curr_delay)
            prob_lower_bound = norm.cdf(x = curr_true_time - val_max, loc = curr_true_time, scale = curr_delay)
            prob_positive_recall = 1-norm.cdf(x = 0, loc = curr_true_time, scale = curr_delay)
            c = (prob_upper_bound - prob_lower_bound)/prob_positive_recall
            prob_reported.extend([c])

        else:
            next

prob_reported = np.array(prob_reported)







