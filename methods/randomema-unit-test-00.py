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
                        observed_dict['matched_latent_event'][idx_assessment] = latent_dict['latent_event_order'][which_idx]
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



