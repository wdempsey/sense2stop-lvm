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
# Read in dictionary where Self-Report, Random EMA, and puffmarker are knitted together into one data frame
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_knitted_with_puffmarker')
infile = open(filename,'rb')
dict_knitted_with_puffmarker = pickle.load(infile)
infile.close()

# Output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'setup-day-limits.py')).read())

# %%
# Create "mock" true latent times using dict_knitted_with_puffmarker
# Then we will test this first assuming that observed times come from Self-Report and/or Random EMA only

latent_data = {}

for participant in dict_knitted_with_puffmarker.keys():
    current_participant_dict = {}
    for days in dict_knitted_with_puffmarker[participant].keys():
        current_data = dict_knitted_with_puffmarker[participant][days]
        all_puff_time = []
        if len(current_data.index)==0:
            next
        else:
            # note that even if participant reported "Yes" smoked, 
            # delta is set to missing if in Self Report participant reported to have smoked "more than 30 minutes ago"
            # or in Random EMAs where participant reported to have smoked "more than 2 hours ago"
            current_data_yes = current_data[(current_data['smoke']=='Yes') & ~(np.isnan(current_data['delta']))]
            if len(current_data_yes)==0:
                next
            else:
                for this_row in range(0, len(current_data_yes.index)):
                    if current_data_yes['adjusted'].iloc[this_row]==1:
                        tmp = current_data_yes['puff_time_adjusted'].iloc[this_row]
                        # tmp is an array with 1 element, and this element is a dictionary
                        # hence, we need the next steps
                        tmp = tmp[0]
                        tmp_to_array = []
                        for pm_key in tmp.keys():
                            tmp_to_array.append(tmp[pm_key])
                        # we're out of the loop
                        all_puff_time.extend(tmp_to_array)
                    elif current_data_yes['adjusted'].iloc[this_row]==0:
                        all_puff_time.append(current_data_yes['puff_time_adjusted'].iloc[this_row])
                    else:
                        next

        # The output of the next line should be one number only, but it will be an array object
        # Hence, the .iloc[0] converts the array object into a float
        current_day_length = data_day_limits[(data_day_limits['participant_id']==participant) & (data_day_limits['study_day']==days)]['day_length'].iloc[0]
        new_dict = {'participant_id':participant, 
                    'study_day':days, 
                    'day_length':current_day_length, 
                    'latent_event_order': (np.arange(len(all_puff_time))),  # begins with zero (not 1)
                    'hours_since_start_day': np.array(all_puff_time)}
        current_participant_dict.update({days: new_dict})
    # Add this participant's data to dictionary
    latent_data.update({participant:current_participant_dict})

# %%
'''
Building a latent poisson process model for smoking times
Input: Latent smoking times for a given PARTICIPANT-DAY
Output: log-likelihood for a given fixed set of parameters
Ex1: PP homogeneous
'''

def latent_poisson_process_ex1(latent_dict, params):
    '''
    latent_dict: a dictionary containing the keys 'day_length', 'hours_since_start_day'
    params: a dictionary containing the keys 'lambda'
    '''
    m = len(latent_dict['hours_since_start_day'])
    total = m*np.log(params['lambda']) - params['lambda']*np.sum(latent_dict['hours_since_start_day'])
    return total


# %%
'''
Building a latent poisson process model for smoking times
Input: Latent smoking times for a given PARTICIPANT-DAY
Output: log-likelihood for a given fixed set of parameters
Ex2: PP is homogenous within pre-quit and within-postquit periods; quit day is day 4
'''

def latent_poisson_process_ex2(latent_dict, params):
    '''
    latent_dict: a dictionary containing the keys 'day_length', 'hours_since_start_day', 'study_day'
    params: a dictionary containing the keys 'lambda_prequit' and 'lambda_postquit'
    '''

    if latent_dict['study_day'] < 4:
        m = len(latent_dict['hours_since_start_day'])
        total = m*np.log(params['lambda_prequit']) - params['lambda_prequit']*np.sum(latent_dict['hours_since_start_day'])
    else:
        m = len(latent_dict['hours_since_start_day'])
        total = m*np.log(params['lambda_postquit']) - params['lambda_postquit']*np.sum(latent_dict['hours_since_start_day'])

    return total


# %%
# Test out the function
use_this_id = None
use_this_days = None

