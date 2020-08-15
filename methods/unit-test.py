# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt
import scipy.special as sc
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

# Test out the function latent_poisson_process_ex1
# pre-quit
print(latent_poisson_process_ex1(latent_dict = latent_data[224][2], params = {'lambda': 0.14}))
# post-quit
print(latent_poisson_process_ex1(latent_dict = latent_data[224][5], params = {'lambda': 0.14}))


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

# Test out the function latent_poisson_process_ex2
# pre-quit
print(latent_poisson_process_ex2(latent_dict = latent_data[224][2], params = {'lambda_prequit': 0.14, 'lambda_postquit': 0.75}))
# post-quit
print(latent_poisson_process_ex2(latent_dict = latent_data[224][5], params = {'lambda': 0.14, 'lambda_postquit': 0.75}))

# %%
class latent(object):
    '''
    This class defines the latent process
    Attributes:
        Initial data: a first initialization of the latent process
        Model: one of the functions latent_poisson_process_ex
        Params: A dictionary of parameters for Model
    '''
    
    # Note: the attributes data, model and params are initialized zero
    # so that, for example, the statement lat_pp_ex1 = latent()
    # will result in
    # 
    # lat_pp_ex1.data >>> 0
    # lat_pp_ex1.model >>> 0
    # lat_pp_ex1.params >>> 0 
    def __init__(self, data=0, model=0, params=0):
        self.data = data
        self.model = model
        self.params = params        
    
    def update_params(self, new_params):
        self.params = new_params
    
    # Compute contribution to log-likelihood for a given PARTICIPANT-DAY
    # and then sum across all PARTICIPANT-DAYs
    def compute_total_pp(self, use_params):
        if use_params is None:
            use_params = self.params
        
        # Initialize log-likelihood to zero
        total = 0 
        for participant in self.data.keys():
            for days in self.data[participant].keys():
                # Grab latent data corresponding to current PARTICIPANT-DAY
                current_latent_data = self.data[participant][days]
                # Add contribution of current PARTICIPANT-DAY to total log-likelihood
                total += self.model(latent_dict = current_latent_data, params = use_params)
        # We are done
        return total


# %%
# Test out the class
lat_pp_ex1 = latent(data=latent_data, model=latent_poisson_process_ex1, params = {'lambda': 0.14})
print(lat_pp_ex1.model)
print(lat_pp_ex1.params)
print(lat_pp_ex1.compute_total_pp(use_params = None))
print(lat_pp_ex1.compute_total_pp(use_params = {'lambda': 0.77}))
print(lat_pp_ex1.params)  # params remain unchanged

# %%
# Another test on the class
lat_pp_ex2 = latent(data=latent_data, model=latent_poisson_process_ex2, params = {'lambda_prequit': 0.14, 'lambda_postquit': 0.75})
print(lat_pp_ex2.model)
print(lat_pp_ex2.params)
print(lat_pp_ex2.compute_total_pp(use_params = None))
print(lat_pp_ex2.compute_total_pp(use_params = {'lambda_prequit': 0.05, 'lambda_postquit': 0.25}))
print(lat_pp_ex2.params)  # params remain unchanged

# %%
# Define functions for creating "mock" observed data
def convert_windowtag_selfreport(windowtag):
    accept_response = [1,2,3,4]
    # windowtag is in hours
    use_this_window_min = {1: 0/60, 2: 5/60, 3: 15/60, 4: np.nan} 
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
    use_this_window_min = {1: 1/60, 2: 20/60, 3: 40/60, 4: 60/60, 5: 80/60, 6: np.nan} 
    use_this_window_max = {1: 19/60, 2: 39/60, 3: 59/60, 4: 79/60, 5: 100/60, 6: np.nan} 

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
# Create "mock" observed data
clean_data = dict_knitted_with_puffmarker

for participant in clean_data.keys():
    for days in clean_data[participant].keys():
        current_data = clean_data[participant][days]
        if len(current_data.index)>0:
            current_data = current_data.loc[:, ['assessment_type', 'hours_since_start_day', 'hours_since_start_day_shifted','smoke','when_smoke']]
            current_data = current_data.rename(columns = {'assessment_type':'assessment_type',
                                                          'hours_since_start_day':'assessment_begin', 
                                                          'hours_since_start_day_shifted':'assessment_begin_shifted',
                                                          'smoke':'smoke',
                                                          'when_smoke':'windowtag'})
            clean_data[participant][days] = current_data


# %%



