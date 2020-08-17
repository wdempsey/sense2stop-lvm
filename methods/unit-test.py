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
execute_test = False

if execute_test:
    use_this_id = None
    use_this_days = None
    # Test out the function latent_poisson_process_ex2
    # pre-quit
    print(latent_poisson_process_ex2(latent_dict = latent_data[use_this_id][use_this_days], params = {'lambda_prequit': 0.14, 'lambda_postquit': 0.75}))
    # post-quit
    print(latent_poisson_process_ex2(latent_dict = latent_data[use_this_id][use_this_days], params = {'lambda': 0.14, 'lambda_postquit': 0.75}))

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

lat_pp_ex1.update_params(new_params = {'lambda': 0.77})
print(lat_pp_ex1.model)
print(lat_pp_ex1.params)
print(lat_pp_ex1.compute_total_pp(use_params = None))

# %%
# Another test on the class
lat_pp_ex2 = latent(data=latent_data, model=latent_poisson_process_ex2, params = {'lambda_prequit': 0.14, 'lambda_postquit': 0.75})
print(lat_pp_ex2.model)
print(lat_pp_ex2.params)
print(lat_pp_ex2.compute_total_pp(use_params = None))

lat_pp_ex2.update_params(new_params = {'lambda_prequit': 0.05, 'lambda_postquit': 0.25})
print(lat_pp_ex2.model)
print(lat_pp_ex2.params)
print(lat_pp_ex2.compute_total_pp(use_params = None))

# %%
# Create "mock" observed data
clean_data = copy.deepcopy(dict_knitted_with_puffmarker)  # Keep dict_knitted_with_puffmarker untouched

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
# Let's simply use Self-Reports for now as out "mock" observed data
for participant in clean_data.keys():
    for days in clean_data[participant].keys():
        current_data = clean_data[participant][days]
        if len(current_data.index)>0:
            current_data = current_data[current_data['assessment_type']=="selfreport"]
            # Remove Self-Reports for "more than 30 minutes ago"
            current_data = current_data[current_data['windowtag']!=4]
            # Create variable: order at which the participant initiated a particular Self-Report
            current_data['assessment_order'] = np.arange(len(current_data.index))  # begins with zero (not 1)
            current_data = current_data.loc[:, ['assessment_order','assessment_begin', 'smoke', 'windowtag']]
            clean_data[participant][days] = current_data
        
# %%
# Now, let's convert each PERSON-DAY of clean_data into a dictionary
for participant in clean_data.keys():
    for days in clean_data[participant].keys():
        current_data = clean_data[participant][days]
        if len(current_data.index)>0:
            current_dict = {'participant_id':participant,
                            'study_day': days,
                            'assessment_order': np.array(current_data['assessment_order']),
                            'assessment_begin': np.array(current_data['assessment_begin']),
                            'smoke': np.array(current_data['smoke']),
                            'windowtag': np.array(current_data['windowtag'])}
            clean_data[participant][days] = current_dict
        else:
            current_dict = {'participant_id':participant,
                            'study_day': days,
                            'assessment_order': np.array([]),
                            'assessment_begin': np.array([]),
                            'smoke': np.array([]),
                            'windowtag': np.array([])}
            clean_data[participant][days] = current_dict

# %%   

# Now, set up matching function

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
            # current observed Self-Report will be matched to the latent event occurring immediately prior to it
            # if that latent event has not yet been matched to any observed event
            this_scalar = observed_dict['assessment_begin'][idx_assessment]
            which_idx = (latent_dict['hours_since_start_day'] < this_scalar)  # check: this line will work if there are no latent_dict['hours_since_start_day'] or observed_dict['assessment_begin'] that are EXACTLY zero
            which_idx = np.where(which_idx)
            which_idx = np.max(which_idx)
            if latent_dict['matched'][which_idx] == False:
                latent_dict['matched'][which_idx] =  True  # A match has been found!
                observed_dict['matched_latent_event'][idx_assessment] = latent_dict['latent_event_order'][which_idx]
                observed_dict['delay'][idx_assessment] = observed_dict['assessment_begin'][idx_assessment] - latent_dict['hours_since_start_day'][which_idx]
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
        if len(current_data['windowtag'])>0:
            tmp_array_delay = np.array(current_data['delay'])
            tmp_array_delay = tmp_array_delay[~np.isnan(tmp_array_delay)]  # nan's occur when an observed time is not matched with a latent time
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
# Define functions for Self Report MEM
def generate_recall_times(arr_latent_times, arr_delay):
    arr_recall_times = []
    for i in range(0, len(arr_latent_times)):
        # If delay is tiny, variance will be very small. If delay is huge, variance will be huge
        # Note that draw_time_i can be negative
        # This means that recall time is before start of participant-day (time zero)
        draw_time_i = norm.rvs(loc = arr_latent_times[i], scale = arr_delay[i], size = 1)
        arr_recall_times.extend(draw_time_i) 

    arr_recall_times = np.array(arr_recall_times)
    return arr_recall_times

# %%
# Test out the function
execute_test = False

if execute_test:
    use_participant = None
    use_days = None
    tmp_clean_data = copy.deepcopy(clean_data[use_participant][use_days])  # keep clean_data[use_participant][use_days] untouched
    tmp_latent_data = copy.deepcopy(latent_data[use_participant][use_days])  # keep latent_data[use_participant][use_days] untouched
    res = generate_recall_times(arr_latent_times = tmp_latent_data['hours_since_start_day'][tmp_latent_data['matched']], arr_delay = tmp_clean_data['delay'])
    print(res)

# %%
def selfreport_mem(observed_dict, latent_dict):
    """
    Prob that participant reports smoked between use_value_min to use_value_max ago
    equal to prob that participant recalled smoking between 
    use_value_min to use_value_max ago given that the time the participant recalled
    is after start of study (recall>0)
    apply function only after generating recall times
    """

    prob_reported = []
    if len(observed_dict['windowtag'])>0:
        for idx_assessment in range(0, len(observed_dict['windowtag'])):
            if observed_dict['recall'][idx_assessment] >=0:
                # Grab true latent time matched to current reported time
                idx_matched_latent_event = observed_dict['matched_latent_event'][idx_assessment]
                idx_matched_latent_event = np.int64(idx_matched_latent_event)
                curr_true_time = latent_dict['hours_since_start_day'][idx_matched_latent_event]
                # Grab current reported time and convert val_min and val_max from minutes to hours
                val_min, val_max = convert_windowtag_selfreport(windowtag = observed_dict['windowtag'][idx_assessment])
                val_min = val_min/60
                val_max = val_max/60
                # Grab current delay
                curr_delay = observed_dict['delay'][idx_assessment]
                # Calculate probabilities
                prob_upper_bound = norm.cdf(x = curr_true_time - val_min, loc = curr_true_time, scale = curr_delay)
                prob_lower_bound = norm.cdf(x = curr_true_time - val_max, loc = curr_true_time, scale = curr_delay)
                prob_positive_recall = 1-norm.cdf(x = 0, loc = curr_true_time, scale = curr_delay)
                c = (prob_upper_bound - prob_lower_bound)/prob_positive_recall
                prob_reported.extend([c])
            else:
                # This is the case when recall time is negative
                # That is participant recalled event to have happened before start of day (time zero)
                # Note that log(0) = -infty. Hence, this case needs to be carefully handled in calculations that follow
                prob_reported.extend([0])

    prob_reported = np.array(prob_reported)
    return(prob_reported)


# %%
# Test out the function
execute_test = False

if execute_test:
    use_participant = None
    use_days = None
    tmp_clean_data = copy.deepcopy(clean_data[use_participant][use_days])  # keep clean_data[use_participant][use_days] untouched
    tmp_latent_data = copy.deepcopy(latent_data[use_participant][use_days])
    if len(tmp_latent_data['matched']) > 0:
        tmp_clean_data['recall'] = generate_recall_times(arr_latent_times = tmp_latent_data['hours_since_start_day'][tmp_latent_data['matched']], arr_delay = tmp_clean_data['delay'])
        res = selfreport_mem(observed_dict = tmp_clean_data, latent_dict = tmp_latent_data)
        print(res)

# %%

def selfreport_mem_total(observed_dict, latent_dict, params):
    """
    Calculates total LOG-likelihood for a given PARTICIPANT-DAY
    """
    current_total_loglik = 0

    # Only proceed if there is any data in observed_dict
    if len(observed_dict['windowtag'])>0:
        m = len(latent_dict['matched'])
        total_matched = sum(latent_dict['matched'])
        current_total_loglik = total_matched*np.log(params['p']) + (m - total_matched)*np.log(1-params['p'])

        # The first condition says that there is NO observed event that is not matched to a latent event
        # since an unmatched observed event is indicated by a missing value in observed_dict['matched_latent_event']
        if np.sum(np.isnan(observed_dict['matched_latent_event']))==0:
            observed_dict['recall'] = generate_recall_times(arr_latent_times = latent_dict['hours_since_start_day'][latent_dict['matched']], arr_delay = observed_dict['delay'])
            probs_reported = selfreport_mem(observed_dict = observed_dict, latent_dict = latent_dict)
            positive_probs_reported = probs_reported[probs_reported>0]
            reported_total_loglik = np.sum(np.log(positive_probs_reported))
            current_total_loglik += reported_total_loglik
        # This next condition says that there is at least one observed event that is NOT matched to a latent event
        else:
            current_total_loglik = -np.inf

    return current_total_loglik

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

class measurement_model(object):
    '''
    This class constructs a measurement error subcomponent
    Attributes: 
        Data: Must provide the observed data
        Model: Computes prob of measurements given latent variables
    '''
    def __init__(self, data=0, model=0, latent = 0, model_params=0):
        # Attributes of an object of class measurement_model
        self.data = data
        self.latent = latent
        self.model = model
        self.model_params = model_params
    
    def update_params(self, new_params):
        self.model_params = new_params

    def compute_mem_userday(self, id, days):
        # Note that selfreport_mem will already give the log-likelihood oer USER-DAY
        # Hence, there is NO NEED to take the logarithm within this function
        total = 0 
        observed = self.data[id][days]
        latent = self.latent[id][days]
        # Note that the argument params is set up so that all PARTICIPANT-DAYs utilize the same set of params
        # This could optionally be changed in the future to have each PARTICIPANT-DAY have a different set of params
        total += self.model(observed_dict = observed, latent_dict = latent, params = self.model_params)
        return total
    
    def compute_total_mem(self):
        total = 0 
        for use_this_id in self.data.keys():
            for use_this_days in self.data[use_this_id].keys():
                observed = self.data[use_this_id][use_this_days]
                latent = self.latent[use_this_id][use_this_days]
                res = self.model(observed_dict = observed, latent_dict = latent, params = self.model_params)
                total += res
        return total


# %% 
# Test out the class
tmp_clean_data = copy.deepcopy(clean_data)  # keep clean_data untouched
tmp_latent_data = copy.deepcopy(latent_data)  # keep latent_data untouched
sr_mem = measurement_model(data=tmp_clean_data, model=selfreport_mem_total, latent = tmp_latent_data, model_params={'p':0.9})
print(sr_mem.model_params)
print(sr_mem.compute_total_mem())
sr_mem.update_params(new_params = {'p':0.4})
print(sr_mem.model_params)
print(sr_mem.compute_total_mem())

# %%
