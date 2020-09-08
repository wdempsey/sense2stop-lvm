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
        # e.g., use_params={'lambda':0.99}
        
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
    use_this_window_min = {1: 0/60, 2: 5/60, 3: 15/60, 4: np.nan} # may change 4:np.nan later on
    use_this_window_max = {1: 5/60, 2: 15/60, 3: 30/60, 4: np.nan} # may change 4:np.nan later on

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
    use_this_window_min = {1: 1/60, 2: 20/60, 3: 40/60, 4: 60/60, 5: 80/60, 6: np.nan} # may change 6:np.nan later on
    use_this_window_max = {1: 19/60, 2: 39/60, 3: 59/60, 4: 79/60, 5: 100/60, 6: np.nan} # may change 6:np.nan later on

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

    prob_reported = np.array(prob_reported)
    return(prob_reported)

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
            probs_reported = selfreport_mem(observed_dict = observed_dict, latent_dict = latent_dict)
            reported_total_loglik = np.sum(np.log(probs_reported))
            current_total_loglik += reported_total_loglik
        # This next condition says that there is at least one observed event that is NOT matched to a latent event
        else:
            current_total_loglik = -np.inf

    return current_total_loglik


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

class model(object):
    '''
    Define the model as a latent object and a list of mem objects
    '''
    
    def __init__(self, init=0, latent=0, model=0):
        self.data = init # Initial smoking estimates
        self.latent = latent # Latent smoking process model
        self.memmodel = model # Measurement-error model
    
    def update_params(self, new_params):
        self.params = new_params
    
    def adapMH_params(self, 
                      adaptive = False, 
                      iteration = 1, 
                      covariance = 0, 
                      barX = 0, 
                      covariance_init = 0, 
                      barX_init = 0, 
                      cutpoint = 500,
                      sigma = 0, 
                      bartau = 0.574,
                      current_sum_X = 0,
                      current_count_accept = 0):
        '''
        Builds an adaptive MH for updating model parameter.
        If adaptive = True 
        then use "An adaptive metropolis algorithm" Haario et al (2001)
        to perform adaptive updates.
        bartau = optimal acceptance rate (here, default is 0.574)
        '''

        # self.latent.params is a dictionary
        # latent_params contains the same values as self.latent.params
        # except that it is a numpy array
        latent_params = np.array(list(self.latent.params.values()))

        # if no adaptive learning, simply perturb current value of
        # latent_params by MVN(0_{size}, 0.01*1_{size x size})
        if adaptive is False:
            #new_params = np.exp(np.log(latent_params) + np.random.normal(scale = 0.01, size = latent_params.size))
            logX_current = np.random.normal(scale = 0.01, size = latent_params.size)
            new_params = np.exp(logX_current)   # new_params is on the exp scale!
        # Next, if adaptive learning is used ...
        else:
            sd = 2.38**2 / latent_params.size  # specific choice of value for sd
            # Note: covariance_init.shape will be of the form (num_rows, num_cols)
            # where num_rows = num_cols = total number of parameters
            # in self.latent.params.values(), i.e., parameters we wish to estimate
            if iteration <= cutpoint:
                if covariance_init.shape[0] > 1:
                    #new_params = np.exp(np.log(latent_params)+ np.random.multivariate_normal(mean = barX_init, cov = sd * covariance_init))
                    logX_current = np.random.multivariate_normal(mean = barX_init, cov = sd * covariance_init)  # make a draw from the proposal distribution
                    new_params = np.exp(logX_current)  # new_params is on the exp scale!
                else:
                    # in this case, there is only 1 parameter in self.latent.params.values()
                    #new_params = np.exp(np.log(latent_params)+ np.random.normal(loc = barX_init, scale = np.sqrt(sd * covariance_init)))
                    logX_current = np.random.normal(loc = barX_init, scale = np.sqrt(sd * covariance_init))  # make a draw from the proposal distribution
                    new_params = np.exp(logX_current)  # new_params is on the exp scale!
            else:
                if covariance_init.shape[0] > 1:
                    #new_params =  np.exp(np.log(self.latent.params) + np.random.multivariate_normal(mean = barX_init, cov = (sigma**2) * covariance))
                    logX_current = np.random.multivariate_normal(mean = barX, cov = (sigma**2) * covariance)
                    new_params = np.exp(logX_current)  # new_params is on the exp scale!
                else:
                    # in this case, there is only 1 parameter in self.latent.params.values()
                    #new_params =  np.exp(np.log(self.latent.params) + np.random.normal(loc = barX_init, scale = sigma*np.sqrt(covariance_init)))
                    logX_current = np.random.normal(loc = barX, scale = sigma*np.sqrt(covariance))
                    new_params = np.exp(logX_current)  # new_params is on the exp scale!

        # Calculate loglikelihood given current value of latent_params
        # We indicate that the current value of latent_params is used
        # by setting the argument of self.latent.compute_total_pp to None
        llik_current = self.latent.compute_total_pp(None)

        # Before proceeding ...
        # self.latent.params has to be updated with new values
        # the new values come from new_params
        # new_params are perturbed values of self.latent_params
        idx_keys_count = 0
        for idx_keys in self.latent.params.keys():
            self.latent.params[idx_keys] = new_params[idx_keys_count]
            idx_keys_count += 1

        # At this point, self.latent_params has already been udpated
        # We now calculate the log-likelihood using the 'jittered' values
        llik_jitter = self.latent.compute_total_pp(use_params = self.latent.params)
        log_acceptprob = (llik_jitter-llik_current)
        acceptprob = np.exp(log_acceptprob)
        acceptprob = np.min([acceptprob,1])
        # temp is equal to 1 with probability acceptprob
        # temp is equal to 0 with probability 1-acceptprob
        #temp = np.random.binomial(1, p = acceptprob)  


        # since llik_current or llik_jitter might be inf
        # we use a try statement
        try:
            temp = np.random.binomial(1, p = acceptprob)  

            # Decision: reject proposal ###########################################
            # This function does not return anything
            if temp == 0:
                rejected = 1-temp
                out_dict = {'rejected':rejected}
                return out_dict

            # Decision: accept proposal ########################################### 
            # In this case, temp==1
            if temp==1:
                if adaptive is True: # Update Covariance and barX
                    # note that the update proposed by Haario et al is on the LOG scale
                    log_new_params = np.log(new_params)
                    current_sum_X = current_sum_X + log_new_params
                    current_average_X = current_sum_X/(current_count_accept+1)
                    delta = log_new_params - current_average_X
                    barX_new = barX + 1/(current_count_accept+1) * (delta)
                    
                    intermediate_step = np.outer(delta, delta)
                    #sigma_new = sigma + 1/iteration * (acceptprob - bartau)
                    #sigma_new = sigma + 1/(current_count_accept+1) * (acceptprob - bartau)
                    #sigma_new=2.38**2/idx_keys_count
                    sigma_new=1

                    if iteration > 1:
                        #covariance_new = covariance + 1/(iteration-1) * ( intermediate_step * iteration/(iteration-1) - covariance )
                        covariance_new = covariance + 1/(current_count_accept+1) * ( intermediate_step - covariance )
                    else: 
                        covariance_new = covariance
                    
                    rejected = 1-temp
                    current_count_accept =  current_count_accept + temp
                    out_dict = {'rejected':rejected,
                                'new_params':new_params,
                                'covariance_new':covariance_new,
                                'barX_new':barX_new,
                                'sigma_new':sigma_new,
                                'current_sum_X':current_sum_X,
                                'current_count_accept': current_count_accept}   
                    return out_dict
                else:
                    # adaptive is False
                    rejected = 1-temp
                    current_count_accept =  current_count_accept + temp
                    out_dict = {'rejected':rejected, 'new_params':new_params, 'current_count_accept': current_count_accept}  # store indicator for whether the parameter was rejected and new_params (on the EXP scale)
                    return out_dict

        except:
            # display llik when temp is nan
            out_dict = {'rejected':-1, 'acceptprob':acceptprob, 'llik_jitter':llik_jitter, 'llik_current':llik_current, 'latent_params':self.latent.params}
            return out_dict

# %%%
# Test out class
tmp_latent_data = copy.deepcopy(latent_data)
tmp_clean_data = copy.deepcopy(clean_data)

lat_pp = latent(data=tmp_latent_data, model=latent_poisson_process_ex2, params = {'lambda_prequit': 0.30, 'lambda_postquit': 0.50})
sr_mem = measurement_model(data=tmp_clean_data, model=selfreport_mem_total, latent = tmp_latent_data, model_params={'p':0.8})
test_model = model(init = clean_data,  latent = lat_pp , model = sr_mem)

# %%
num_iters = 5000
use_cutpoint = 500

cov_init = np.array([[0.05,0.0],[0.0,0.05]])
barX_init = np.array([0.10,0.30])
cov_new = np.array([[0.01,0.0],[0.0,0.01]])
barX_new = np.log(np.array(list(lat_pp.params.values())))
sigma_new = 1 #2.38**2/len(lat_pp.params.keys())
current_sum_X = 0
current_count_accept = 0

# %%
dict_accept = {}
count_accept = 0

for iter in range(1,num_iters):
    current_out_dict = test_model.adapMH_params(adaptive = True,
                                                covariance = cov_new, 
                                                barX = barX_new,
                                                covariance_init = cov_init, 
                                                barX_init = barX_init,
                                                iteration = iter, 
                                                cutpoint = use_cutpoint, 
                                                sigma = sigma_new,
                                                current_sum_X = current_sum_X,
                                                current_count_accept = current_count_accept)
    
    if current_out_dict['rejected'] == 0:  # if not rejected
        barX_new = current_out_dict['barX_new']
        sigma_new = current_out_dict['sigma_new']
        cov_new = current_out_dict['covariance_new']
        current_count_accept = current_out_dict['current_count_accept']
        current_sum_X = current_out_dict['current_sum_X']
        lat_pp.update_params(new_params = {'lambda_prequit':current_out_dict['new_params'][0], 'lambda_postquit':current_out_dict['new_params'][1]})

        if iter > use_cutpoint:
            # display values after cutpoint
            print(current_out_dict['new_params'])
            dict_accept.update({count_accept:current_out_dict})
            count_accept = count_accept+1
    elif current_out_dict['rejected'] == -1:
        next
    else:
        next

print(count_accept/(num_iters - use_cutpoint))


# %%
temp = np.zeros(shape = (count_accept, len(lat_pp.params.keys())))

for iter in range(0,count_accept):
    temp[iter,:] = dict_accept[iter]['new_params']

# %%
plt.plot(np.arange(count_accept), temp[:,0])

# %%
plt.hist(temp[:,0], bins=40)

# %%
plt.plot(np.arange(count_accept), temp[:,1])

# %%
plt.hist(temp[:,1], bins=40)


# %%
