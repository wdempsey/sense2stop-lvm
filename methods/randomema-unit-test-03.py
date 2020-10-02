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

# Read in dictionary where Self-Report, Random EMA, and puffmarker are knitted together into one data frame
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_knitted_with_puffmarker')
infile = open(filename,'rb')
dict_knitted_with_puffmarker = pickle.load(infile)
infile.close()

# Output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'setup-day-limits.py')).read())

# Create mock latent and observed data
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'create-input-data-03.py')).read())

# note:
# output of the above script is
# latent_data
# clean_data

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
    elif (len(observed_dict['windowtag'])>0) and (len(latent_dict['hours_since_start_day'])==0):
        # this is the case when all responses are 'NO'
        observed_dict['matched_latent_event'] = np.array([])
        observed_dict['delay'] = np.array([])
        latent_dict['matched'] = np.array([])
    else:
        next
    
    return observed_dict, latent_dict

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
    use_this_window_min = {1: 0/60, 2: 20/60, 3: 40/60, 4: 60/60, 5: 80/60, 6: 100/60}
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
def personday_mem(observed_dict, latent_dict, personday_mem_params = {'lambda_delay_sr': 7.59}):

    array_check_flag = []
    prob_reported = []
    prob_delay = []
    log_prob_reported = []
    log_prob_delay = []
    
    tot_measurements = len(observed_dict['windowtag'])
    tot_true_latent_events = len(latent_dict['hours_since_start_day'])

    if tot_measurements>0 and tot_true_latent_events>0:
        for idx_assessment in range(0, tot_measurements):
            if observed_dict['smoke'][idx_assessment]=='Yes' and observed_dict['assessment_type'][idx_assessment]=='selfreport':
                # Grab true latent time matched to current reported time
                idx_matched_latent_event = observed_dict['matched_latent_event'][idx_assessment]
                if np.isnan(idx_matched_latent_event):
                    prob_reported.extend([1]) # does not contribute towards mem likelihood
                    prob_delay.extend([np.nan])

                    array_check_flag.extend([-222])
                    
                else:
                    idx_matched_latent_event = np.int64(idx_matched_latent_event)
                    curr_true_time = latent_dict['hours_since_start_day'][idx_matched_latent_event]
                    # Grab current reported time
                    # val_min and val_max have been converted to hours
                    val_min, val_max = convert_windowtag_selfreport(windowtag = observed_dict['windowtag'][idx_assessment])

                    # convert val_min and val_max to number of hours since start of day
                    val_min = observed_dict['assessment_begin'][idx_assessment] - val_min
                    val_max = observed_dict['assessment_begin'][idx_assessment] - val_max

                    # note: val_max is earlier in the day after val_min after the above calculation
                    # a truncated distribution will be used
                    # need to check whether assessment_begin_shifted < val_min < assessment_begin
                    # need to check whether assessment_begin_shifted < val_max < assessment_begin

                    if observed_dict['windowtag'][idx_assessment]==4:
                        val_max = 0                     
                        check_val_max = True
                        check_val_min = (val_min >= observed_dict['assessment_begin_shifted'][idx_assessment])
                    else:
                        check_val_max = (val_max >= observed_dict['assessment_begin_shifted'][idx_assessment])
                        check_val_min = (val_min >= observed_dict['assessment_begin_shifted'][idx_assessment])


                    if idx_assessment == 0:
                        if check_val_max and check_val_min:
                            # Grab current delay
                            curr_delay = observed_dict['delay'][idx_assessment]
                            # Calculated probability of reporting "between t1 to t2 hours ago"
                            prob_upper_bound = norm.cdf(x = val_min, loc = curr_true_time, scale = curr_delay)
                            prob_lower_bound = norm.cdf(x = val_max, loc = curr_true_time, scale = curr_delay)
                            lower_bound_constraint = norm.cdf(x = observed_dict['assessment_begin_shifted'][idx_assessment], 
                                                            loc = curr_true_time, 
                                                            scale = curr_delay)
                            upper_bound_constraint = norm.cdf(x = observed_dict['assessment_begin'][idx_assessment], 
                                                            loc = curr_true_time, 
                                                            scale = curr_delay)
                            tot_prob_constrained = upper_bound_constraint - lower_bound_constraint
                            c = (prob_upper_bound - prob_lower_bound)/tot_prob_constrained
                            d = personday_mem_params['lambda_delay_sr']*np.exp(-(personday_mem_params['lambda_delay_sr'])*(observed_dict['delay'][idx_assessment]))  

                            check_flag = 0
                        else:
                            c = 0
                            d = personday_mem_params['lambda_delay_sr']*np.exp(-(personday_mem_params['lambda_delay_sr'])*(observed_dict['delay'][idx_assessment]))
                            check_flag = -999
                    else:
                        if check_val_max and check_val_min:
                            # Grab current delay
                            curr_delay = observed_dict['delay'][idx_assessment]
                            # Calculated probability of reporting "between t1 to t2 hours ago"
                            prob_upper_bound = norm.cdf(x = val_min, loc = curr_true_time, scale = curr_delay)
                            prob_lower_bound = norm.cdf(x = val_max, loc = curr_true_time, scale = curr_delay)
                            lower_bound_constraint = norm.cdf(x = observed_dict['assessment_begin_shifted'][idx_assessment], 
                                                            loc = curr_true_time, 
                                                            scale = curr_delay)
                            upper_bound_constraint = norm.cdf(x = observed_dict['assessment_begin'][idx_assessment], 
                                                            loc = curr_true_time, 
                                                            scale = curr_delay)
                            tot_prob_constrained = upper_bound_constraint - lower_bound_constraint
                            c = (prob_upper_bound - prob_lower_bound)/tot_prob_constrained
                            d = personday_mem_params['lambda_delay_sr']*np.exp(-(personday_mem_params['lambda_delay_sr'])*(observed_dict['delay'][idx_assessment]))

                            check_flag = 0

                        elif (~check_val_max) and check_val_min:
                            # is previous EMA 'Yes'?
                            if observed_dict['smoke'][idx_assessment-1] == 'Yes':
                                # Grab current delay
                                curr_delay = observed_dict['delay'][idx_assessment]
                                # Calculated probability of reporting "between t1 to t2 hours ago"
                                prob_upper_bound = norm.cdf(x = val_min, loc = curr_true_time, scale = curr_delay)
                                prob_lower_bound = norm.cdf(x = val_max, loc = curr_true_time, scale = curr_delay)
                                upper_bound_constraint = norm.cdf(x = observed_dict['assessment_begin'][idx_assessment], 
                                                                loc = curr_true_time, 
                                                                scale = curr_delay)
                                tot_prob_constrained = upper_bound_constraint
                                c = (prob_upper_bound - prob_lower_bound)/tot_prob_constrained
                                d = personday_mem_params['lambda_delay_sr']*np.exp(-(personday_mem_params['lambda_delay_sr'])*(observed_dict['delay'][idx_assessment]))

                                check_flag = 0

                            # is previous EMA 'No'?
                            elif observed_dict['smoke'][idx_assessment-1] == 'No':
                                # Grab current delay
                                curr_delay = observed_dict['delay'][idx_assessment]
                                # Calculated probability of reporting "between t1 to t2 hours ago"
                                prob_upper_bound = norm.cdf(x = val_min, loc = curr_true_time, scale = curr_delay)
                                prob_lower_bound = norm.cdf(x = val_max, loc = curr_true_time, scale = curr_delay)
                                lower_bound_constraint = norm.cdf(x = observed_dict['assessment_begin_shifted'][idx_assessment], 
                                                                loc = curr_true_time, 
                                                                scale = curr_delay)
                                upper_bound_constraint = norm.cdf(x = observed_dict['assessment_begin'][idx_assessment], 
                                                                loc = curr_true_time, 
                                                                scale = curr_delay)
                                tot_prob_constrained = upper_bound_constraint - lower_bound_constraint
                                c = (prob_upper_bound - prob_lower_bound)/tot_prob_constrained
                                d = personday_mem_params['lambda_delay_sr']*np.exp(-(personday_mem_params['lambda_delay_sr'])*(observed_dict['delay'][idx_assessment]))

                                check_flag = 0
                            else:
                                c = 0
                                d = personday_mem_params['lambda_delay_sr']*np.exp(-(personday_mem_params['lambda_delay_sr'])*(observed_dict['delay'][idx_assessment]))  
                                check_flag = -555         
                        elif (~check_val_max) and (~check_val_min):
                            c = 0
                            d = personday_mem_params['lambda_delay_sr']*np.exp(-(personday_mem_params['lambda_delay_sr'])*(observed_dict['delay'][idx_assessment]))
                            check_flag = -666
                        else:
                            c = 0
                            d = personday_mem_params['lambda_delay_sr']*np.exp(-(personday_mem_params['lambda_delay_sr'])*(observed_dict['delay'][idx_assessment]))
                            check_flag = -444
                   
                   # Append for self-report assessment
                    prob_reported.extend([c])
                    prob_delay.extend([d])

                    array_check_flag.extend([check_flag])
                    

            elif observed_dict['smoke'][idx_assessment]=='Yes' and observed_dict['assessment_type'][idx_assessment]=='random_ema':
                # Grab true latent time matched to current reported time
                idx_matched_latent_event = observed_dict['matched_latent_event'][idx_assessment]
                if np.isnan(idx_matched_latent_event):
                    prob_reported.extend([1]) # does not contribute towards mem likelihood
                    prob_delay.extend([np.nan])

                    array_check_flag.extend([-222])
                    
                else:
                    idx_matched_latent_event = np.int64(idx_matched_latent_event)
                    curr_true_time = latent_dict['hours_since_start_day'][idx_matched_latent_event]
                    # Grab current reported time
                    # val_min and val_max have been converted to hours
                    val_min, val_max = convert_windowtag_random_ema(windowtag = observed_dict['windowtag'][idx_assessment])

                    # convert val_min and val_max to number of hours since start of day
                    val_min = observed_dict['assessment_begin'][idx_assessment] - val_min
                    val_max = observed_dict['assessment_begin'][idx_assessment] - val_max

                    # note: val_max is earlier in the day after val_min after the above calculation
                    if observed_dict['windowtag'][idx_assessment]==6:
                        val_max = 0                     
                        check_val_max = True
                        check_val_min = (val_min >= observed_dict['assessment_begin_shifted'][idx_assessment])
                    else:
                        check_val_max = (val_max >= observed_dict['assessment_begin_shifted'][idx_assessment])
                        check_val_min = (val_min >= observed_dict['assessment_begin_shifted'][idx_assessment])

                    if idx_assessment == 0:
                        if check_val_max and check_val_min:
                            # Grab current delay
                            curr_delay = observed_dict['delay'][idx_assessment]
                            # Calculated probability of reporting "between t1 to t2 hours ago"
                            prob_upper_bound = norm.cdf(x = val_min, loc = curr_true_time, scale = curr_delay)
                            prob_lower_bound = norm.cdf(x = val_max, loc = curr_true_time, scale = curr_delay)
                            lower_bound_constraint = norm.cdf(x = observed_dict['assessment_begin_shifted'][idx_assessment], 
                                                            loc = curr_true_time, 
                                                            scale = curr_delay)
                            upper_bound_constraint = norm.cdf(x = observed_dict['assessment_begin'][idx_assessment], 
                                                            loc = curr_true_time, 
                                                            scale = curr_delay)
                            tot_prob_constrained = upper_bound_constraint - lower_bound_constraint
                            c = (prob_upper_bound - prob_lower_bound)/tot_prob_constrained
                            d = np.nan

                            check_flag = 0
                        else:
                            c = 0
                            d = np.nan
                            check_flag = -999
                    else:
                        if check_val_max and check_val_min:
                            # Grab current delay
                            curr_delay = observed_dict['delay'][idx_assessment]
                            # Calculated probability of reporting "between t1 to t2 hours ago"
                            prob_upper_bound = norm.cdf(x = val_min, loc = curr_true_time, scale = curr_delay)
                            prob_lower_bound = norm.cdf(x = val_max, loc = curr_true_time, scale = curr_delay)
                            lower_bound_constraint = norm.cdf(x = observed_dict['assessment_begin_shifted'][idx_assessment], 
                                                            loc = curr_true_time, 
                                                            scale = curr_delay)
                            upper_bound_constraint = norm.cdf(x = observed_dict['assessment_begin'][idx_assessment], 
                                                            loc = curr_true_time, 
                                                            scale = curr_delay)
                            tot_prob_constrained = upper_bound_constraint - lower_bound_constraint
                            c = (prob_upper_bound - prob_lower_bound)/tot_prob_constrained
                            d = np.nan

                            check_flag = 0

                        elif (~check_val_max) and check_val_min:
                            # is previous EMA 'Yes'?
                            if observed_dict['smoke'][idx_assessment-1] == 'Yes':
                                # Grab current delay
                                curr_delay = observed_dict['delay'][idx_assessment]
                                # Calculated probability of reporting "between t1 to t2 hours ago"
                                prob_upper_bound = norm.cdf(x = val_min, loc = curr_true_time, scale = curr_delay)
                                prob_lower_bound = norm.cdf(x = val_max, loc = curr_true_time, scale = curr_delay)
                                upper_bound_constraint = norm.cdf(x = observed_dict['assessment_begin'][idx_assessment], 
                                                                loc = curr_true_time, 
                                                                scale = curr_delay)
                                tot_prob_constrained = upper_bound_constraint
                                c = (prob_upper_bound - prob_lower_bound)/tot_prob_constrained
                                d = np.nan

                                check_flag = 0

                            # is previous EMA 'No'?
                            elif observed_dict['smoke'][idx_assessment-1] == 'No':
                                # Grab current delay
                                curr_delay = observed_dict['delay'][idx_assessment]
                                # Calculated probability of reporting "between t1 to t2 hours ago"
                                prob_upper_bound = norm.cdf(x = val_min, loc = curr_true_time, scale = curr_delay)
                                prob_lower_bound = norm.cdf(x = val_max, loc = curr_true_time, scale = curr_delay)
                                lower_bound_constraint = norm.cdf(x = observed_dict['assessment_begin_shifted'][idx_assessment], 
                                                                loc = curr_true_time, 
                                                                scale = curr_delay)
                                upper_bound_constraint = norm.cdf(x = observed_dict['assessment_begin'][idx_assessment], 
                                                                loc = curr_true_time, 
                                                                scale = curr_delay)
                                tot_prob_constrained = upper_bound_constraint - lower_bound_constraint
                                c = (prob_upper_bound - prob_lower_bound)/tot_prob_constrained
                                d = np.nan

                                check_flag = 0
                            else:
                                c = 0
                                d = np.nan
                                check_flag = -555       
                        
                        elif (~check_val_max) and (~check_val_min):
                            c = 0
                            d = personday_mem_params['lambda_delay_sr']*np.exp(-(personday_mem_params['lambda_delay_sr'])*(observed_dict['delay'][idx_assessment]))
                            check_flag = -666
                        else:
                            c = 0
                            d = np.nan
                            check_flag = -444
                   
                   # Append for self-report assessment
                    prob_reported.extend([c])
                    prob_delay.extend([d])

                    array_check_flag.extend([check_flag])                    

            # Cases when a 'No' smoking was reported in a Random EMA
            else:  #observed_dict['smoke'][idx_assessment]=='No':
                previous_time = observed_dict['assessment_begin_shifted'][idx_assessment]
                current_time = observed_dict['assessment_begin'][idx_assessment]
                all_true_smoking_times = latent_dict['hours_since_start_day']

                any_detected = 0
                for j in range(0, len(all_true_smoking_times)):
                    current_check = (all_true_smoking_times[j] >= previous_time) and (all_true_smoking_times[j] <= current_time)
                    if current_check is True:
                        any_detected = 1
                        break

                if any_detected == 0:
                    c = 1
                else:
                    c = 0

                prob_reported.extend([c])
                prob_delay.extend([np.nan])
                array_check_flag.extend([-888])

        # Format output
        array_check_flag = np.array(array_check_flag)

        prob_reported = np.array(prob_reported)
        log_prob_reported = np.log(prob_reported)
        prob_delay = np.array(prob_delay)
        log_prob_delay = np.log(prob_delay)
    
    elif tot_measurements>0 and tot_true_latent_events==0:  # participant reported smoking 'no' all throughout
        for idx_assessment in range(0, tot_measurements):
            if observed_dict['smoke'][idx_assessment]=='No': 
                prob_reported.extend([1])
                prob_delay.extend([np.nan])

                array_check_flag.extend([-888])
            else:
                prob_reported.extend([0])
                prob_delay.extend([np.nan])   
                array_check_flag.extend([-111])

        prob_reported = np.array(prob_reported)
        log_prob_reported = np.log(prob_reported)
        prob_delay = np.array(prob_delay)
        log_prob_delay = np.log(prob_delay)

    else:
        pass

    observed_dict['array_check_flag'] = array_check_flag

    observed_dict['prob_reported'] = prob_reported
    observed_dict['log_prob_reported'] = log_prob_reported
    observed_dict['prob_delay'] = prob_delay
    observed_dict['log_prob_delay'] = log_prob_delay

    return observed_dict

# %%
def personday_mem_total_loglik(observed_dict, latent_dict, mem_params = {'p':0.9,'lambda_delay_sr': 7.59}):
    """
    Calculates total likelihood for a given PARTICIPANT-DAY
    """
    
    observed_dict = personday_mem(observed_dict = observed_dict, latent_dict = latent_dict, personday_mem_params = mem_params)
    tot_measurements = len(observed_dict['windowtag'])
    tot_true_latent_events = len(latent_dict['hours_since_start_day'])
    total_matched = np.sum(latent_dict['matched'])

    if tot_measurements>0 and tot_true_latent_events>0:
        total_loglik = total_matched * np.log(mem_params['p']) + (tot_true_latent_events - total_matched) * np.log(1-mem_params['p'])

        for idx_assessment in range(0, tot_measurements):
            if observed_dict['smoke'][idx_assessment]=='Yes' and observed_dict['assessment_type'][idx_assessment]=='selfreport':
                total_loglik = total_loglik + observed_dict['log_prob_delay'][idx_assessment]
                total_loglik = total_loglik + observed_dict['log_prob_reported'][idx_assessment]
            elif observed_dict['smoke'][idx_assessment]=='Yes' and observed_dict['assessment_type'][idx_assessment]=='random_ema':
                total_loglik = total_loglik + observed_dict['log_prob_reported'][idx_assessment]
            else:  # 'No' is reported
                total_loglik = total_loglik + observed_dict['log_prob_reported'][idx_assessment]
        
    elif tot_measurements>0 and tot_true_latent_events==0:
        total_loglik = 0
        for idx_assessment in range(0, tot_measurements):
            total_loglik = total_loglik + observed_dict['log_prob_reported'][idx_assessment]
    elif tot_measurements==0 and tot_true_latent_events>0:
        total_loglik = total_matched * np.log(mem_params['p']) + (tot_true_latent_events - total_matched) * np.log(1-mem_params['p'])
    else:
        total_loglik = np.nan
    
    return total_loglik


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
        observed = self.data[id][days]
        latent = self.latent[id][days]
        # Note that the argument params is set up so that all PARTICIPANT-DAYs utilize the same set of params
        # This could optionally be changed in the future to have each PARTICIPANT-DAY have a different set of params
        total = self.model(observed_dict = observed, latent_dict = latent, mem_params = self.model_params)
        return total
    
    def compute_total_mem(self):
        total = 0 
        for use_this_id in self.data.keys():
            for use_this_days in self.data[use_this_id].keys():
                observed = self.data[use_this_id][use_this_days]
                latent = self.latent[use_this_id][use_this_days]
                res = self.model(observed_dict = observed, latent_dict = latent, mem_params = self.model_params)
                if np.isnan(res):
                    pass
                else:
                    total += res
        return total

# %%
def latent_poisson_process(latent_dict, params):
    '''
    latent_dict: a dictionary containing the keys 'day_length', 'hours_since_start_day', 'study_day'
    params: a dictionary containing the keys 'lambda_prequit' and 'lambda_postquit'
    '''

    if latent_dict['study_day'] < 4:
        m = len(latent_dict['hours_since_start_day'])
        total_loglik = m*np.log(params['lambda_prequit']) - params['lambda_prequit']*np.sum(latent_dict['hours_since_start_day'])
    else:
        m = len(latent_dict['hours_since_start_day'])
        total_loglik = m*np.log(params['lambda_postquit']) - params['lambda_postquit']*np.sum(latent_dict['hours_since_start_day'])

    return total_loglik

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
    def compute_total_pp(self, use_params=None):
        # e.g., use_params={'lambda':0.99}
        
        if use_params is None:
            use_params = self.params
        
        # Initialize log-likelihood to zero
        total = 0 
        for participant in self.data.keys():
            for days in self.data[participant].keys():
                # Grab latent data corresponding to current PARTICIPANT-DAY
                current_latent_dict = self.data[participant][days]
                # Add contribution of current PARTICIPANT-DAY to total log-likelihood
                total += self.model(latent_dict = current_latent_dict, params = use_params)
        # We are done
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
                      bartau = 0.574):
        '''
        Builds an adaptive MH for updating model parameter.
        If adaptive = True 
        then use "An adaptive metropolis algorithm" Haario et al (2001)
        to perform adaptive updates.
        bartau = optimal acceptance rate (here, default is 0.574)
        '''

        epsilon = 0.10

        # self.latent.params is a dictionary
        # latent_params contains the same values as self.latent.params
        # except that it is a numpy array
        latent_params = np.array(list(self.latent.params.values()))
        num_latent_params = len(latent_params)

        # if no adaptive learning, simply perturb current value of
        # latent_params by MVN(0_{size}, 0.01*1_{size x size})
        if adaptive is False:
            # new_params is on the exp scale!
            new_params = np.exp(np.log(latent_params) + np.random.normal(loc = 0, scale = .000001, size = num_latent_params))  
        # Next, if adaptive learning is used ...
        else:
            sd = 2.38**2 / latent_params.size
            if iteration <= cutpoint:
                if covariance_init.shape[0] > 1:
                    #new_params = np.exp(np.log(latent_params) + np.random.multivariate_normal(mean = np.repeat(0,num_latent_params), cov = (sd**2) * covariance_init))
                    new_params = np.exp(np.random.multivariate_normal(mean = np.log(latent_params), cov = (sd**2) * covariance_init + (sd**2) * epsilon * np.eye(num_latent_params)) )
                else:
                    #new_params = np.exp(np.log(latent_params) + np.random.normal(loc = np.repeat(0,num_latent_params), scale = sd * np.sqrt(covariance_init)))
                    new_params = np.exp(np.random.normal(loc = np.log(latent_params), scale = sd * np.sqrt(covariance_init) + sd * epsilon * np.eye(num_latent_params)) )
            else:
                if covariance.shape[0] > 1:
                    #new_params = np.exp(np.log(latent_params) + np.random.multivariate_normal(mean = np.repeat(0,num_latent_params), cov = (sigma**2) * covariance + (sigma**2) * epsilon * np.eye(num_latent_params)))
                    if sigma < 1e-4:
                        sigma = 1e-4

                    new_params = np.exp(np.random.multivariate_normal(mean = np.log(latent_params), cov = (sigma**2) * covariance + (sigma**2) * epsilon * np.eye(num_latent_params)) )
                else:
                    #new_params = np.exp(np.log(latent_params) + np.random.normal(loc = np.repeat(0,num_latent_params), scale = sigma * np.sqrt(covariance) + sigma * epsilon * np.eye(num_latent_params)))
                    if sigma < 1e-4:
                        sigma = 1e-4

                    new_params = np.exp(np.random.normal(loc = np.log(latent_params), scale = sigma * np.sqrt(covariance) + sigma * epsilon * np.eye(num_latent_params)) )

        # Calculate loglikelihood given current value of latent_params
        # We indicate that the current value of latent_params is used
        # by setting the argument of self.latent.compute_total_pp to None
        llik_current = self.latent.compute_total_pp(None)

        # Calculate loglikelihood given proposed value of latent_params
        idx_keys_count = 0
        new_dict_latent_params = copy.deepcopy(self.latent.params)
        for idx_keys in new_dict_latent_params.keys():
            new_dict_latent_params[idx_keys] = new_params[idx_keys_count]
            idx_keys_count += 1

        # We now calculate the log-likelihood using the 'jittered' values of the parameters
        llik_jitter = self.latent.compute_total_pp(use_params = new_dict_latent_params)
        log_acceptprob = (llik_jitter-llik_current)
        acceptprob = np.exp(log_acceptprob)
        acceptprob = np.min([acceptprob,1])

        if adaptive is False: 
            try:  
                temp = np.random.binomial(1, p = acceptprob)
                rejected = 1-temp
                if temp == 0:
                    # new_dict_latent_params is new_params but in dictionary form
                    out_dict = {'rejected':rejected,
                                'new_params': self.latent.params}
                else:
                    out_dict = {'rejected':rejected, 
                                'new_params':new_dict_latent_params}
            except:
                # display llik when temp is nan
                out_dict = {'rejected':-1, 'acceptprob':acceptprob, 'llik_jitter':llik_jitter, 'llik_current':llik_current}
        else:
            try:  
                temp = np.random.binomial(1, p = acceptprob)
                rejected = 1-temp
                
                if temp==1:
                    log_new_params = np.log(new_params)
                else:
                    log_new_params = np.log(latent_params)


                sigma_new = sigma + 1/iteration * (acceptprob - bartau) 
                delta = log_new_params-barX
                barX_new = barX + 1/iteration * (delta)
                intermediate_step = np.outer(delta, delta)
                

                if iteration==1:
                    covariance_new = covariance
                else:
                    covariance_new = covariance + 1/(iteration-1) * ( intermediate_step * iteration/(iteration-1) - covariance ) 
                    #covariance_new = covariance + 1/(iteration-1) * ( intermediate_step - covariance ) 

                if rejected==0:
                    out_dict = {'rejected':rejected, 'new_params':new_dict_latent_params,
                                'barX_new':barX_new, 'covariance_new':covariance_new, 
                                'sigma_new':sigma_new, 'log_new_params':log_new_params,
                                'acceptprob':acceptprob}
                else:
                    out_dict = {'rejected':rejected, 'new_params':self.latent.params,
                                'barX_new':barX_new, 'covariance_new':covariance_new, 
                                'sigma_new':sigma_new, 'log_new_params':log_new_params,
                                'acceptprob':acceptprob}                    
            except:
                # display llik when temp is nan
                out_dict = {'rejected':-1, 'acceptprob':acceptprob, 'llik_jitter':llik_jitter, 'llik_current':llik_current}

        return out_dict


    def adapMH_times(self):
        '''
        Builds an adaptive MH for updating the latent smoking times 
        Current: Simple Jitter
        '''
        total_possible_jitter = 0.
        total_accept_jitter = 0.

        for participant in self.data.keys():
            for days in self.data[participant].keys():
                
                observed_dict = self.memmodel.data[participant][days]  # observed_dict
                latent_dict = self.latent.data[participant][days] # latent_dict
                tot_measurements = len(observed_dict['windowtag'])
                tot_true_latent_events = len(latent_dict['hours_since_start_day'])
                new_latent_dict = copy.deepcopy(latent_dict)

                
                if tot_measurements>0 and tot_true_latent_events>0: 
                    try:
                        total_possible_jitter += 1.
                        llik_mem_current = self.memmodel.model(observed_dict = observed_dict, latent_dict = latent_dict, mem_params = {'p':0.9, 'lambda_delay_sr':7.59})
                        llik_current = self.latent.model(latent_dict = latent_dict, params = self.latent.params)
                        new_latent_dict['hours_since_start_day'] = new_latent_dict['hours_since_start_day'] + np.random.normal(scale = 2.5/60., size=tot_true_latent_events)
                        llik_mem_jitter = self.memmodel.model(observed_dict = observed_dict, latent_dict = new_latent_dict, mem_params = {'p':0.9, 'lambda_delay_sr':7.59})
                        llik_jitter = self.latent.model(latent_dict = new_latent_dict, params = self.latent.params)

                        log_acceptprob = (llik_jitter-llik_current) + (llik_mem_jitter-llik_mem_current)
                        acceptprob = np.exp(log_acceptprob)
                        temp = np.random.binomial(1, p = np.min([acceptprob,1]))
                        if temp == 1:
                            total_accept_jitter += 1.
                            latent_dict['hours_since_start_day'] = copy.deepcopy(new_latent_dict['hours_since_start_day'])
                    except:
                        pass
                    
                elif tot_measurements>0 and tot_true_latent_events==0:  # participant reported smoking 'no' all throughout
                    pass # do not jitter

                elif tot_measurements==0 and tot_true_latent_events>0: 
                    pass  # do not jitter
                else:
                    pass  # do not jitter
        
        return total_accept_jitter/total_possible_jitter



# %%
if False:
    # Perform match for each PARTICIPANT-DAY
    for participant in clean_data.keys():
        for days in clean_data[participant].keys():
            clean_data[participant][days], latent_data[participant][days] = matching(observed_dict = clean_data[participant][days], latent_dict = latent_data[participant][days])

    for participant in clean_data.keys():
        for day in current_participant_dict.keys():
            observed_dict = copy.deepcopy(clean_data[participant][day])
            latent_dict = copy.deepcopy(latent_data[participant][day])
            current_loglik = personday_mem_total_loglik(observed_dict = observed_dict, latent_dict = latent_dict)
            if np.isnan(current_loglik) and observed_dict['array_check_flag']!=[]:
                print("do check")

# %%

if False:
    # Perform match for each PARTICIPANT-DAY
    for participant in clean_data.keys():
        for days in clean_data[participant].keys():
            clean_data[participant][days], latent_data[participant][days] = matching(observed_dict = clean_data[participant][days], latent_dict = latent_data[participant][days])


    total_possible_jitter = 0.
    total_accept_jitter = 0.

    for participant in clean_data.keys():
        for days in clean_data[participant].keys():
            observed_dict = clean_data[participant][days]  # observed_dict
            latent_dict = latent_data[participant][days]  # latent_dict
            tot_measurements = len(observed_dict['windowtag'])
            tot_true_latent_events = len(latent_dict['hours_since_start_day'])
            
            if tot_measurements>0 and tot_true_latent_events>0: 
                try:
                    total_possible_jitter += 1. 
                    llik_mem_current = personday_mem_total_loglik(observed_dict = observed_dict, latent_dict = latent_dict, mem_params = {'p':0.9, 'lambda_delay_sr':7.59})
                    llik_current= latent_poisson_process(latent_dict = latent_dict, params = {'lambda_prequit': 1, 'lambda_postquit': 1})
                    new_latent_dict = copy.deepcopy(latent_dict)
                    new_latent_dict['hours_since_start_day'] = new_latent_dict['hours_since_start_day'] + np.random.normal(scale = 2.5/60., size=tot_true_latent_events)
                    llik_mem_jitter = personday_mem_total_loglik(observed_dict = observed_dict, latent_dict = new_latent_dict, mem_params = {'p':0.9, 'lambda_delay_sr':7.59})
                    llik_jitter = latent_poisson_process(latent_dict = new_latent_dict, params = {'lambda_prequit': 1, 'lambda_postquit': 1})

                    log_acceptprob = (llik_jitter-llik_current) + (llik_mem_jitter-llik_mem_current)
                    acceptprob = np.exp(log_acceptprob)
                    temp = np.random.binomial(1, p = np.min([acceptprob,1]))
                    if temp == 1:
                        total_accept_jitter += 1.
                        latent_dict['hours_since_start_day'] = new_latent_dict['hours_since_start_day']  
                except:
                    pass

            elif tot_measurements>0 and tot_true_latent_events==0:  # participant reported smoking 'no' all throughout
                pass # do not jitter

            elif tot_measurements==0 and tot_true_latent_events>0: 
                pass  # do not jitter
            else:
                pass  # do not jitter

    
    print(total_possible_jitter)
    print(total_accept_jitter)


# %%
# Perform match for each PARTICIPANT-DAY
for participant in clean_data.keys():
    for days in clean_data[participant].keys():
        clean_data[participant][days], latent_data[participant][days] = matching(observed_dict = clean_data[participant][days], latent_dict = latent_data[participant][days])

# %%
tmp_latent_data = copy.deepcopy(latent_data)
tmp_clean_data = copy.deepcopy(clean_data)
lat_pp = latent(data=tmp_latent_data, model=latent_poisson_process, params = {'lambda_prequit': 1, 'lambda_postquit': 1})
big_mem = measurement_model(data=tmp_clean_data, model=personday_mem_total_loglik, latent = tmp_latent_data, model_params={'p':0.9, 'lambda_delay_sr':7.59})
test_model = model(init = clean_data,  latent = lat_pp , model = big_mem)

# %%
num_iters = 15000 #105000
use_cutpoint = 5000
np.random.seed(seed = 412983)

dict_store_params = {}
cov_init = ((.000001*2)/(2.38**2))*np.eye(2)
barX_init = np.array([0., 0.])


# %%

for current_iter in range(1,num_iters):
    print(current_iter)
    
    if current_iter == 1:
        current_out_dict = test_model.adapMH_params(adaptive = True,
                                                    covariance = cov_init, 
                                                    barX = barX_init,
                                                    covariance_init = cov_init, 
                                                    barX_init = barX_init,
                                                    iteration = current_iter, 
                                                    cutpoint = use_cutpoint, 
                                                    sigma = 1)
        # Store parameters
        lat_pp.update_params(new_params = {'lambda_prequit':current_out_dict['new_params']['lambda_prequit'],
                                           'lambda_postquit':current_out_dict['new_params']['lambda_postquit']})
        dict_store_params.update({current_iter:current_out_dict})
        
        # Update params
        cov_new = current_out_dict['covariance_new']
        sigma_new = current_out_dict['sigma_new']
        barX_new = current_out_dict['barX_new']

        dict_store_params[current_iter]["acceptprob_jitter"] = np.nan

    else:
        current_out_dict = test_model.adapMH_params(adaptive = True,
                                                    covariance = cov_new, 
                                                    barX = barX_new,
                                                    covariance_init = cov_init, 
                                                    barX_init = barX_init,
                                                    iteration = current_iter, 
                                                    cutpoint = use_cutpoint, 
                                                    sigma = sigma_new)
        # Store parameters
        lat_pp.update_params(new_params = {'lambda_prequit':current_out_dict['new_params']['lambda_prequit'],
                                           'lambda_postquit':current_out_dict['new_params']['lambda_postquit']})
        dict_store_params.update({current_iter:current_out_dict})

        # Update params
        cov_new = current_out_dict['covariance_new']
        sigma_new = current_out_dict['sigma_new']
        barX_new = current_out_dict['barX_new']
        



# %%
# Print out acceptance probability
cnt = 0

for iter in range(use_cutpoint+1, num_iters):
    cnt = cnt + dict_store_params[iter]['rejected']

accept_prob = 1 - cnt/(num_iters - (use_cutpoint+1))
print(accept_prob)

# %%
cnt = 0

for iter in range(1, 5000):
    cnt = cnt + dict_store_params[iter]['rejected']

burn_in_accept_prob = 1 - cnt/(5000-1)
print(burn_in_accept_prob)



# %%
temp = np.zeros(shape = (num_iters, 2+len(lat_pp.params.keys())))

for iter in range(1,num_iters):
    temp[iter,0] = dict_store_params[iter]['new_params']['lambda_prequit']
    temp[iter,1] = dict_store_params[iter]['new_params']['lambda_postquit']
    temp[iter,2] = dict_store_params[iter]['sigma_new']
    temp[iter,3] = dict_store_params[iter]['acceptprob']

# %%
plot_cutpoint = use_cutpoint + 1

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
df = pd.DataFrame(data=temp[plot_cutpoint:,3].flatten())
ma = df.rolling(window=100).mean()
plt.plot(ma)