# %%
from multiprocessing import Pool
import time
import numpy as np
from scipy.stats import mvn
import os
import pickle
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm

# %%
exec(open('../../env_vars.py').read())
dir_picklejar = os.environ['dir_picklejar']

filename = os.path.join(os.path.realpath(dir_picklejar), 'data_day_limits')
infile = open(filename,'rb')
data_day_limits = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'init_latent_data_small')
infile = open(filename,'rb')
init_dict_latent_data = pickle.load(infile)  # Initialization of the latent smoking times
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_eod_survey')
infile = open(filename,'rb')
init_dict_observed_eod_survey = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_all_ema')
infile = open(filename,'rb')
init_dict_observed_ema = pickle.load(infile)
infile.close()

# %%
def grow_tree(depth):
    if depth==1:
        current_data = list([0,1])
        return current_data

    elif depth > 1:
        curr_level = 1
        current_data = list([0,1])

        curr_level = 2
        while curr_level <= depth:
            # Sweep through all leaves at the current level
            list_curr_level = list(np.repeat(np.nan, repeats=2**curr_level))
            for i in range(0, len(current_data)):
                left_leaf = np.append(np.array(current_data[i]), 0)
                right_leaf = np.append(np.array(current_data[i]), 1)
                list_curr_level[2*i] = list(left_leaf)
                list_curr_level[2*i + 1] = list(right_leaf)
                
            # Go one level below
            current_data = list_curr_level
            curr_level += 1
        return current_data

    else:
        return 0


# %%

class Latent:
    '''
    A collection of objects and methods related to latent process subcomponent
    '''
    def __init__(self, participant = None, day = None, latent_data = None, params = None, index = None):
        self.participant = participant
        self.day = day
        self.latent_data = copy.deepcopy(latent_data)
        self.params = copy.deepcopy(params)
        self.index = index

    def update_params(self, new_params):
        '''
        Update parameters
        '''
        self.params = copy.deepcopy(new_params)
    
    def calc_loglik(self):  
        '''
        Calculate loglikelihood for latent process subcomponent
        ''' 
        smoking_times = self.latent_data['hours_since_start_day']
        day_length = self.latent_data['day_length']
        lambda_prequit = self.params['lambda_prequit']
        lambda_postquit = self.params['lambda_postquit']
        
        # Calculate the total number of latent smoking times in the current iteration
        m = len(smoking_times)

        # lambda_prequit: number of events per hour during prequit period
        # lambda_postquit: number of events per hour during postquit period
        # day_length: total number of hours between wakeup time to sleep time on a given participant day
        if self.day <4:
            lik = np.exp(-lambda_prequit*day_length) * ((lambda_prequit*day_length) ** m) / np.math.factorial(m)
            loglik = np.log(lik)
        else:
            lik = np.exp(-lambda_postquit*day_length) * ((lambda_postquit*day_length) ** m) / np.math.factorial(m)
            loglik = np.log(lik)

        return loglik


# %%
class EODSurvey:
    '''
    A collection of objects and methods related to end-of-day survey subcomponent
    '''

    def __init__(self, participant = None, day = None, latent_data = None, observed_data = None, params = None, index = None):
        self.participant = participant
        self.day = day
        self.latent_data = copy.deepcopy(latent_data)
        self.observed_data = copy.deepcopy(observed_data)
        self.params = copy.deepcopy(params)
        self.index = index

    def update_params(self, new_params):
        '''
        Update parameters
        '''
        self.params = copy.deepcopy(new_params)

    def calc_loglik(self):
        '''
        Calculate loglikelihood corresponding to end-of-day EMA subcomponent
        '''
        
        # Inputs to be checked ----------------------------------------------------------------------------
        any_eod_ema = len(self.observed_data['assessment_begin'])

        if any_eod_ema > 0:
            # Begin after checks on inputs have been passed ---------------------------------------------------            
            # Go through each box one by one
            collect_box_probs = np.array([])
            arr_ticked = self.observed_data['ticked_box_raw']  # which boxes were ticked?
            m = len(self.latent_data['hours_since_start_day'])  # are there any latent smoking events?
            all_boxes = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20])

            if (m == 0) and (len(arr_ticked) == 0):
                collect_box_probs = np.repeat(1, len(all_boxes))
            elif (m == 0) and (len(arr_ticked) > 0): 
                collect_box_probs = np.repeat(0, len(all_boxes))
            else:
                start_day = 0
                end_day = 24
                
                # Rescale time to be within 24 hour clock
                all_true_smoke_times = self.latent_data['hours_since_start_day'] + self.observed_data['start_time_hour_of_day']
                
                for k in range(0, len(all_boxes)):
                    curr_box = all_boxes[k] # lower limit of Box k; setting curr_lk and curr_box to be separate variables in case change of scale is needed for curr_lk
                    curr_lk = all_boxes[k] # lower limit of Box k
                    curr_uk = curr_lk + 1 # upper limit of Box k; add one hour to lower limit
                    recall_epsilon = self.params['recall_epsilon'] # in hours
                    num_points_to_sample = self.params['budget']

                    if len(all_true_smoke_times) <=  num_points_to_sample:
                        true_smoke_times = all_true_smoke_times
                    else:
                        true_smoke_times = all_true_smoke_times[(all_true_smoke_times > curr_lk - recall_epsilon) * (all_true_smoke_times < curr_uk + recall_epsilon)]
                        if len(true_smoke_times) > num_points_to_sample:
                            true_smoke_times = np.random.choice(a = true_smoke_times, size = num_points_to_sample, replace = False)
                    
                    # At this point, the length of true_smoke_times will always be at most num_points_to_sample
                    if len(true_smoke_times) > 0:
                        # Specify covariance matrix based on an exchangeable correlation matrix
                        rho = self.params['rho']
                        use_cormat = np.eye(len(true_smoke_times)) + rho*(np.ones((len(true_smoke_times),1)) * np.ones((1,len(true_smoke_times))) - np.eye(len(true_smoke_times)))
                        use_sd = self.params['sd']
                        use_covmat = (use_sd**2) * use_cormat
                        
                        # Calculate total possible probability
                        total_possible_prob, error_code_total_possible_prob = mvn.mvnun(lower = np.repeat(start_day, len(true_smoke_times)),
                                                                                        upper = np.repeat(end_day, len(true_smoke_times)),
                                                                                        means = true_smoke_times,
                                                                                        covar = use_covmat)

                        # Begin calculating edge probabilities
                        collect_edge_probabilities = np.array([])
                        limits_of_integration = grow_tree(depth=len(true_smoke_times))

                        for j in range(0, len(limits_of_integration)):
                            curr_limits = np.array(limits_of_integration[j])
                            curr_lower_limits = np.where(curr_limits==0, start_day, curr_uk)
                            curr_upper_limits = np.where(curr_limits==0, curr_lk, end_day)
                            edge_probabilities, error_code_edge_probabilities = mvn.mvnun(lower = curr_lower_limits,
                                                                                          upper = curr_upper_limits, 
                                                                                          means = true_smoke_times, 
                                                                                          covar = use_covmat)
                            collect_edge_probabilities = np.append(collect_edge_probabilities, edge_probabilities)

                        total_edge_probabilities = np.sum(collect_edge_probabilities)
                        
                        prob_none_recalled_within_current_box = total_edge_probabilities/total_possible_prob

                        # prob_none_recalled_within_current_box may be slightly above 1, e.g., 1.000000XXXXX
                        if (prob_none_recalled_within_current_box-1) > 0:
                            prob_none_recalled_within_current_box = 1

                        prob_at_least_one_recalled_within_box = 1-prob_none_recalled_within_current_box                    

                    else:
                        prob_none_recalled_within_current_box = 1
                        prob_at_least_one_recalled_within_box = 1-prob_none_recalled_within_current_box    

                    # Exit the first IF-ELSE statement
                    if curr_box in arr_ticked:
                        collect_box_probs = np.append(collect_box_probs, prob_at_least_one_recalled_within_box)
                    else:
                        collect_box_probs = np.append(collect_box_probs, prob_none_recalled_within_current_box)


            # Exit if-else statement
            prob_observed_box_checking_pattern = np.prod(collect_box_probs)
            loglik = np.log(prob_observed_box_checking_pattern)

            self.observed_data['prob_bk'] = collect_box_probs
            self.observed_data['product_prob_bk'] = prob_observed_box_checking_pattern
            self.observed_data['log_product_prob_bk'] = loglik

        else:
            # If participant did not complete EOD survey, then this measurement type should NOT contribute to the loglikelihood
            loglik = 0

        return loglik


# %%
class SelfReport:

    def __init__(self, participant = None, day = None, latent_data = None, observed_data = None, params = None, index = None):
        self.participant = participant
        self.day = day
        self.latent_data = copy.deepcopy(latent_data)
        self.observed_data = copy.deepcopy(observed_data)
        self.params = copy.deepcopy(params)
        self.index = index

    def update_params(self, new_params):
        '''
        Update parameters
        '''
        self.params = copy.deepcopy(new_params)    

    def match(self):
        '''
        Matches each EMA with one latent smoking time occurring before the Self Report EMA
        After a latent smoking time is matched, it is removed
        '''

        # Inputs to be checked --------------------------------------------
        all_latent_times = self.latent_data['hours_since_start_day']
        tot_ema = len(self.observed_data['assessment_type'])

        if tot_ema > 0:
            self.observed_data['matched_latent_time'] = np.repeat(np.nan, tot_ema)
            remaining_latent_times = copy.deepcopy(all_latent_times)
            remaining_latent_times = np.sort(remaining_latent_times)
            for i in range(0, tot_ema):
                current_lb = self.observed_data['assessment_begin_shifted'][i]
                current_ub = self.observed_data['assessment_begin'][i]
                #current_assessment_type = self.observed_data['assessment_type'][i]
                which_within = (remaining_latent_times >= 0) & (remaining_latent_times < current_ub)
                if np.sum(which_within)>0:
                    which_idx = np.where(which_within)
                    matched_idx = np.max(which_idx)
                    matched_latent_time = remaining_latent_times[matched_idx]
                    self.observed_data['matched_latent_time'][i] = matched_latent_time
                    remaining_latent_times = np.delete(remaining_latent_times, matched_idx)
                    remaining_latent_times = np.sort(remaining_latent_times)
                else:
                    # This case can occur when between time 0 and time t there is no
                    # latent smoking time, but a self-report occurred between time 0 and time t
                    # This case may happen after a dumb death move
                    self.observed_data['matched_latent_time'][i] = np.nan
        else:
            self.observed_data['matched_latent_time'] = np.array([])


    def calc_loglik(self):
        '''
        Call the method calc_loglik after the method match has been called
        Calculate loglikelihood corresponding to self report EMA subcomponent
        '''

        # Inputs to be checked --------------------------------------------
        all_latent_times = np.sort(self.latent_data['hours_since_start_day'])
        tot_latent_events = len(all_latent_times)

        if len(self.observed_data['assessment_type']) == 0:
            tot_sr = 0
        else:
            # Total number of Self-Report
            tot_sr = np.sum(self.observed_data['assessment_type']=='selfreport')  

        # Specify parameter values ----------------------------------------
        lambda_delay = self.params['lambda_delay']
        use_scale = self.params['sd']
        prob_reporting_when_any = self.params['prob_reporting_when_any']
        prob_reporting_when_none = self.params['prob_reporting_when_none']
        
        if tot_latent_events == 0 and tot_sr > 0 :
            # Note: in this case, any Self-Report EMA cannot be matched to a latent smoking time
            # This case could happen if, for example, previous move might have been a 'death'
            # but participant initiated at least one self-report.
            # Assume that participant can lie/misremember when they Self-Report
            total_lik = prob_reporting_when_none**tot_sr
            total_loglik = np.log(total_lik)

        elif tot_latent_events > 0 and tot_sr == 0:
            # Note: in this case, latent smoking times exist but they were not reported in a Self Report EMA
            # This case could happen if, for example, previous move might have been a 'birth'
            # but there was no self-report observed.
            # Assume that participant does not lie when they Self-Report
            # However, participant may neglect to Self-Report a smoking incident
            # for example, due to burden
            total_lik = (1 - prob_reporting_when_any)**tot_latent_events
            total_loglik = np.log(total_lik)

        elif tot_latent_events > 0 and tot_sr > 0:    
            total_loglik = 0

            # Subcomponent due to delay ---------------------------------------
            self.observed_data['delay'] = self.observed_data['assessment_begin'] - self.observed_data['matched_latent_time']
            total_loglik += tot_sr * np.log(lambda_delay) - lambda_delay * np.nansum(self.observed_data['delay'])

            # Subcomponent due to recall --------------------------------------
            tot_ema = len(self.observed_data['assessment_order'])
            self.observed_data['prob_bk'] = np.repeat(np.nan, tot_ema)
            self.observed_data['log_prob_bk'] = np.repeat(np.nan, tot_ema)

            tot_sr_with_matched = 0
            for i in range(0, tot_ema):
                if self.observed_data['assessment_type'][i]=='selfreport':
                    current_lb = self.observed_data['assessment_begin_shifted'][i]
                    current_ub = self.observed_data['assessment_begin'][i] 
                    curr_matched_time = self.observed_data['matched_latent_time'][i]
                    # Check: Is current Self-Report EMA matched to any latent smoking time?
                    if np.isnan(curr_matched_time):
                        # Current Self-Report EMA is NOT matched to any latent smoking time
                        self.observed_data['prob_bk'][i] = prob_reporting_when_none
                        self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                    else:
                        # Current Self-Report EMA is matched to a latent smoking time
                        tot_sr_with_matched += 1  # update counter

                        # Calculate numerator of bk
                        windowtag = self.observed_data['windowtag'][i]

                        # Note: each value of windowtag corresponds to a response option in hours
                        # use_this_window_max will be based on time when prevous EMA was delivered
                        use_this_window_min = {1: 0/60, 2: 5/60, 3: 15/60, 4: 30/60}
                        use_this_window_max = {1: 5/60, 2: 15/60, 3: 30/60, 4: np.nan} 

                        # upper limit of integration
                        current_uk = self.observed_data['assessment_begin'][i] - use_this_window_min[windowtag]

                        if windowtag == 4:
                            if self.observed_data['assessment_begin_shifted'][i] > current_uk:
                                current_lk = self.observed_data['assessment_begin_shifted'][i] - 24 # subtract 24 hours
                            else:
                                current_lk = self.observed_data['assessment_begin_shifted'][i] 
                        else:
                            current_lk = self.observed_data['assessment_begin'][i] - use_this_window_max[windowtag]


                        # Calculate denominator of bk
                        if current_lk <= current_lb:
                            total_prob_constrained_lb = norm.cdf(x = current_lk, loc = curr_matched_time, scale = use_scale)
                        else:
                            total_prob_constrained_lb = norm.cdf(x = current_lb, loc = curr_matched_time, scale = use_scale)
                        
                        total_prob_constrained_ub = norm.cdf(x = current_ub, loc = curr_matched_time, scale = use_scale)
                        tot_prob_constrained = total_prob_constrained_ub - total_prob_constrained_lb

                        prob_constrained_lk = norm.cdf(x = current_lk, loc = curr_matched_time, scale = use_scale)
                        prob_constrained_uk = norm.cdf(x = current_uk, loc = curr_matched_time, scale = use_scale)

                        if (prob_constrained_uk - prob_constrained_lk) == tot_prob_constrained:
                            self.observed_data['prob_bk'][i] = (current_uk - current_lk)/(current_ub - current_lb)
                            self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                        else:
                            self.observed_data['prob_bk'][i] = (prob_constrained_uk - prob_constrained_lk)/tot_prob_constrained
                            self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])

            # We have already exited the for loop
            total_loglik += np.nansum(self.observed_data['log_prob_bk'])
            # Subcomponent due to propensity to self-report
            total_loglik += tot_sr_with_matched * np.log(prob_reporting_when_any) + (tot_latent_events - tot_sr_with_matched) * np.log(1-prob_reporting_when_any)

        else: #tot_latent_events == 0 and tot_sr == 0:
            total_lik = 1
            total_loglik = np.log(total_lik)

        return total_loglik 

# %%
class RandomEMA:

    def __init__(self, participant = None, day = None, latent_data = None, observed_data = None, params = None, index = None):
        self.participant = participant
        self.day = day
        self.latent_data = copy.deepcopy(latent_data)
        self.observed_data = copy.deepcopy(observed_data)
        self.params = copy.deepcopy(params)
        self.index = index

    def update_params(self, new_params):
        '''
        Update parameters
        '''
        self.params = copy.deepcopy(new_params)    

    def match(self):
        '''
        Matches each EMA with one latent smoking time occurring before the Random EMA
        After a latent smoking time is matched, it is removed
        '''

        # Inputs to be checked --------------------------------------------
        all_latent_times = self.latent_data['hours_since_start_day']
        tot_ema = len(self.observed_data['assessment_type'])

        if tot_ema > 0:
            self.observed_data['matched_latent_time'] = np.repeat(np.nan, tot_ema)
            remaining_latent_times = copy.deepcopy(all_latent_times)
            remaining_latent_times = np.sort(remaining_latent_times)
            for i in range(0, tot_ema):
                current_lb = self.observed_data['assessment_begin_shifted'][i]
                current_ub = self.observed_data['assessment_begin'][i]
                #current_assessment_type = self.observed_data['assessment_type'][i]
                which_within = (remaining_latent_times >= 0) & (remaining_latent_times < current_ub)
                if np.sum(which_within)>0:
                    which_idx = np.where(which_within)
                    matched_idx = np.max(which_idx)
                    matched_latent_time = remaining_latent_times[matched_idx]
                    self.observed_data['matched_latent_time'][i] = matched_latent_time
                    remaining_latent_times = np.delete(remaining_latent_times, matched_idx)
                    remaining_latent_times = np.sort(remaining_latent_times)
                else:
                    # This case can occur when between time 0 and time t there is no
                    # latent smoking time, but a self-report occurred between time 0 and time t
                    # This case may happen after a dumb death move
                    self.observed_data['matched_latent_time'][i] = np.nan
        else:
            self.observed_data['matched_latent_time'] = np.array([])

    def calc_loglik(self):
        '''
        Call the method calc_loglik after the method match has been called
        Calculate loglikelihood corresponding to Random EMA subcomponent
        '''

        use_scale = self.params['sd']
        prob_reporting_when_any = self.params['prob_reporting_when_any']
        prob_reporting_when_none = self.params['prob_reporting_when_none']

        all_latent_times = np.sort(self.latent_data['hours_since_start_day'])
        tot_latent_events = len(all_latent_times)
        tot_ema = len(self.observed_data['assessment_type'])

        if tot_ema == 0:
            tot_random_ema = 0
        else:
            tot_random_ema = np.sum(self.observed_data['assessment_type']=='random_ema')

        self.observed_data['prob_bk'] = np.repeat(np.nan, tot_ema)
        self.observed_data['log_prob_bk'] = np.repeat(np.nan, tot_ema)

        if tot_random_ema > 0:
        
            total_loglik = 0
            # Note: each value of windowtag corresponds to a response option in hours
            # use_this_window_max will be based on time when prevous EMA was delivered
            use_this_window_min = {1: 0/60, 2: 20/60, 3: 40/60, 4: 60/60, 5: 80/60, 6: 100/60}
            use_this_window_max = {1: 20/60, 2: 40/60, 3: 60/60, 4: 80/60, 5: 100/60, 6: np.nan}

            for i in range(0, tot_ema):
                if (self.observed_data['assessment_type'][i]=='random_ema') and (self.observed_data['smoke'][i]=='Yes'):
                    curr_matched_time = self.observed_data['matched_latent_time'][i]

                    if np.isnan(curr_matched_time):
                        self.observed_data['prob_bk'][i] = prob_reporting_when_none  # i.e., prob of reporting when no latent smoking time can be matched
                        self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                        total_loglik += self.observed_data['log_prob_bk'][i]
                    else:
                        current_lb = self.observed_data['assessment_begin_shifted'][i]
                        current_ub = self.observed_data['assessment_begin'][i] 
                        windowtag = self.observed_data['windowtag'][i]
                        # upper limit of integration
                        current_uk = self.observed_data['assessment_begin'][i] - use_this_window_min[windowtag]
                        # lower limit of integration
                        if windowtag == 6:
                            if self.observed_data['assessment_begin_shifted'][i] > current_uk:
                                current_lk = self.observed_data['assessment_begin_shifted'][i] - 24 # subtract 24 hours
                            else:
                                current_lk = self.observed_data['assessment_begin_shifted'][i] 
                        else:
                            current_lk = self.observed_data['assessment_begin'][i] - use_this_window_max[windowtag]


                        if (current_lk <= current_lb and current_uk <= current_lb):
                            # i.e., the upper bound and lower bound of the recalled smoking time both come before current_lb
                            # adding a point to this region should be a very unlikely occurrence
                            total_prob_constrained_lb = norm.cdf(x = current_lk, loc = curr_matched_time, scale = use_scale) # note that x = current_lk
                            total_prob_constrained_ub = norm.cdf(x = current_ub, loc = curr_matched_time, scale = use_scale)
                            tot_prob_constrained = total_prob_constrained_ub - total_prob_constrained_lb

                            prob_constrained_lk = norm.cdf(x = current_lk, loc = curr_matched_time, scale = use_scale)
                            prob_constrained_uk = norm.cdf(x = current_uk, loc = curr_matched_time, scale = use_scale)

                            if (prob_constrained_uk - prob_constrained_lk) == tot_prob_constrained:
                                self.observed_data['prob_bk'][i] = (current_uk - current_lk)/(current_ub - current_lb)
                                self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                                total_loglik += self.observed_data['log_prob_bk'][i]
                                total_loglik += np.log(prob_reporting_when_any)
                            else:
                                self.observed_data['prob_bk'][i] = (prob_constrained_uk - prob_constrained_lk)/tot_prob_constrained
                                self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                                total_loglik += self.observed_data['log_prob_bk'][i]
                                total_loglik += np.log(prob_reporting_when_any)

                        elif (current_lk <= current_lb and current_uk > current_lb):
                            # i.e., the lower bound of the recalled smoking time come before current_lb
                            # but the upper bound comes after current_lb
                            total_prob_constrained_lb = norm.cdf(x = current_lk, loc = curr_matched_time, scale = use_scale) # note that x = current_lk
                            total_prob_constrained_ub = norm.cdf(x = current_ub, loc = curr_matched_time, scale = use_scale)
                            tot_prob_constrained = total_prob_constrained_ub - total_prob_constrained_lb

                            prob_constrained_lk = norm.cdf(x = current_lk, loc = curr_matched_time, scale = use_scale)
                            prob_constrained_uk = norm.cdf(x = current_uk, loc = curr_matched_time, scale = use_scale)

                            if (prob_constrained_uk - prob_constrained_lk) == tot_prob_constrained:
                                self.observed_data['prob_bk'][i] = (current_uk - current_lk)/(current_ub - current_lb)
                                self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                                total_loglik += self.observed_data['log_prob_bk'][i]
                                total_loglik += np.log(prob_reporting_when_any)
                            else:
                                self.observed_data['prob_bk'][i] = (prob_constrained_uk - prob_constrained_lk)/tot_prob_constrained
                                self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                                total_loglik += self.observed_data['log_prob_bk'][i]
                                total_loglik += np.log(prob_reporting_when_any)

                        elif (current_lk >= current_lb and current_uk >= current_lb):
                            total_prob_constrained_lb = norm.cdf(x = current_lb, loc = curr_matched_time, scale = use_scale)
                            total_prob_constrained_ub = norm.cdf(x = current_ub, loc = curr_matched_time, scale = use_scale)
                            tot_prob_constrained = total_prob_constrained_ub - total_prob_constrained_lb

                            prob_constrained_lk = norm.cdf(x = current_lk, loc = curr_matched_time, scale = use_scale)
                            prob_constrained_uk = norm.cdf(x = current_uk, loc = curr_matched_time, scale = use_scale)

                            if (prob_constrained_uk - prob_constrained_lk) == tot_prob_constrained:
                                self.observed_data['prob_bk'][i] = (current_uk - current_lk)/(current_ub - current_lb)
                                self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                                total_loglik += self.observed_data['log_prob_bk'][i]
                                total_loglik += np.log(prob_reporting_when_any)
                            else:
                                self.observed_data['prob_bk'][i] = (prob_constrained_uk - prob_constrained_lk)/tot_prob_constrained
                                self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                                total_loglik += self.observed_data['log_prob_bk'][i]
                                total_loglik += np.log(prob_reporting_when_any)
                        
                        else:
                            total_loglik += np.nan # this case should not occur; sanity check on whether any cases were not accounted for


                elif (self.observed_data['assessment_type'][i]=='random_ema') and (self.observed_data['smoke'][i]=='No'):
                    curr_matched_time = self.observed_data['matched_latent_time'][i]
                    current_lb = self.observed_data['assessment_begin_shifted'][i]
                    current_ub = self.observed_data['assessment_begin'][i]

                    if np.isnan(curr_matched_time):
                        self.observed_data['prob_bk'][i] = 1-prob_reporting_when_none  # i.e., prob of NOT reporting when no latent smoking time can be matched
                        self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                        total_loglik += self.observed_data['log_prob_bk'][i]
                    else:
                        self.observed_data['prob_bk'][i] = 1-prob_reporting_when_any  # i.e., prob of NOT reporting when a latent smoking time can be matched
                        self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                        total_loglik += self.observed_data['log_prob_bk'][i]
                else:
                    # this is a case when we have a self-report EMA; do not adjust total_loglik
                    pass

        else:
            # This is the case when total number of Random EMA=0
            # Random EMA will not make a contribution to the overall loglikelihood
            total_loglik = 0

        return total_loglik


# %%
class DumbJitter():
    '''
    A collection of objects and methods related to dumb jitter move
    '''
    # needs to have objects as inputs
    def __init__(self):
        self.iter = 0
        self.accept = 0
    
    def initialize(self):
        self.sigma = 0.01
    
    def propose_new(self, latent_obj, eodsurvey_obj, selfreport_obj, randomema_obj):
        # Note that latent_obj, eodsurvey_obj, selfreport_obj, and randomema_obj are at the participant-day level
        # Note: this function assumes that latent_obj will have at least 1 latent event
        current_latent_times = latent_obj.latent_data['hours_since_start_day']
        tot_latent_events = len(current_latent_times)
        # Calculate pi_x
        selfreport_obj.match() # this line
        randomema_obj.match() # and this line should yield the same output
        total_loglik_current = latent_obj.calc_loglik() + eodsurvey_obj.calc_loglik() + selfreport_obj.calc_loglik() + randomema_obj.calc_loglik()
        pi_x = np.exp(total_loglik_current)

        # Dumb jitter
        log_current_times = np.log(current_latent_times)
        proposed_log_current_times = log_current_times + np.random.normal(scale = self.sigma, size=tot_latent_events)

        # Calculate pi_xprime
        proposed_latent_times = np.exp(proposed_log_current_times)
        proposed_latent_obj = copy.deepcopy(latent_obj)
        proposed_latent_obj.latent_data['hours_since_start_day'] = proposed_latent_times
        total_loglik_proposed = proposed_latent_obj.calc_loglik() + eodsurvey_obj.calc_loglik() + selfreport_obj.calc_loglik() + randomema_obj.calc_loglik()
        pi_xprime = np.exp(total_loglik_proposed)

        # Should we accept the proposal?
        mh_ratio = pi_xprime/pi_x
        acceptance_prob = np.min([1.0, mh_ratio])
        decision = np.random.binomial(n=1, p=acceptance_prob, size=1)
        decision = decision[0]

        # Keep tabs
        self.iter += 1
        self.accept += decision

        if decision==1:
            output = proposed_latent_times
        else:
            output = current_latent_times
        
        return output



# %% 
class SmartJitter():
    '''
    A collection of objects and methods related to smart jitter move
    '''
    def __init__(self, dim):
        self.dimension = dim
        self.iter = 0
        self.accept = 0    

    def initialize(self):
        self.mu_current = 0.01 * np.ones(self.dimension)
        self.sigma_current = 0.01 * np.identity(self.dimension)
        self.mu_next = None
        self.sigma_next = None

    def propose_new(self, latent_obj, eodsurvey_obj, selfreport_obj, randomema_obj):
        # Note that latent_obj, eodsurvey_obj, selfreport_obj, and randomema_obj are at the participant-day level
        # Note: this function assumes that latent_obj will have at least 1 latent event
        
        current_latent_times = latent_obj.latent_data['hours_since_start_day']
        # Calculate pi_x
        selfreport_obj.match() # this line
        randomema_obj.match() # and this line should yield the same output
        total_loglik_current = latent_obj.calc_loglik() + eodsurvey_obj.calc_loglik() + selfreport_obj.calc_loglik() + randomema_obj.calc_loglik()
        pi_x = np.exp(total_loglik_current)

        # Propose new latent smoking times        
        log_current_times = np.log(current_latent_times)
        proposed_log_current_times = log_current_times + np.random.multivariate_normal(mean = np.repeat(0, self.dimension), cov = self.sigma_current)

        # Calculate pi_xprime
        proposed_latent_times = np.exp(proposed_log_current_times)
        proposed_latent_obj = copy.deepcopy(latent_obj)
        proposed_latent_obj.latent_data['hours_since_start_day'] = proposed_latent_times
        total_loglik_proposed = proposed_latent_obj.calc_loglik() + eodsurvey_obj.calc_loglik() + selfreport_obj.calc_loglik() + randomema_obj.calc_loglik()
        pi_xprime = np.exp(total_loglik_proposed)

        # Should we accept the proposal?
        mh_ratio = pi_xprime/pi_x
        acceptance_prob = np.min([1.0, mh_ratio])
        decision = np.random.binomial(n=1, p=acceptance_prob, size=1)
        decision = decision[0]

        # Keep tabs
        self.iter += 1
        self.accept += decision

        if decision==1:
            output = proposed_latent_times
        else:
            output = current_latent_times

        if self.iter>1:
            # Update parameters; uses Haario et al
            delta = proposed_latent_times - self.mu_current
            self.mu_next = self.mu_current + (1/(self.iter - 1)) * delta
            self.sigma_next = self.sigma_current + (1/(self.iter - 1)) * (np.outer(delta, delta) - self.sigma_current)
        else:
            self.mu_next = self.mu_current
            self.sigma_next = self.sigma_current
        
        # Prepare for next iteration
        self.mu_current = self.mu_next
        self.sigma_current = self.sigma_next

        return output
        
# %%
# EXAMPLE
use_this_participant = None
use_this_day = None

curr_latent_params = {'lambda_prequit':1, 'lambda_postquit':1}
curr_selfreport_params = {'prob_reporting_when_any': 0.90, 'prob_reporting_when_none': 0.01, 'lambda_delay': 0.5, 'sd': 30/60}
curr_randomema_params = {'prob_reporting_when_any': 0.90, 'prob_reporting_when_none': 0.01, 'sd': 30/60}
curr_eodsurvey_params = {'recall_epsilon':3, 'sd': 60/60, 'rho':0.8, 'budget':10}

latent_obj = Latent(participant = use_this_participant,
                    day = use_this_day,
                    latent_data = init_dict_latent_data[use_this_participant][use_this_day],
                    params = copy.deepcopy(curr_latent_params))

eodsurvey_obj = EODSurvey(participant = use_this_participant, 
                            day = use_this_day, 
                            latent_data = init_dict_latent_data[use_this_participant][use_this_day],
                            observed_data = init_dict_observed_eod_survey[use_this_participant][use_this_day],
                            params = copy.deepcopy(curr_eodsurvey_params))

selfreport_obj = SelfReport(participant = use_this_participant, 
                            day = use_this_day, 
                            latent_data = init_dict_latent_data[use_this_participant][use_this_day],
                            observed_data = init_dict_observed_ema[use_this_participant][use_this_day],
                            params = copy.deepcopy(curr_selfreport_params))

randomema_obj = RandomEMA(participant = use_this_participant, 
                            day = use_this_day, 
                            latent_data = init_dict_latent_data[use_this_participant][use_this_day],
                            observed_data = init_dict_observed_ema[use_this_participant][use_this_day],  
                            params = copy.deepcopy(curr_randomema_params))

# %%
dumbjitter_obj = DumbJitter()
dumbjitter_obj.initialize()
dumbjitter_obj.propose_new(latent_obj = latent_obj, 
                           eodsurvey_obj = eodsurvey_obj, 
                           selfreport_obj = selfreport_obj,
                           randomema_obj = randomema_obj)


# %%
smartjitter_obj = SmartJitter(dim=2) # note the need to set appropriate dimension depending on the participant day being utilized
smartjitter_obj.initialize()
smartjitter_obj.propose_new(latent_obj = latent_obj, 
                            eodsurvey_obj = eodsurvey_obj, 
                            selfreport_obj = selfreport_obj,
                            randomema_obj = randomema_obj)

# %%



