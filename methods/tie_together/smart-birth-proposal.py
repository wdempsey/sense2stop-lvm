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
# Helper functions

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
    A collection of objects and methods related to latent process subcomponent
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
        tot_latent_events = len(all_latent_times)
        tot_ema = len(self.observed_data['assessment_type'])

        if tot_ema > 0:
            self.observed_data['matched_latent_time'] = np.repeat(np.nan, tot_ema)
            remaining_latent_times = copy.deepcopy(all_latent_times)
            remaining_latent_times = np.sort(remaining_latent_times)
            for i in range(0, tot_ema):
                current_lb = self.observed_data['assessment_begin_shifted'][i]
                current_ub = self.observed_data['assessment_begin'][i]
                current_assessment_type = self.observed_data['assessment_type'][i]
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
        Matches each EMA with one latent smoking time occurring before the Self Report EMA
        After a latent smoking time is matched, it is removed
        '''

        # Inputs to be checked --------------------------------------------
        all_latent_times = self.latent_data['hours_since_start_day']
        tot_latent_events = len(all_latent_times)
        tot_ema = len(self.observed_data['assessment_type'])

        if tot_ema > 0:
            self.observed_data['matched_latent_time'] = np.repeat(np.nan, tot_ema)
            remaining_latent_times = copy.deepcopy(all_latent_times)
            remaining_latent_times = np.sort(remaining_latent_times)
            for i in range(0, tot_ema):
                current_lb = self.observed_data['assessment_begin_shifted'][i]
                current_ub = self.observed_data['assessment_begin'][i]
                current_assessment_type = self.observed_data['assessment_type'][i]
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
def get_for_all_current_state_lik(all_participant_ids, 
                                  all_days, 
                                  curr_dict_latent_data,
                                  curr_dict_observed_ema,
                                  curr_dict_observed_eod_survey,
                                  curr_latent_params,
                                  curr_selfreport_params,
                                  curr_randomema_params,
                                  curr_eodsurvey_params):

    # Calculate likelihood for current configuration of points (prior to any proposal)
    dict_current_state = {}
    for this_participant in all_participant_ids: 
        current_dict = {}
        for this_day in all_days: 
            # Initialize Latent object
            latent_obj = Latent(participant = this_participant,
                                day = this_day,
                                latent_data = curr_dict_latent_data[this_participant][this_day],
                                params = copy.deepcopy(curr_latent_params))

            # Initialize EODSurvey object
            eodsurvey_obj = EODSurvey(participant = this_participant, 
                                      day = this_day, 
                                      latent_data = curr_dict_latent_data[this_participant][this_day],
                                      observed_data = curr_dict_observed_eod_survey[this_participant][this_day],
                                      params = copy.deepcopy(curr_eodsurvey_params))

            # Initialize SelfReport object
            selfreport_obj = SelfReport(participant = this_participant, 
                                        day = this_day, 
                                        latent_data = curr_dict_latent_data[this_participant][this_day],
                                        observed_data = curr_dict_observed_ema[this_participant][this_day],
                                        params = copy.deepcopy(curr_selfreport_params))

            # Initialize RandomEMA object
            randomema_obj = RandomEMA(participant = this_participant, 
                                      day = this_day, 
                                      latent_data = curr_dict_latent_data[this_participant][this_day],
                                      observed_data = curr_dict_observed_ema[this_participant][this_day],  
                                      params = copy.deepcopy(curr_randomema_params))

            # Calculate likelihood
            selfreport_obj.match()
            randomema_obj.match()
            total_loglik = latent_obj.calc_loglik() + eodsurvey_obj.calc_loglik() + selfreport_obj.calc_loglik() + randomema_obj.calc_loglik()
            total_lik = np.exp(total_loglik)
            current_dict.update({this_day:{'x':curr_dict_latent_data[this_participant][this_day]['hours_since_start_day'],
                                           'pi_x':total_lik}})
        dict_current_state.update({this_participant:current_dict})
    
    return dict_current_state


# %%
# Helper functions for birth or death proposal

def get_this_loglik(x):
    loglik = x.calc_loglik()
    return(x.index, loglik)


# %%
# Helper functions for birth proposal
def construct_grid(increment, day_length):
    
    # Construct grid of points to consider for a smart birth

    if day_length <= increment:
        init_grid = np.array([0, day_length])
    else:
        init_grid = np.arange(0, day_length, increment)

    return init_grid


def get_sets_along_grid_birth(init_grid, current_latent_data):

    # What are the various configurations of points to consider in a smart/dumb birth proposal?
    # We will consider birthing a new point from init_grid that does not yet exist in current_latent_data

    grid = np.setdiff1d(ar1 = init_grid, ar2 = current_latent_data)
    grid = np.sort(grid)

    M = len(grid)

    sets_along_grid = {}
    for idx_grid in range(0,M):
        new_latent_data = np.append(current_latent_data, grid[idx_grid])
        new_latent_data = np.sort(new_latent_data)
        sets_along_grid.update({idx_grid:new_latent_data})

    return grid, sets_along_grid


def parallelize_class_method(list_objects, num_processes = 8):
    '''
    list_objects is a list containing instances of classes
    '''

    with Pool(processes = num_processes) as p:
        my_output = p.map(get_this_loglik, list_objects)
        return my_output


def grid_likelihood_latent_birth(current_participant, current_day, latent_params, dict_latent_data):

    '''
    Calculate the likelihood at each point of a grid
    Note that smart birth and smart death differ in the grids they consider
    '''
    # Initialize Latent object
    init_latent_obj = Latent(participant = current_participant,
                             day = current_day,
                             latent_data = dict_latent_data[current_participant][current_day],
                             params = copy.deepcopy(latent_params))

    # Construct grid for smart birth
    latent_grid = construct_grid(increment = 1/60, day_length = init_latent_obj.latent_data['day_length'])
    latent_grid, latent_grid_sets = get_sets_along_grid_birth(init_grid = latent_grid, current_latent_data = init_latent_obj.latent_data['hours_since_start_day'])

    # Work with Latent class objects
    latent_total_grid_sets = len(latent_grid_sets)

    # Each element of the list is an instance of the Latent class
    latent_my_list = []
    for idx_set in range(0, latent_total_grid_sets):
        candidate_latent_data = copy.deepcopy(init_latent_obj.latent_data)
        candidate_latent_data['hours_since_start_day'] = latent_grid_sets[idx_set]
        latent_my_list.append(Latent(participant = current_participant,
                                     day = current_day,
                                     latent_data = candidate_latent_data,
                                     params = copy.deepcopy(latent_params),
                                     index = idx_set))

    element_wise_loglik = []
    for idx_set in range(0, latent_total_grid_sets):
        res = latent_my_list[idx_set].calc_loglik()
        element_wise_loglik.append(res)

    element_wise_lik = np.exp(element_wise_loglik)
    f = interpolate.interp1d(x = latent_grid, y = element_wise_lik, fill_value="extrapolate")

    return f 


def grid_likelihood_eodsurvey_birth(current_participant, current_day, latent_params, eodsurvey_params, dict_latent_data, dict_observed_eod_survey):

    '''
    Calculate the likelihood at each point of a grid
    '''
    # Initialize EODSurvey object
    init_eodsurvey_obj = EODSurvey(participant = current_participant, 
                                    day = current_day, 
                                    latent_data = dict_latent_data[current_participant][current_day],
                                    observed_data = dict_observed_eod_survey[current_participant][current_day],
                                    params = copy.deepcopy(eodsurvey_params))

    # Construct grid for smart birth
    eodsurvey_grid = construct_grid(increment = 30/60, day_length = init_eodsurvey_obj.latent_data['day_length'])
    eodsurvey_grid, eodsurvey_grid_sets = get_sets_along_grid_birth(init_grid = eodsurvey_grid, current_latent_data = init_eodsurvey_obj.latent_data['hours_since_start_day'])

    # Work with EODSurvey class objects
    eodsurvey_total_grid_sets = len(eodsurvey_grid_sets)

    # Each element of the list is an instance of the EODSurvey class
    eodsurvey_my_list = []
    for idx_set in range(0, eodsurvey_total_grid_sets):
        candidate_latent_data = copy.deepcopy(init_eodsurvey_obj.latent_data)
        candidate_latent_data['hours_since_start_day'] = eodsurvey_grid_sets[idx_set]
        eodsurvey_my_list.append(EODSurvey(participant = current_participant, 
                                            day = current_day, 
                                            latent_data = candidate_latent_data,
                                            observed_data = dict_observed_eod_survey[current_participant][current_day],
                                            params = copy.deepcopy(eodsurvey_params),
                                            index = idx_set))
    
    # No need to parallelize calculations when current number of latent smoking times is less than 6
    if len(init_eodsurvey_obj.latent_data['hours_since_start_day']) < 6:
        eodsurvey_grid_loglik = []
        for idx_set in range(0, eodsurvey_total_grid_sets):
            res = eodsurvey_my_list[idx_set].calc_loglik()
            eodsurvey_grid_loglik.append(res)

    else:
        eodsurvey_my_output = parallelize_class_method(list_objects = eodsurvey_my_list)
        eodsurvey_my_output = sorted(eodsurvey_my_output, key=lambda tup: tup[0], reverse=False)
        # Get calculated loglik
        eodsurvey_grid_loglik = []
        for a_tuple in eodsurvey_my_output:
            eodsurvey_grid_loglik.append(a_tuple[1])
    
    eodsurvey_grid_lik = np.exp(eodsurvey_grid_loglik)
    # Perform interpolation of eodsurvey at the minute-level
    # Note: interpolate likelihood instead of loglikelihood to avoid having to interpolate over -inf values. This will produce an error.
    f = interpolate.interp1d(x = eodsurvey_grid, y = eodsurvey_grid_lik, fill_value="extrapolate")
        
    return f 


def grid_likelihood_selfreport_birth(current_participant, current_day, latent_params, selfreport_params, dict_latent_data, dict_observed_ema):

    '''
    Calculate the likelihood at each point of a grid
    '''
    # Initialize SelfReport object
    init_selfreport_obj = SelfReport(participant = current_participant, 
                                     day = current_day, 
                                     latent_data = dict_latent_data[current_participant][current_day],
                                     observed_data = dict_observed_ema[current_participant][current_day],
                                     params = copy.deepcopy(selfreport_params))

    # Construct grid for smart birth
    selfreport_grid = construct_grid(increment = 1/60, day_length = init_selfreport_obj.latent_data['day_length'])
    selfreport_grid, selfreport_grid_sets = get_sets_along_grid_birth(init_grid = selfreport_grid, current_latent_data = init_selfreport_obj.latent_data['hours_since_start_day'])

    # Work with selfreport class objects
    selfreport_total_grid_sets = len(selfreport_grid_sets)

    # Each element of the list is an instance of the selfreport class
    selfreport_my_list = []
    for idx_set in range(0, selfreport_total_grid_sets):
        candidate_latent_data = copy.deepcopy(init_selfreport_obj.latent_data)
        candidate_latent_data['hours_since_start_day'] = selfreport_grid_sets[idx_set]
        selfreport_my_list.append(SelfReport(participant = current_participant, 
                                             day = current_day, 
                                             latent_data = candidate_latent_data,
                                             observed_data = dict_observed_ema[current_participant][current_day],
                                             params = copy.deepcopy(selfreport_params),
                                             index = idx_set))
    
    element_wise_loglik = []
    for idx_set in range(0, selfreport_total_grid_sets):
        selfreport_my_list[idx_set].match()
        res = selfreport_my_list[idx_set].calc_loglik()
        element_wise_loglik.append(res)

    element_wise_lik = np.exp(element_wise_loglik)

    f = interpolate.interp1d(x = selfreport_grid, y = element_wise_lik, fill_value="extrapolate")

    return f


def grid_likelihood_randomema_birth(current_participant, current_day, latent_params, randomema_params, dict_latent_data, dict_observed_ema):

    '''
    Calculate the likelihood at each point of a grid
    '''
    # Initialize RandomEMA object
    init_randomema_obj = RandomEMA(participant = current_participant, 
                                   day = current_day, 
                                   latent_data = dict_latent_data[current_participant][current_day],
                                   observed_data = dict_observed_ema[current_participant][current_day],
                                   params = copy.deepcopy(randomema_params))

    # Construct grid for smart birth
    randomema_grid = construct_grid(increment = 1/60, day_length = init_randomema_obj.latent_data['day_length'])
    randomema_grid, randomema_grid_sets = get_sets_along_grid_birth(init_grid = randomema_grid, current_latent_data = init_randomema_obj.latent_data['hours_since_start_day'])

    # Work with randomema class objects
    randomema_total_grid_sets = len(randomema_grid_sets)

    # Each element of the list is an instance of the randomema class
    randomema_my_list = []
    for idx_set in range(0, randomema_total_grid_sets):
        candidate_latent_data = copy.deepcopy(init_randomema_obj.latent_data)
        candidate_latent_data['hours_since_start_day'] = randomema_grid_sets[idx_set]
        randomema_my_list.append(RandomEMA(participant = current_participant, 
                                           day = current_day, 
                                           latent_data = candidate_latent_data,
                                           observed_data = dict_observed_ema[current_participant][current_day],
                                           params = copy.deepcopy(randomema_params),
                                           index = idx_set))
    
    element_wise_loglik = []
    for idx_set in range(0, randomema_total_grid_sets):
        randomema_my_list[idx_set].match()
        res = randomema_my_list[idx_set].calc_loglik()
        element_wise_loglik.append(res)

    element_wise_lik = np.exp(element_wise_loglik)

    f = interpolate.interp1d(x = randomema_grid, y = element_wise_lik, fill_value="extrapolate")

    return f 




# %%
def get_for_all_smart_birth_lik(use_increment,
                                all_participant_ids, 
                                all_days, 
                                curr_dict_latent_data,
                                curr_dict_observed_ema,
                                curr_dict_observed_eod_survey,
                                curr_latent_params,
                                curr_selfreport_params,
                                curr_randomema_params,
                                curr_eodsurvey_params):

    # Latent model: Likelihood corresponding to each point on the grid
    dict_latent_likelihood = {}
    for this_participant in all_participant_ids: 
        current_dict = {}
        for this_day in all_days: 
            interp_func = grid_likelihood_latent_birth(current_participant = this_participant, 
                                                       current_day = this_day, 
                                                       latent_params = curr_latent_params, 
                                                       dict_latent_data = curr_dict_latent_data)
            use_this_grid = construct_grid(increment = use_increment, day_length = curr_dict_latent_data[this_participant][this_day]['day_length'])
            smoothed_lik = interp_func(use_this_grid)
            # Note: if coarse grid ends at t* and the likelihood at t* is very close to zero, 
            # e.g., 1e-13, then a point on a fine grid, say at t* + 10 minutes
            # might have a negative interpolated value, say -1e-10
            # when this happens, we set the interpolated value to zero
            smoothed_lik[(smoothed_lik < 0)] = 0
            current_dict.update({this_day:smoothed_lik})
        dict_latent_likelihood.update({this_participant:current_dict})


    # MEM -- end of day survey subcomponent: Likelihood corresponding to each point on the grid
    dict_mem_eodsurvey_likelihood = {}
    for this_participant in all_participant_ids:
        current_dict = {}
        for this_day in all_days:  
            interp_func = grid_likelihood_eodsurvey_birth(current_participant = this_participant, 
                                                          current_day = this_day, 
                                                          latent_params = curr_latent_params, 
                                                          eodsurvey_params = curr_eodsurvey_params, 
                                                          dict_latent_data = curr_dict_latent_data, 
                                                          dict_observed_eod_survey = curr_dict_observed_eod_survey)
            
            use_this_grid = construct_grid(increment = use_increment, day_length = curr_dict_latent_data[this_participant][this_day]['day_length'])
            smoothed_lik = interp_func(use_this_grid)
            # Note: if coarse grid ends at t* and the likelihood at t* is very close to zero, 
            # e.g., 1e-13, then a point on a fine grid, say at t* + 10 minutes
            # might have a negative interpolated value, say -1e-10
            # when this happens, we set the interpolated value to zero
            smoothed_lik[(smoothed_lik < 0)] = 0
            current_dict.update({this_day:smoothed_lik})
        dict_mem_eodsurvey_likelihood.update({this_participant:current_dict})


    # MEM -- selfreport subcomponent: Likelihood corresponding to each point on the grid
    dict_mem_selfreport_likelihood = {}
    for this_participant in all_participant_ids:
        current_dict = {}
        for this_day in all_days: 
            interp_func = grid_likelihood_selfreport_birth(current_participant = this_participant, 
                                                           current_day = this_day, 
                                                           latent_params = curr_latent_params, 
                                                           selfreport_params = curr_selfreport_params, 
                                                           dict_latent_data = curr_dict_latent_data, 
                                                           dict_observed_ema = curr_dict_observed_ema)
            
            use_this_grid = construct_grid(increment = use_increment, day_length = curr_dict_latent_data[this_participant][this_day]['day_length'])
            smoothed_lik = interp_func(use_this_grid)
             # Note: if coarse grid ends at t* and the likelihood at t* is very close to zero, 
            # e.g., 1e-13, then a point on a fine grid, say at t* + 10 minutes
            # might have a negative interpolated value, say -1e-10
            # when this happens, we set the interpolated value to zero
            smoothed_lik[(smoothed_lik < 0)] = 0
            current_dict.update({this_day:smoothed_lik})
        dict_mem_selfreport_likelihood.update({this_participant:current_dict})


    # MEM -- Random EMA subcomponent: Likelihood corresponding to each point on the grid
    dict_mem_randomema_likelihood = {}
    for this_participant in all_participant_ids:
        current_dict = {}
        for this_day in all_days:  # all_days here
            interp_func = grid_likelihood_randomema_birth(current_participant = this_participant, 
                                                          current_day = this_day, 
                                                          latent_params = curr_latent_params, 
                                                          randomema_params = curr_randomema_params, 
                                                          dict_latent_data = curr_dict_latent_data, 
                                                          dict_observed_ema = curr_dict_observed_ema)
            
            use_this_grid = construct_grid(increment = use_increment, day_length = curr_dict_latent_data[this_participant][this_day]['day_length'])
            smoothed_lik = interp_func(use_this_grid)
            # Note: if coarse grid ends at t* and the likelihood at t* is very close to zero, 
            # e.g., 1e-13, then a point on a fine grid, say at t* + 10 minutes
            # might have a negative interpolated value, say -1e-10
            # when this happens, we set the interpolated value to zero
            smoothed_lik[(smoothed_lik < 0)] = 0
            current_dict.update({this_day:smoothed_lik})
        dict_mem_randomema_likelihood.update({this_participant:current_dict})
    
    dict_all = {'latent':dict_latent_likelihood, 
                'eodsurvey':dict_mem_eodsurvey_likelihood, 
                'selfreport':dict_mem_selfreport_likelihood, 
                'randomema':dict_mem_randomema_likelihood,
                'all_participant_ids':all_participant_ids,
                'all_days':all_days,
                'curr_dict_latent_data':curr_dict_latent_data,
                'use_increment':use_increment}

    return dict_all


# %%

def get_for_all_smart_birth_pdf(dict_smart_birth_lik, dict_current_state):
    
    dict_latent_likelihood = copy.deepcopy(dict_smart_birth_lik['latent'])
    dict_mem_eodsurvey_likelihood = copy.deepcopy(dict_smart_birth_lik['eodsurvey'])
    dict_mem_selfreport_likelihood = copy.deepcopy(dict_smart_birth_lik['selfreport'])
    dict_mem_randomema_likelihood = copy.deepcopy(dict_smart_birth_lik['randomema'])

    all_participant_ids = dict_smart_birth_lik['all_participant_ids']
    all_days = dict_smart_birth_lik['all_days']
    curr_dict_latent_data = dict_smart_birth_lik['curr_dict_latent_data']
    use_increment = dict_smart_birth_lik['use_increment']

    dict_smart_birth_pdf = {}

    for current_participant in all_participant_ids: 
        current_dict_smart_birth_pdf = {}
        for current_day in all_days:  
            # Calculate smart birth pdf
            lik_latent = dict_latent_likelihood[current_participant][current_day]
            lik_eodsurvey = dict_mem_eodsurvey_likelihood[current_participant][current_day]
            lik_selfreport = dict_mem_selfreport_likelihood[current_participant][current_day]
            lik_randomema = dict_mem_randomema_likelihood[current_participant][current_day]

            current_element_wise_lik = lik_latent * lik_eodsurvey * lik_selfreport * lik_randomema
            current_denominator_pdf_smart_birth = np.sum(current_element_wise_lik)
            current_pdf_smart_birth = current_element_wise_lik/current_denominator_pdf_smart_birth

            # Update dictionary for this day
            use_this_grid = construct_grid(increment = use_increment, day_length = curr_dict_latent_data[current_participant][current_day]['day_length'])
            current_grid, sets_along_current_grid = get_sets_along_grid_birth(use_this_grid, dict_current_state[current_participant][current_day]['x'])

            current_dict_smart_birth_pdf.update({current_day:{'grid':current_grid, 
                                                              'proposed_latent_smoking_times':sets_along_current_grid, 
                                                              'pdf_smart_birth':current_pdf_smart_birth,
                                                              'lik_smart_birth':current_element_wise_lik}})
        # Update dictionary for this person
        dict_smart_birth_pdf.update({current_participant:current_dict_smart_birth_pdf})

    dict_all = {'dict_smart_birth_pdf':dict_smart_birth_pdf,
                'all_participant_ids':all_participant_ids,
                'all_days':all_days}

    return dict_all


# %%
# Helper functions for death proposal

def get_sets_along_grid_death(current_latent_data):

    # What are the various configurations of points to consider in a smart/dumb death proposal?
    M = len(current_latent_data)

    if M == 0:
        sets_along_grid = np.array([])
        grid = np.array([])
    else:
        current_latent_data = np.sort(current_latent_data)
        grid = current_latent_data
        sets_along_grid = {}
        for idx_grid in range(0,M):
            new_latent_data = np.delete(current_latent_data, idx_grid)
            new_latent_data = np.sort(new_latent_data)
            sets_along_grid.update({idx_grid:new_latent_data})

    return grid, sets_along_grid


# %%
def grid_likelihood_latent_death(current_participant, current_day, latent_params, dict_latent_data):

    '''
    Calculate the likelihood at each point of a grid
    Note that smart birth and smart death differ in the grids they consider
    '''


    # Initialize Latent object
    init_latent_obj = Latent(participant = current_participant,
                            day = current_day,
                            latent_data = dict_latent_data[current_participant][current_day],
                            params = copy.deepcopy(latent_params))

    # Construct grid for smart death
    current_grid, latent_grid_sets = get_sets_along_grid_death(current_latent_data = init_latent_obj.latent_data['hours_since_start_day'])

    # Work with Latent class objects
    latent_total_grid_sets = len(latent_grid_sets)

    # Each element of the list is an instance of the Latent class
    latent_my_list = []
    for idx_set in range(0, latent_total_grid_sets):
        candidate_latent_data = copy.deepcopy(init_latent_obj.latent_data)
        candidate_latent_data['hours_since_start_day'] = latent_grid_sets[idx_set]
        latent_my_list.append(Latent(participant = current_participant,
                                    day = current_day,
                                    latent_data = candidate_latent_data,
                                    params = copy.deepcopy(latent_params),
                                    index = idx_set))

    element_wise_loglik = []
    for idx_set in range(0, latent_total_grid_sets):
        res = latent_my_list[idx_set].calc_loglik()
        element_wise_loglik.append(res)

    element_wise_lik = np.exp(element_wise_loglik)

    return element_wise_lik


# %%
def grid_likelihood_eodsurvey_death(current_participant, current_day, latent_params, eodsurvey_params, dict_latent_data, dict_observed_eod_survey):

    '''
    Calculate the likelihood at each point of a grid
    '''

    # Initialize EODSurvey object
    init_eodsurvey_obj = EODSurvey(participant = current_participant, 
                                    day = current_day, 
                                    latent_data = dict_latent_data[current_participant][current_day],
                                    observed_data = dict_observed_eod_survey[current_participant][current_day],
                                    params = copy.deepcopy(eodsurvey_params))


    # Construct grid for smart death
    current_grid, eodsurvey_grid_sets = get_sets_along_grid_death(current_latent_data = init_eodsurvey_obj.latent_data['hours_since_start_day'])

    # Work with EODSurvey class objects
    eodsurvey_total_grid_sets = len(eodsurvey_grid_sets)

    # Each element of the list is an instance of the EODSurvey class
    eodsurvey_my_list = []
    for idx_set in range(0, eodsurvey_total_grid_sets):
        candidate_latent_data = copy.deepcopy(init_eodsurvey_obj.latent_data)
        candidate_latent_data['hours_since_start_day'] = eodsurvey_grid_sets[idx_set]
        eodsurvey_my_list.append(EODSurvey(participant = current_participant, 
                                            day = current_day, 
                                            latent_data = candidate_latent_data,
                                            observed_data = dict_observed_eod_survey[current_participant][current_day],
                                            params = copy.deepcopy(eodsurvey_params),
                                            index = idx_set))
    
    # No need to parallelize calculations when current number of latent smoking times is less than 6
    if len(init_eodsurvey_obj.latent_data['hours_since_start_day']) < 6:
        eodsurvey_grid_loglik = []
        for idx_set in range(0, eodsurvey_total_grid_sets):
            res = eodsurvey_my_list[idx_set].calc_loglik()
            eodsurvey_grid_loglik.append(res)

    else:
        eodsurvey_my_output = parallelize_class_method(list_objects = eodsurvey_my_list)
        eodsurvey_my_output = sorted(eodsurvey_my_output, key=lambda tup: tup[0], reverse=False)
        # Get calculated loglik
        eodsurvey_grid_loglik = []
        for a_tuple in eodsurvey_my_output:
            eodsurvey_grid_loglik.append(a_tuple[1])
    
    eodsurvey_grid_lik = np.exp(eodsurvey_grid_loglik)
        
    return eodsurvey_grid_lik 


def grid_likelihood_selfreport_death(current_participant, current_day, latent_params, selfreport_params, dict_latent_data, dict_observed_ema):

    '''
    Calculate the likelihood at each point of a grid
    '''


    # Initialize SelfReport object
    init_selfreport_obj = SelfReport(participant = current_participant, 
                                    day = current_day, 
                                    latent_data = dict_latent_data[current_participant][current_day],
                                    observed_data = dict_observed_ema[current_participant][current_day],
                                    params = copy.deepcopy(selfreport_params))

    # Construct grid for smart death
    current_grid, selfreport_grid_sets = get_sets_along_grid_death(current_latent_data = init_selfreport_obj.latent_data['hours_since_start_day'])

    # Work with selfreport class objects
    selfreport_total_grid_sets = len(selfreport_grid_sets)

    # Each element of the list is an instance of the selfreport class
    selfreport_my_list = []
    for idx_set in range(0, selfreport_total_grid_sets):
        candidate_latent_data = copy.deepcopy(init_selfreport_obj.latent_data)
        candidate_latent_data['hours_since_start_day'] = selfreport_grid_sets[idx_set]
        selfreport_my_list.append(SelfReport(participant = current_participant, 
                                            day = current_day, 
                                            latent_data = candidate_latent_data,
                                            observed_data = dict_observed_ema[current_participant][current_day],
                                            params = copy.deepcopy(selfreport_params),
                                            index = idx_set))
    
    element_wise_loglik = []
    for idx_set in range(0, selfreport_total_grid_sets):
        selfreport_my_list[idx_set].match()
        res = selfreport_my_list[idx_set].calc_loglik()
        element_wise_loglik.append(res)

    element_wise_lik = np.exp(element_wise_loglik)

    return element_wise_lik


def grid_likelihood_randomema_death(current_participant, current_day, latent_params, randomema_params, dict_latent_data, dict_observed_ema):

    '''
    Calculate the likelihood at each point of a grid
    '''


    # Initialize RandomEMA object
    init_randomema_obj = RandomEMA(participant = current_participant, 
                                day = current_day, 
                                latent_data = dict_latent_data[current_participant][current_day],
                                observed_data = dict_observed_ema[current_participant][current_day],
                                params = copy.deepcopy(randomema_params))

    # Construct grid for smart death
    current_grid, randomema_grid_sets = get_sets_along_grid_death(current_latent_data = init_randomema_obj.latent_data['hours_since_start_day'])

    # Work with randomema class objects
    randomema_total_grid_sets = len(randomema_grid_sets)

    # Each element of the list is an instance of the randomema class
    randomema_my_list = []
    for idx_set in range(0, randomema_total_grid_sets):
        candidate_latent_data = copy.deepcopy(init_randomema_obj.latent_data)
        candidate_latent_data['hours_since_start_day'] = randomema_grid_sets[idx_set]
        randomema_my_list.append(RandomEMA(participant = current_participant, 
                                        day = current_day, 
                                        latent_data = candidate_latent_data,
                                        observed_data = dict_observed_ema[current_participant][current_day],
                                        params = copy.deepcopy(randomema_params),
                                        index = idx_set))
    
    element_wise_loglik = []
    for idx_set in range(0, randomema_total_grid_sets):
        randomema_my_list[idx_set].match()
        res = randomema_my_list[idx_set].calc_loglik()
        element_wise_loglik.append(res)

    element_wise_lik = np.exp(element_wise_loglik)

    return element_wise_lik



# %%
def get_for_all_smart_death_lik(all_participant_ids, 
                                all_days, 
                                curr_dict_latent_data,
                                curr_dict_observed_ema,
                                curr_dict_observed_eod_survey,
                                curr_latent_params,
                                curr_selfreport_params,
                                curr_randomema_params,
                                curr_eodsurvey_params):

    # Latent model: Likelihood corresponding to each point on the grid
    dict_latent_likelihood = {}
    for this_participant in all_participant_ids: 
        current_dict = {}
        for this_day in all_days: 
            lik = grid_likelihood_latent_death(current_participant = this_participant, 
                                               current_day = this_day, 
                                               latent_params = curr_latent_params, 
                                               dict_latent_data = curr_dict_latent_data)
            current_dict.update({this_day:lik})
        dict_latent_likelihood.update({this_participant:current_dict})


    # MEM -- end of day survey subcomponent: Likelihood corresponding to each point on the grid
    dict_mem_eodsurvey_likelihood = {}
    for this_participant in all_participant_ids:
        current_dict = {}
        for this_day in all_days:  
            lik = grid_likelihood_eodsurvey_death(current_participant = this_participant, 
                                                  current_day = this_day, 
                                                  latent_params = curr_latent_params, 
                                                  eodsurvey_params = curr_eodsurvey_params, 
                                                  dict_latent_data = curr_dict_latent_data, 
                                                  dict_observed_eod_survey = curr_dict_observed_eod_survey)
            
            current_dict.update({this_day:lik})
        dict_mem_eodsurvey_likelihood.update({this_participant:current_dict})


    # MEM -- selfreport subcomponent: Likelihood corresponding to each point on the grid
    dict_mem_selfreport_likelihood = {}
    for this_participant in all_participant_ids:
        current_dict = {}
        for this_day in all_days: 
            lik = grid_likelihood_selfreport_death(current_participant = this_participant, 
                                                   current_day = this_day, 
                                                   latent_params = curr_latent_params, 
                                                   selfreport_params = curr_selfreport_params, 
                                                   dict_latent_data = curr_dict_latent_data, 
                                                   dict_observed_ema = curr_dict_observed_ema)
            
            current_dict.update({this_day:lik})
        dict_mem_selfreport_likelihood.update({this_participant:current_dict})


    # MEM -- Random EMA subcomponent: Likelihood corresponding to each point on the grid
    dict_mem_randomema_likelihood = {}
    for this_participant in all_participant_ids:
        current_dict = {}
        for this_day in all_days:  # all_days here
            lik = grid_likelihood_randomema_death(current_participant = this_participant, 
                                                  current_day = this_day, 
                                                  latent_params = curr_latent_params, 
                                                  randomema_params = curr_randomema_params, 
                                                  dict_latent_data = curr_dict_latent_data, 
                                                  dict_observed_ema = curr_dict_observed_ema)
            
            current_dict.update({this_day:lik})
        dict_mem_randomema_likelihood.update({this_participant:current_dict})
    
    dict_all = {'latent':dict_latent_likelihood, 
                'eodsurvey':dict_mem_eodsurvey_likelihood, 
                'selfreport':dict_mem_selfreport_likelihood, 
                'randomema':dict_mem_randomema_likelihood,
                'all_participant_ids':all_participant_ids,
                'all_days':all_days,
                'curr_dict_latent_data':curr_dict_latent_data}

    return dict_all

# %%

def get_for_all_smart_death_pdf(dict_smart_death_lik, dict_current_state):
    
    dict_latent_likelihood = copy.deepcopy(dict_smart_death_lik['latent'])
    dict_mem_eodsurvey_likelihood = copy.deepcopy(dict_smart_death_lik['eodsurvey'])
    dict_mem_selfreport_likelihood = copy.deepcopy(dict_smart_death_lik['selfreport'])
    dict_mem_randomema_likelihood = copy.deepcopy(dict_smart_death_lik['randomema'])

    all_participant_ids = dict_smart_death_lik['all_participant_ids']
    all_days = dict_smart_death_lik['all_days']
    curr_dict_latent_data = dict_smart_death_lik['curr_dict_latent_data']

    dict_smart_death_pdf = {}

    for current_participant in all_participant_ids: 
        current_dict_smart_death_pdf = {}
        for current_day in all_days:  
            if len(dict_current_state[current_participant][current_day]['x'])==0:
                current_element_wise_lik = np.array([])
                current_pdf_smart_death = np.array([])
                # Update dictionary for this day
                current_grid, sets_along_current_grid = get_sets_along_grid_death(current_latent_data = dict_current_state[current_participant][current_day]['x'])

            elif len(dict_current_state[current_participant][current_day]['x'])==1:
                # Calculate likelihood of moving to new state (deleting the one existing point)
                lik_latent = dict_latent_likelihood[current_participant][current_day]
                lik_eodsurvey = dict_mem_eodsurvey_likelihood[current_participant][current_day]
                lik_selfreport = dict_mem_selfreport_likelihood[current_participant][current_day]
                lik_randomema = dict_mem_randomema_likelihood[current_participant][current_day]
                current_element_wise_lik = lik_latent * lik_eodsurvey * lik_selfreport * lik_randomema
                current_pdf_smart_death = 1
                # Update dictionary for this day
                current_grid, sets_along_current_grid = get_sets_along_grid_death(current_latent_data = dict_current_state[current_participant][current_day]['x'])
            else:
                # Calculate smart death pdf
                lik_latent = dict_latent_likelihood[current_participant][current_day]
                lik_eodsurvey = dict_mem_eodsurvey_likelihood[current_participant][current_day]
                lik_selfreport = dict_mem_selfreport_likelihood[current_participant][current_day]
                lik_randomema = dict_mem_randomema_likelihood[current_participant][current_day]
                current_element_wise_lik = lik_latent * lik_eodsurvey * lik_selfreport * lik_randomema
                current_denominator_pdf_smart_death = np.sum(current_element_wise_lik)
                current_pdf_smart_death = current_element_wise_lik/current_denominator_pdf_smart_death
                # Update dictionary for this day
                current_grid, sets_along_current_grid = get_sets_along_grid_death(current_latent_data = dict_current_state[current_participant][current_day]['x'])

            current_dict_smart_death_pdf.update({current_day:{'grid':current_grid, 
                                                              'proposed_latent_smoking_times':sets_along_current_grid, 
                                                              'pdf_smart_death':current_pdf_smart_death,
                                                              'lik_smart_death':current_element_wise_lik}})
        # Update dictionary for this person
        dict_smart_death_pdf.update({current_participant:current_dict_smart_death_pdf})

    dict_all = {'dict_smart_death_pdf':dict_smart_death_pdf,
                'all_participant_ids':all_participant_ids,
                'all_days':all_days}

    return dict_all



# %%
if __name__ == '__main__':

    exec(open('../../env_vars.py').read())
    dir_picklejar = os.environ['dir_picklejar']

    filename = os.path.join(os.path.realpath(dir_picklejar), 'data_day_limits')
    infile = open(filename,'rb')
    data_day_limits = pickle.load(infile)
    infile.close()

    filename = os.path.join(os.path.realpath(dir_picklejar), 'init_latent_data_small')
    infile = open(filename,'rb')
    init_dict_latent_data = pickle.load(infile)
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
    # Enumerate all unique participant ID's and study days
    all_participant_ids = data_day_limits['participant_id'].unique()
    all_days = data_day_limits['study_day'].unique()

# %%

    current_state = get_for_all_current_state_lik(all_participant_ids = all_participant_ids, 
                                                  all_days = all_days, 
                                                  curr_dict_latent_data = init_dict_latent_data,
                                                  curr_dict_observed_ema = init_dict_observed_ema,
                                                  curr_dict_observed_eod_survey = init_dict_observed_eod_survey,
                                                  curr_latent_params = {'lambda_prequit':1, 'lambda_postquit':1},
                                                  curr_selfreport_params = {'prob_reporting_when_any': 0.90, 'prob_reporting_when_none': 0.01, 'lambda_delay': 0.5, 'sd': 30/60},
                                                  curr_randomema_params = {'prob_reporting_when_any': 0.90, 'prob_reporting_when_none': 0.01, 'sd': 30/60},
                                                  curr_eodsurvey_params = {'recall_epsilon':3, 'sd': 60/60, 'rho':0.8, 'budget':10})



# %%

    smart_birth_lik =  get_for_all_smart_birth_lik(use_increment = 1/60,
                                                   all_participant_ids = all_participant_ids, 
                                                   all_days = all_days, 
                                                   curr_dict_latent_data = init_dict_latent_data,
                                                   curr_dict_observed_ema = init_dict_observed_ema,
                                                   curr_dict_observed_eod_survey = init_dict_observed_eod_survey,
                                                   curr_latent_params = {'lambda_prequit':1, 'lambda_postquit':1},
                                                   curr_selfreport_params = {'prob_reporting_when_any': 0.90, 'prob_reporting_when_none': 0.01, 'lambda_delay': 0.5, 'sd': 30/60},
                                                   curr_randomema_params = {'prob_reporting_when_any': 0.90, 'prob_reporting_when_none': 0.01, 'sd': 30/60},
                                                   curr_eodsurvey_params = {'recall_epsilon':3, 'sd': 60/60, 'rho':0.8, 'budget':10})

    dict_smart_birth_pdf = get_for_all_smart_birth_pdf(dict_smart_birth_lik = smart_birth_lik, dict_current_state = current_state)



# %%
    for current_participant in all_participant_ids: 
        for current_day in all_days: 
            lik_latent = smart_birth_lik['latent'][current_participant][current_day]
            lik_eodsurvey = smart_birth_lik['eodsurvey'][current_participant][current_day]
            lik_selfreport = smart_birth_lik['selfreport'][current_participant][current_day]
            lik_randomema = smart_birth_lik['randomema'][current_participant][current_day]


            current_element_wise_lik = lik_latent * lik_eodsurvey * lik_selfreport * lik_randomema
            current_denominator_pdf_smart_birth = np.sum(current_element_wise_lik)
            current_pdf_smart_birth = current_element_wise_lik/current_denominator_pdf_smart_birth

            current_element_wise_lik_overall = current_element_wise_lik
            current_cdf_smart_birth_overall = np.cumsum(current_pdf_smart_birth)

            # Latent
            current_element_wise_lik = lik_latent 
            current_denominator_pdf_smart_birth = np.sum(current_element_wise_lik)
            current_pdf_smart_birth = current_element_wise_lik/current_denominator_pdf_smart_birth

            current_element_wise_lik_latent = current_element_wise_lik
            current_cdf_smart_birth_latent = np.cumsum(current_pdf_smart_birth)

            # End of day survey
            current_element_wise_lik = lik_eodsurvey 
            current_denominator_pdf_smart_birth = np.sum(current_element_wise_lik)
            current_pdf_smart_birth = current_element_wise_lik/current_denominator_pdf_smart_birth

            current_element_wise_lik_eodsurvey = current_element_wise_lik
            current_cdf_smart_birth_eodsurvey = np.cumsum(current_pdf_smart_birth)


            # Self report
            current_element_wise_lik = lik_selfreport
            current_denominator_pdf_smart_birth = np.sum(current_element_wise_lik)
            current_pdf_smart_birth = current_element_wise_lik/current_denominator_pdf_smart_birth

            current_element_wise_lik_selfreport = current_element_wise_lik
            current_cdf_smart_birth_selfreport = np.cumsum(current_pdf_smart_birth)

            # Random EMA
            current_element_wise_lik = lik_randomema
            current_denominator_pdf_smart_birth = np.sum(current_element_wise_lik)
            current_pdf_smart_birth = current_element_wise_lik/current_denominator_pdf_smart_birth

            current_element_wise_lik_randomema = current_element_wise_lik
            current_cdf_smart_birth_randomema = np.cumsum(current_pdf_smart_birth)
            
            # Plot grid
            current_grid = construct_grid(increment = 1/60, day_length = init_dict_latent_data[current_participant][current_day]['day_length'])

            # Preparation for plotting current set of latent smoking times
            current_latent_smoking_times = init_dict_latent_data[current_participant][current_day]['hours_since_start_day']
            # Preparation for plotting observed measurements -- end of day survey
            any_eod_survey = init_dict_observed_eod_survey[current_participant][current_day]['assessment_begin']
            current_checked_boxes_eod_survey = init_dict_observed_eod_survey[current_participant][current_day]['ticked_box_scaled']
            # Preparation for plotting observed measurements -- ema
            if len(init_dict_observed_ema[current_participant][current_day]['assessment_type'])>0:
                idx_selfreport = np.where(init_dict_observed_ema[current_participant][current_day]['assessment_type']=='selfreport')
                idx_random_ema = np.where(init_dict_observed_ema[current_participant][current_day]['assessment_type']=='random_ema')
                current_selfreport_ema = init_dict_observed_ema[current_participant][current_day]['assessment_begin'][idx_selfreport]        
                current_random_ema = init_dict_observed_ema[current_participant][current_day]['assessment_begin'][idx_random_ema]     
                current_random_ema_responses = init_dict_observed_ema[current_participant][current_day]['smoke'][idx_random_ema]   
            else:
                current_selfreport_ema = np.array([])
                current_random_ema = np.array([])
                current_random_ema_responses = np.array([])
            
            # Show plot
            current_day_length = np.max(current_grid)
            plt.xticks(np.arange(0, current_day_length+1, 1.0))
            plt.yticks(np.arange(0,1.1,0.1))
            plt.ylim(bottom=-0.40, top=1.50)
            plt.step(current_grid, current_cdf_smart_birth_overall, 'r-', where='post') 

            plt.step(current_grid, current_cdf_smart_birth_latent, 'grey', where='post', alpha = 0.20, linewidth = 10) 
            if len(current_latent_smoking_times)>0:
                plt.scatter(current_latent_smoking_times, np.repeat(-0.07, len(current_latent_smoking_times)), c = 'black', s=35, marker = 'o', label='Current Latent Smoking Times')
            
            plt.step(current_grid, current_cdf_smart_birth_selfreport, 'y', where='post', alpha = 0.20, linewidth = 10) 
            if len(current_selfreport_ema)>0:
                plt.scatter(current_selfreport_ema, np.repeat(-0.18, len(current_selfreport_ema)), s=30, marker = '^', c = 'orange', label='Self-Report EMA')

            plt.step(current_grid, current_cdf_smart_birth_randomema, 'b', where='post', alpha = 0.20, linewidth = 10) 
            if len(current_random_ema)>0:
                plt.scatter(current_random_ema, np.repeat(-0.18, len(current_random_ema)), s=30, marker = '^', c = 'blue', label='Random EMA')
                for idx in range(0, len(current_random_ema)):
                    plt.text(current_random_ema[idx], -0.28, current_random_ema_responses[idx], ha = 'center')
            
            plt.step(current_grid, current_cdf_smart_birth_eodsurvey, 'g', where='post', alpha = 0.20, linewidth = 10) 
            if len(any_eod_survey) > 0 and len(current_checked_boxes_eod_survey)==0:
                plt.text(0,-0.35,"End of Day Survey Completed but No Boxes Checked", ha = 'left')
            elif len(any_eod_survey)==0:
                plt.text(0,-0.35,"End of Day Survey Not Completed", ha = 'left')
            else:
                pass

            if len(current_checked_boxes_eod_survey)>0:
                list_seg = []
                for idx in range(0, len(current_checked_boxes_eod_survey)):
                    lower_lim = current_checked_boxes_eod_survey[idx]
                    upper_lim = lower_lim + 1   
                    
                    plt.scatter(lower_lim, -.13, marker = '|', s=30, c='g')
                    plt.scatter(upper_lim, -.13, marker = '|', s=30, c='g')

                    list_seg.append((lower_lim, upper_lim))
                    list_seg.append((-.13,-.13))
                    list_seg.append('g')
                
                plt.plot(*list_seg)

            plt.xlabel('Hours Elapsed Since Start of Day')
            plt.ylabel('Cumulative Density')
            plt.legend(loc='upper left', prop={'size': 9})

            plt.savefig(os.path.join(os.path.realpath(dir_picklejar), 'smart_birth_cdf_plot', 'smart_birth_cdf_{}_{}.jpg'.format(current_participant, current_day)))
            plt.clf()

# %%

    
    smart_death_lik =  get_for_all_smart_death_lik(all_participant_ids = all_participant_ids, 
                                                   all_days = all_days, 
                                                   curr_dict_latent_data = init_dict_latent_data,
                                                   curr_dict_observed_ema = init_dict_observed_ema,
                                                   curr_dict_observed_eod_survey = init_dict_observed_eod_survey,
                                                  curr_latent_params = {'lambda_prequit':1, 'lambda_postquit':1},
                                                  curr_selfreport_params = {'prob_reporting_when_any': 0.9, 'prob_reporting_when_none': 0.01, 'lambda_delay': 0.5, 'sd': 30/60},
                                                  curr_randomema_params = {'prob_reporting_when_any': 0.9, 'prob_reporting_when_none': 0.01, 'sd': 30/60},
                                                  curr_eodsurvey_params = {'recall_epsilon':3, 'sd': 60/60, 'rho':0.8, 'budget':10})

    dict_smart_death_pdf = get_for_all_smart_death_pdf(dict_smart_death_lik = smart_death_lik, dict_current_state = current_state)


# %%
    for current_participant in all_participant_ids: 
        for current_day in all_days: 
            # Preparation for plotting current set of latent smoking times
            current_latent_smoking_times = init_dict_latent_data[current_participant][current_day]['hours_since_start_day']
            # Preparation for plotting observed measurements -- end of day survey
            any_eod_survey = init_dict_observed_eod_survey[current_participant][current_day]['assessment_begin']
            current_checked_boxes_eod_survey = init_dict_observed_eod_survey[current_participant][current_day]['ticked_box_scaled']
            # Preparation for plotting observed measurements -- ema
            if len(init_dict_observed_ema[current_participant][current_day]['assessment_type'])>0:
                idx_selfreport = np.where(init_dict_observed_ema[current_participant][current_day]['assessment_type']=='selfreport')
                idx_random_ema = np.where(init_dict_observed_ema[current_participant][current_day]['assessment_type']=='random_ema')
                current_selfreport_ema = init_dict_observed_ema[current_participant][current_day]['assessment_begin'][idx_selfreport]        
                current_random_ema = init_dict_observed_ema[current_participant][current_day]['assessment_begin'][idx_random_ema]     
                current_random_ema_responses = init_dict_observed_ema[current_participant][current_day]['smoke'][idx_random_ema]   
            else:
                current_selfreport_ema = np.array([])
                current_random_ema = np.array([])
                current_random_ema_responses = np.array([])
            
            # Show plot
            current_day_length = np.max(init_dict_latent_data[current_participant][current_day]['day_length'])
            plt.xticks(np.arange(0, current_day_length+1, 1.0))
            plt.yticks(np.arange(0,1.1,0.1))
            plt.xlim(0,current_day_length+1.5)

            if len(current_latent_smoking_times)>0:
                plt.scatter(current_latent_smoking_times, np.repeat(-0.07, len(current_latent_smoking_times)), c = 'black', s=35, marker = 'o', label='Current Latent Smoking Times')
            
            
            if len(current_selfreport_ema)>0:
                plt.scatter(current_selfreport_ema, np.repeat(-0.18, len(current_selfreport_ema)), s=30, marker = '^', c = 'orange', label='Self-Report EMA')

           
            if len(current_random_ema)>0:
                plt.scatter(current_random_ema, np.repeat(-0.18, len(current_random_ema)), s=30, marker = '^', c = 'blue', label='Random EMA')
                for idx in range(0, len(current_random_ema)):
                    plt.text(current_random_ema[idx], -0.28, current_random_ema_responses[idx], ha = 'center')
            
            
            if len(any_eod_survey) > 0 and len(current_checked_boxes_eod_survey)==0:
                plt.text(0,-0.35,"End of Day Survey Completed but No Boxes Checked", ha = 'left')
            elif len(any_eod_survey)==0:
                plt.text(0,-0.35,"End of Day Survey Not Completed", ha = 'left')
            else:
                pass

            if len(current_checked_boxes_eod_survey)>0:
                list_seg = []
                for idx in range(0, len(current_checked_boxes_eod_survey)):
                    lower_lim = current_checked_boxes_eod_survey[idx]
                    upper_lim = lower_lim + 1   
                    
                    plt.scatter(lower_lim, -.13, marker = '|', s=30, c='g')
                    plt.scatter(upper_lim, -.13, marker = '|', s=30, c='g')

                    list_seg.append((lower_lim, upper_lim))
                    list_seg.append((-.13,-.13))
                    list_seg.append('g')
                
                plt.plot(*list_seg)

            plt.xlabel('Hours Elapsed Since Start of Day')

            plt.ylim(bottom=-0.40, top=1.50)
            plt.scatter(dict_smart_death_pdf['dict_smart_death_pdf'][current_participant][current_day]['grid'], 
                        np.cumsum(dict_smart_death_pdf['dict_smart_death_pdf'][current_participant][current_day]['pdf_smart_death']), 
                        s = 50, c = 'red')
            plt.ylabel('Cumulative Density')

            plt.legend(loc='upper left', prop={'size': 9})
            for idx in range(0, len(dict_smart_death_pdf['dict_smart_death_pdf'][current_participant][current_day]['grid'])):
                curr_x = dict_smart_death_pdf['dict_smart_death_pdf'][current_participant][current_day]['grid'][idx]
                curr_y = np.cumsum(dict_smart_death_pdf['dict_smart_death_pdf'][current_participant][current_day]['pdf_smart_death'])[idx]
                plt.axvline(x=curr_x, c = 'r', ls = '--')

            plt.savefig(os.path.join(os.path.realpath(dir_picklejar), 'smart_death_cdf_plot', 'smart_death_cdf_{}_{}.jpg'.format(current_participant, current_day)))
            plt.clf()


# %%
    # Calculate acceptance probability for the smart birth proposal
    # ADD HERE


# %%
    # Calculate acceptance probability for the smart death proposal
    # ADD HERE


