# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime

from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import mvn

import os
import pickle
import copy

exec(open('../../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']
dir_code_methods = os.environ['dir_code_methods']

# %%

# Output of this script is the data frame data_day_limits
# Each row of data_day_limits corresponds to a given participant-day
# Columns contain start of day & end of day timestamps
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'tie_together', 'setup-day-limits.py')).read())

# This is an output of init_latent_data.py
# init_latent_data.py has the following outputs:
# init_latent_data_small and init_latent_data
filename = os.path.join(os.path.realpath(dir_picklejar), 'init_latent_data_small')
infile = open(filename,'rb')
init_latent_data = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_all_ema')
infile = open(filename,'rb')
dict_observed_ema = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_eod_survey')
infile = open(filename,'rb')
dict_observed_eod_survey = pickle.load(infile)
infile.close()

# %%

def GrowTree(depth):
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
                #print(list_curr_level)
                
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
    def __init__(self, participant = None, day = None, latent_data = None, params = None):
        self.participant = participant
        self.day = day
        self.latent_data = copy.deepcopy(latent_data)
        self.params = copy.deepcopy(params)
    
    def update_latent_data(self, latent_smoking_times):
        '''
        Update latent data
        '''
        self.latent_data['hours_since_start_day'] = latent_smoking_times
        self.latent_data['latent_event_order'] = np.arange(0, len(latent_smoking_times), 1)

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
            total_lik = np.exp(-lambda_prequit*day_length) * ((lambda_prequit*day_length) ** m) / np.math.factorial(m)
            total_loglik = np.log(total_lik)
        else:
            total_lik = np.exp(-lambda_postquit*day_length) * ((lambda_postquit*day_length) ** m) / np.math.factorial(m)
            total_loglik = np.log(total_lik)

        return total_loglik
    
    def construct_grid(self, increment = 1/60, sampling = False):
        '''
        Determine the points at which the overall likelihood function is to be evaluated on based on day length
        Args: 
            increment (in hours): used to construct a partition 0, increment, 2*increment, 3*increment, ...
        '''
        day_length = self.latent_data['day_length']

        if sampling == True:
            if day_length <= increment:
                buckets = np.array([0, day_length])
            else:
                buckets = np.arange(0, day_length, increment)
                
            lower_bounds = buckets[0:(len(buckets)-1)]
            upper_bounds = buckets[1:(len(buckets))]
            grid = np.random.uniform(low=lower_bounds, high=upper_bounds)
            grid = np.append(0, grid)
            grid = np.setdiff1d(ar1 = grid, ar2 = self.latent_data['hours_since_start_day'])

        else:
            if day_length <= increment:
                buckets = np.array([0, day_length])
            else:
                buckets = np.arange(0, day_length, increment)
            
            grid = buckets
            grid = np.setdiff1d(ar1 = grid, ar2 = self.latent_data['hours_since_start_day'])
        
        return(grid)

# %%

class SelfReport:
    def __init__(self, participant = None, day = None, latent_data = None, observed_data = None):
        self.participant = participant
        self.day = day
        self.latent_data = copy.deepcopy(latent_data)
        self.observed_data = copy.deepcopy(observed_data)

    def update_latent_data(self, latent_smoking_times):
        '''
        Update latent data
        '''
        self.latent_data['hours_since_start_day'] = latent_smoking_times
        self.latent_data['latent_event_order'] = np.arange(0, len(latent_smoking_times), 1)

    def match(self):
        '''
        Call the method match after SelfReport inherits all data from ParticipantDayMEM
        '''

        # Inputs to be checked --------------------------------------------
        all_latent_times = self.latent_data['hours_since_start_day']
        tot_latent_events = len(all_latent_times)
        tot_sr = np.sum(self.observed_data['assessment_type']=='selfreport')

        if tot_latent_events > 0 and tot_sr > 0:
            tot_observed = len(self.observed_data['assessment_order'])
            self.observed_data['matched_latent_time'] = np.repeat(np.nan, tot_observed)

            for i in range(0, tot_observed):
                current_lb = self.observed_data['assessment_begin_shifted'][i]
                current_ub = self.observed_data['assessment_begin'][i]
                current_assessment_type = self.observed_data['assessment_type'][i]

                if current_assessment_type=='selfreport':
                    # All latent smoking times which occur between start of day
                    # and when the current Self-Report EMA was initiated are 
                    # candidates for being matched to current Self-Report EMA
                    which_within = (all_latent_times >= 0) & (all_latent_times < current_ub)
                    if np.sum(which_within)>0:
                        which_idx = np.where(which_within)
                        matched_idx = np.max(which_idx)
                        matched_latent_time = all_latent_times[matched_idx]
                        self.observed_data['matched_latent_time'][i] = matched_latent_time
                    else:
                        # This case can occur when between time 0 and time t there is no
                        # latent smoking time, but a self-report occurred at time t
                        # This may happen after a dumb death
                        self.observed_data['matched_latent_time'][i] = np.nan

        
        else:
            # In this case, matching cannot occur
            tot_observed = len(self.observed_data['assessment_order'])
            self.observed_data['matched_latent_time'] = np.repeat(np.nan, tot_observed)


    def calc_loglik(self):
        '''
        Call the method calc_loglik after the method match has been called
        Calculate loglikelihood corresponding to self report EMA subcomponent
        '''

        # Inputs to be checked --------------------------------------------
        all_latent_times = self.latent_data['hours_since_start_day']
        tot_latent_events = len(all_latent_times)
        tot_sr = np.sum(self.observed_data['assessment_type']=='selfreport')  # Total number of Self-Report

        # Specify parameter values ----------------------------------------
        prob_reporting = 0.9
        lambda_delay = 12
        
        if tot_latent_events > 0 and tot_sr > 0:            
            all_latent_times = self.latent_data['hours_since_start_day']
            tot_latent_events = len(all_latent_times)

            # Subcomponent due to propensity to self-report
            total_loglik = tot_sr * np.log(prob_reporting) + (tot_latent_events - tot_sr) * np.log(1-prob_reporting)

            # Subcomponent due to delay
            self.observed_data['delay'] = self.observed_data['assessment_begin'] - self.observed_data['matched_latent_time']
            total_loglik += tot_sr * np.log(lambda_delay) - lambda_delay * np.nansum(self.observed_data['delay'])

            # Subcomponent due to recall
            tot_ema = len(self.observed_data['assessment_order'])
            self.observed_data['prob_bk'] = np.repeat(np.nan, tot_ema)
            self.observed_data['log_prob_bk'] = np.repeat(np.nan, tot_ema)

            check_any = 0
            
            for i in range(0, tot_ema):
                if self.observed_data['assessment_type'][i]=='selfreport':
                    current_lb = self.observed_data['assessment_begin_shifted'][i]
                    current_ub = self.observed_data['assessment_begin'][i] 
                    curr_true_time = self.observed_data['matched_latent_time'][i]

                    if np.isnan(curr_true_time):
                        check_any = -np.inf
                        break
            
            if check_any == -np.inf:
                total_loglik = -np.inf
            else:
                for i in range(0, tot_ema):
                    if self.observed_data['assessment_type'][i]=='selfreport':
                        current_lb = self.observed_data['assessment_begin_shifted'][i]
                        current_ub = self.observed_data['assessment_begin'][i] 
                        curr_true_time = self.observed_data['matched_latent_time'][i]

                        # Calculate denominator of bk
                        use_scale = current_ub - curr_true_time
                        total_prob_constrained_lb = norm.cdf(x = current_lb, loc = curr_true_time, scale = use_scale)
                        total_prob_constrained_ub = norm.cdf(x = current_ub, loc = curr_true_time, scale = use_scale)
                        tot_prob_constrained = total_prob_constrained_ub - total_prob_constrained_lb

                        # Calculate numerator of bk
                        windowtag = self.observed_data['windowtag'][i]
                        
                        # Note: each value of windowtag corresponds to a response option in hours
                        # use_this_window_max will be based on time when prevous EMA was delivered
                        use_this_window_min = {1: 0/60, 2: 5/60, 3: 15/60, 4: 30/60}
                        use_this_window_max = {1: 5/60, 2: 15/60, 3: 30/60, 4: np.nan} 
                        
                        # lower limit of integration
                        if windowtag == 4:
                            current_lk = self.observed_data['assessment_begin_shifted'][i] 
                        else:
                            current_lk = self.observed_data['assessment_begin'][i] - use_this_window_max[windowtag] 

                        # upper limit of integration
                        current_uk = self.observed_data['assessment_begin'][i] - use_this_window_min[windowtag]

                        prob_constrained_lk = norm.cdf(x = current_lk, loc = curr_true_time, scale = use_scale)
                        prob_constrained_uk = norm.cdf(x = current_uk, loc = curr_true_time, scale = use_scale)
                        self.observed_data['prob_bk'][i] = (prob_constrained_uk - prob_constrained_lk)/tot_prob_constrained
                        self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])

                # We have already exited the for loop
                total_loglik += np.nansum(self.observed_data['log_prob_bk'])

        elif tot_latent_events == 0 and tot_sr > 0:
            # This case could happen if, for example, previous move might have been a 'death'
            # but participant initiated at least one self-report.
            # Assume that participant does not lie when they Self-Report
            # Hence, set total_loglik to -np.inf
            total_loglik = -np.inf

        elif tot_latent_events > 0 and tot_sr == 0:  
            # This case could happen if, for example, previous move might have been a 'birth'
            # but there was no self-report observed.
            # Assume that participant does not lie when they Self-Report
            # However, participant may neglect to Self-Report a smoking incident
            # for example, due to burden
            total_loglik = tot_sr * np.log(prob_reporting) + (tot_latent_events - tot_sr) * np.log(1-prob_reporting)

        else:
            # In this case: tot_latent_events == 0 and tot_sr == 0
            total_loglik = 0

        return total_loglik 

# %%

class EODSurvey:
    def __init__(self, participant = None, day = None, latent_data = None, observed_data = None):
        self.participant = participant
        self.day = day
        self.latent_data = copy.deepcopy(latent_data)
        self.observed_data = copy.deepcopy(observed_data)

    def update_latent_data(self, latent_smoking_times):
        '''
        Update latent data
        '''
        self.latent_data['hours_since_start_day'] = latent_smoking_times
        self.latent_data['latent_event_order'] = np.arange(0, len(latent_smoking_times), 1)

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
            m = len(self.latent_data['latent_event_order'])  # are there any latent smoking events?
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
                    recall_epsilon = 3 # in hours

                    subset_true_smoke_times = all_true_smoke_times[(all_true_smoke_times > curr_lk - recall_epsilon) * (all_true_smoke_times < curr_uk + recall_epsilon)]
                    
                    # This is a source of randomness because at each iteration, we are not guaranteed to draw the same five points. 
                    if len(subset_true_smoke_times) > 5:
                        true_smoke_times = np.random.choice(a = subset_true_smoke_times, size = 5, replace = False)
                    else:
                        true_smoke_times = subset_true_smoke_times

                    if len(true_smoke_times) > 0:
                        # Specify covariance matrix based on an exchangeable correlation matrix
                        rho = 0.8
                        use_cormat = np.eye(len(true_smoke_times)) + rho*(np.ones((len(true_smoke_times),1)) * np.ones((1,len(true_smoke_times))) - np.eye(len(true_smoke_times)))
                        use_sd = 45/60 # in hours
                        use_covmat = (use_sd**2) * use_cormat
                        limits_of_integration = GrowTree(depth=len(true_smoke_times))

                        # Calculate total possible probability
                        total_possible_prob, error_code_total_possible_prob = mvn.mvnun(lower = np.repeat(start_day, len(true_smoke_times)),
                                                                                        upper = np.repeat(end_day, len(true_smoke_times)),
                                                                                        means = true_smoke_times,
                                                                                        covar = use_covmat)

                        # Begin calculating edge probabilities
                        collect_edge_probabilities = np.array([])

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
            loglik = 0

        return loglik

# %%

def SamplerSmartBirthDumbDeath(latent_obj, selfreport_obj, eodsurvey_obj):
    # Calculate likelihood of current configuration of points
    total_loglik = 0
    # Latent Class ########################################################
    loglik_contribution_latent = latent_obj.calc_loglik()
    total_loglik += loglik_contribution_latent
    # SelfReport Class ####################################################
    selfreport_obj.match()
    loglik_contribution_selfreport = selfreport_obj.calc_loglik()
    total_loglik += loglik_contribution_selfreport
    # EOD Survey Class ####################################################
    loglik_contribution_eodsurvey = eodsurvey_obj.calc_loglik()
    total_loglik += loglik_contribution_eodsurvey
    pi_x = np.exp(total_loglik)

    # Begin sampler: use smart birth proposal together with dumb death proposal
    # if 'add_point is selected', then
    # forward move proposal distribution is smart birth, 
    # reverse move proposal distribution is dumb death
    # on the other hand, if 'delete_point' is selected, then
    # forward move proposal distribution is dumb death, 
    # reverse move proposal distribution is smart birth
    which_move_type = np.random.choice(['add_point','delete_point'], 1, p=[0.5,0.5])
    which_move_type = which_move_type[0] # from array to string

    if which_move_type == 'add_point':
        # Construct forward move proposal distribution: smart birth
        current_latent_smoking_times = latent_obj.latent_data['hours_since_start_day']
        current_grid = latent_obj.construct_grid(increment=1, sampling=False)
        current_grid_loglik = np.array([])

        for idx_grid in range(0, len(current_grid)):
            total_loglik = 0
            try_new_point = current_grid[idx_grid]
            candidate_latent_smoking_times = np.sort(np.append(current_latent_smoking_times, try_new_point))
            latent_obj.update_latent_data(latent_smoking_times=candidate_latent_smoking_times)
            selfreport_obj.update_latent_data(latent_smoking_times=candidate_latent_smoking_times)
            eodsurvey_obj.update_latent_data(latent_smoking_times=candidate_latent_smoking_times)
            # Latent Class ########################################################
            loglik_contribution_latent = latent_obj.calc_loglik()
            total_loglik += loglik_contribution_latent
            # SelfReport Class ####################################################
            selfreport_obj.match()
            loglik_contribution_selfreport = selfreport_obj.calc_loglik()
            total_loglik += loglik_contribution_selfreport
            # EOD Survey Class ####################################################
            loglik_contribution_eodsurvey = eodsurvey_obj.calc_loglik()
            total_loglik += loglik_contribution_eodsurvey
            # Print intermediate calculations #####################################
            current_grid_loglik = np.append(current_grid_loglik, total_loglik)

        # We have exited the for loop
        current_grid_lik = np.exp(current_grid_loglik)
        current_denominator_pdf_smart_birth = np.sum(current_grid_lik)
        current_pdf_smart_birth = current_grid_lik/current_denominator_pdf_smart_birth

        # Sample from smart birth proposal distribution
        birth_point = np.random.choice(current_grid, 1, p=current_pdf_smart_birth)
        prob_smart_birth = current_pdf_smart_birth[current_grid==birth_point]
        new_latent_smoking_times_after_birth = np.sort(np.append(current_latent_smoking_times, birth_point))
        pi_xprime = np.exp(current_grid_lik[current_grid==birth_point])
        # Construct reverse move proposal distribution: dumb death
        prob_dumb_death = 1/len(new_latent_smoking_times_after_birth)
        q_xprime_given_x = prob_smart_birth
        q_x_given_xprime = prob_dumb_death

        #######################################################################
        # Calculate acceptance ratio and acceptance probability
        # Afterwards, update latent smoking times
        #######################################################################
        acceptance_ratio = (pi_xprime/pi_x) * (q_x_given_xprime/q_xprime_given_x)
        acceptance_ratio = acceptance_ratio[0]  # from array to numeric

        if ~np.isnan(acceptance_ratio):
            acceptance_prob = np.min([1, acceptance_ratio])
        else:
            acceptance_prob = 0
        
        accept_proposal = np.random.choice([0,1], 1, p=[1-acceptance_prob,acceptance_prob])
        accept_proposal = accept_proposal[0]  # from array to numeric

        if accept_proposal==1:
            use_these_latent_smoking_times = new_latent_smoking_times_after_birth
        else:
            use_these_latent_smoking_times = current_latent_smoking_times

        # 'Side effect' of function
        latent_obj.update_latent_data(latent_smoking_times=use_these_latent_smoking_times)
        selfreport_obj.update_latent_data(latent_smoking_times=use_these_latent_smoking_times)
        eodsurvey_obj.update_latent_data(latent_smoking_times=use_these_latent_smoking_times)
    
    else:
        current_latent_smoking_times = np.sort(latent_obj.latent_data['hours_since_start_day'])
        current_grid = current_latent_smoking_times
        # Sample from dumb death proposal distribution
        prob_dumb_death = 1/len(current_grid)
        deleted_point = np.random.choice(current_grid, 1, p=np.repeat(prob_dumb_death, len(current_grid)))
        new_latent_smoking_times_after_death = current_grid[current_grid!=deleted_point]
        # Calculate loglikelihood corresponding to this configuration of points
        latent_obj.update_latent_data(latent_smoking_times=new_latent_smoking_times_after_death)
        selfreport_obj.update_latent_data(latent_smoking_times=new_latent_smoking_times_after_death)
        eodsurvey_obj.update_latent_data(latent_smoking_times=new_latent_smoking_times_after_death)
        # Latent Class ########################################################
        loglik_contribution_latent = latent_obj.calc_loglik()
        total_loglik += loglik_contribution_latent
        # SelfReport Class ####################################################
        selfreport_obj.match()
        loglik_contribution_selfreport = selfreport_obj.calc_loglik()
        total_loglik += loglik_contribution_selfreport
        # EOD Survey Class ####################################################
        loglik_contribution_eodsurvey = eodsurvey_obj.calc_loglik()
        total_loglik += loglik_contribution_eodsurvey
        # Finally, calculate pi_xprime
        pi_xprime = np.exp(total_loglik)

        # Construct reverse move proposal distribution: smart birth
        # Use new_latent_smoking_times_after_death to do so
        latent_obj.update_latent_data(latent_smoking_times=new_latent_smoking_times_after_death)
        current_grid = latent_obj.construct_grid(increment=1, sampling=False)
        current_grid_loglik = np.array([])

        for idx_grid in range(0, len(current_grid)):
            total_loglik = 0
            try_new_point = current_grid[idx_grid]
            candidate_latent_smoking_times = np.sort(np.append(new_latent_smoking_times_after_death, try_new_point))
            latent_obj.update_latent_data(latent_smoking_times=candidate_latent_smoking_times)
            selfreport_obj.update_latent_data(latent_smoking_times=candidate_latent_smoking_times)
            eodsurvey_obj.update_latent_data(latent_smoking_times=candidate_latent_smoking_times)
            # Latent Class ########################################################
            loglik_contribution_latent = latent_obj.calc_loglik()
            total_loglik += loglik_contribution_latent
            # SelfReport Class ####################################################
            selfreport_obj.match()
            loglik_contribution_selfreport = selfreport_obj.calc_loglik()
            total_loglik += loglik_contribution_selfreport
            # EOD Survey Class ####################################################
            loglik_contribution_eodsurvey = eodsurvey_obj.calc_loglik()
            total_loglik += loglik_contribution_eodsurvey
            # Print intermediate calculations #####################################
            current_grid_loglik = np.append(current_grid_loglik, total_loglik)

        # We have exited the for loop
        current_grid_lik = np.exp(current_grid_loglik)
        current_denominator_pdf_smart_birth = np.sum(current_grid_lik)
        current_pdf_smart_birth = current_grid_lik/current_denominator_pdf_smart_birth

        grid_point = np.min(current_grid[deleted_point>=current_grid])
        prob_smart_birth = current_grid_lik[current_grid == grid_point]
        q_xprime_given_x = prob_dumb_death
        q_x_given_xprime = prob_smart_birth

        #######################################################################
        # Calculate acceptance ratio and acceptance probability
        # Afterwards, update latent smoking times
        #######################################################################
        acceptance_ratio = (pi_xprime/pi_x) * (q_x_given_xprime/q_xprime_given_x)
        acceptance_ratio = acceptance_ratio[0]  # from array to numeric

        if ~np.isnan(acceptance_ratio):
            acceptance_prob = np.min([1, acceptance_ratio])
        else:
            acceptance_prob = 0
        
        accept_proposal = np.random.choice([0,1], 1, p=[1-acceptance_prob,acceptance_prob])
        accept_proposal = accept_proposal[0]  # from array to numeric

        if accept_proposal==1:
            use_these_latent_smoking_times = new_latent_smoking_times_after_death
        else:
            use_these_latent_smoking_times = current_latent_smoking_times

        # 'Side effect' of function
        latent_obj.update_latent_data(latent_smoking_times=use_these_latent_smoking_times)
        selfreport_obj.update_latent_data(latent_smoking_times=use_these_latent_smoking_times)
        eodsurvey_obj.update_latent_data(latent_smoking_times=use_these_latent_smoking_times)
    
    # Calculations end here
    # Prepare output of function
    return which_move_type, acceptance_ratio, pi_xprime, pi_x, q_x_given_xprime, q_xprime_given_x, acceptance_prob, accept_proposal, current_pdf_smart_birth, current_grid

# %%

def PlotCDFSmartBirthDumbDeath(latent_obj, selfreport_obj, eodsurvey_obj, current_pdf_smart_birth, current_grid):
    current_latent_smoking_times = latent_obj.latent_data['hours_since_start_day']
    current_cdf_smart_birth = np.cumsum(current_pdf_smart_birth)
    plt.figure(clear=True)
    plt.title('Smart Birth')
    plt.xlim(left=-0.5, right= latent_obj.latent_data['day_length'] + 0.75)
    plt.ylim(bottom=-0.19, top=1.1)
    plt.vlines(x = current_grid, ymin = -10, ymax=1.2, linestyles='dashed')
    plt.vlines(x = 0, ymin = 0, ymax=np.min(current_cdf_smart_birth), colors='red')
    plt.hlines(y = 1, xmin = -100, xmax = 100, linestyles='dashed')
    plt.hlines(y = 0, xmin = -100, xmax = 100, linestyles='dashed')
    plt.hlines(y = 0, xmin = -100, xmax = np.min(current_grid), colors='red')
    plt.step(current_grid, current_cdf_smart_birth, 'r-', where='post')
    plt.scatter(current_grid, current_cdf_smart_birth, c='r', s=100)
    plt.xlabel("Hours Elapsed Since Start of Day")
    plt.ylabel("Cumulative Density Function")
    # Add latent smoking times
    plt.hlines(y = -0.05, xmin = -100, xmax = 100, colors='g')
    plt.scatter(current_latent_smoking_times, np.repeat(-0.05, len(current_latent_smoking_times)), c='g', s=100)
    # Add observed data: end-of-day survey
    plt.scatter(eodsurvey_obj.observed_data['ticked_box_scaled'], np.repeat(-0.08, len(eodsurvey_obj.observed_data['ticked_box_scaled'])), c='grey', s=100, marker='^')
    # Add observed data: selfreport
    idx_selfreport = np.where(selfreport_obj.observed_data['assessment_type']=='selfreport')
    assessment_begin_times_selfreport = selfreport_obj.observed_data['assessment_begin'][idx_selfreport]
    if len(assessment_begin_times_selfreport)>0:
        plt.scatter(assessment_begin_times_selfreport, np.repeat(-0.13, len(assessment_begin_times_selfreport)), c='grey', s=150, marker='*')

# %%
def PlotPDFSmartBirthDumbDeath(latent_obj, selfreport_obj, eodsurvey_obj, current_pdf_smart_birth, current_grid):
    current_latent_smoking_times = latent_obj.latent_data['hours_since_start_day']
    plt.figure(clear=True)
    plt.title('Smart Birth')
    plt.xlim(left=-0.5, right= latent_obj.latent_data['day_length'] + 0.75)
    plt.ylim(bottom=-0.19, top=1.1)
    plt.vlines(x = current_grid, ymin = -10, ymax=1.2, linestyles='dashed')
    plt.vlines(x = current_grid[0], ymin = 0, ymax=current_pdf_smart_birth[0], colors='red')
    plt.hlines(y = current_pdf_smart_birth[len(current_pdf_smart_birth)-1], xmin = np.max(current_grid), xmax = 100, colors='red')
    plt.hlines(y = 0, xmin = -100, xmax = np.min(current_grid), colors='red')
    plt.step(current_grid, current_pdf_smart_birth, 'r-', where='post')
    plt.scatter(current_grid, current_pdf_smart_birth, c='r', s=100)
    plt.xlabel("Hours Elapsed Since Start of Day")
    plt.ylabel("Probability Density Function")
    # Add latent smoking times
    plt.hlines(y = -0.05, xmin = -100, xmax = 100, colors='g')
    plt.scatter(current_latent_smoking_times, np.repeat(-0.05, len(current_latent_smoking_times)), c='g', s=100)
    # Add observed data: end-of-day survey
    plt.scatter(eodsurvey_obj.observed_data['ticked_box_scaled'], np.repeat(-0.08, len(eodsurvey_obj.observed_data['ticked_box_scaled'])), c='grey', s=100, marker='^')
    # Add observed data: selfreport
    idx_selfreport = np.where(selfreport_obj.observed_data['assessment_type']=='selfreport')
    assessment_begin_times_selfreport = selfreport_obj.observed_data['assessment_begin'][idx_selfreport]
    if len(assessment_begin_times_selfreport)>0:
        plt.scatter(assessment_begin_times_selfreport, np.repeat(-0.13, len(assessment_begin_times_selfreport)), c='grey', s=150, marker='*')

# %%
current_participant = None
current_day = None

# Instantiate classes
latent_obj = Latent(participant = current_participant, 
                    day = current_day,
                    latent_data = init_latent_data[current_participant][current_day],
                    params = {'lambda_prequit':0.45, 'lambda_postquit':0.30})

selfreport_obj = SelfReport(participant = current_participant, 
                            day = current_day,
                            latent_data = init_latent_data[current_participant][current_day],
                            observed_data = dict_observed_ema[current_participant][current_day])

eodsurvey_obj = EODSurvey(participant = current_participant, 
                          day = current_day,
                          latent_data = init_latent_data[current_participant][current_day],
                          observed_data = dict_observed_eod_survey[current_participant][current_day])

# Save current configuration of points before any smart birth dumb death combo will be performed
saved_latent_data = copy.deepcopy(latent_obj.latent_data)

# %%
for idx_iter in range(0,5):
    which_move_type, acceptance_ratio, pi_xprime, pi_x, q_x_given_xprime, q_xprime_given_x, acceptance_prob, accept_proposal, current_pdf_smart_birth, current_grid = SamplerSmartBirthDumbDeath(latent_obj=latent_obj, selfreport_obj=selfreport_obj, eodsurvey_obj=eodsurvey_obj)
    #print(acceptance_ratio, pi_xprime, pi_x, q_x_given_xprime, q_xprime_given_x)
    print(which_move_type, acceptance_ratio, accept_proposal)
    PlotCDFSmartBirthDumbDeath(latent_obj=latent_obj, selfreport_obj=selfreport_obj, eodsurvey_obj=eodsurvey_obj, current_pdf_smart_birth=current_pdf_smart_birth, current_grid=current_grid)
    #PlotPDFSmartBirthDumbDeath(latent_obj=latent_obj, selfreport_obj=selfreport_obj, eodsurvey_obj=eodsurvey_obj, current_pdf_smart_birth=current_pdf_smart_birth, current_grid=current_grid)

# %%

