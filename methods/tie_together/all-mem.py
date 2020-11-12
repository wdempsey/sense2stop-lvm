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

# Output of this script is the data frame data_day_limits
# Each row of data_day_limits corresponds to a given participant-day
# Columns contain start of day & end of day timestamps
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'tie_together', 'setup-day-limits.py')).read())

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

filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_puffmarker')
infile = open(filename,'rb')
dict_observed_puffmarker = pickle.load(infile)
infile.close()

# %%
###############################################################################
# Helper functions
###############################################################################

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
###############################################################################
# Define work horse classes
###############################################################################

class ParticipantDayMEM:
    """
    ParticipantDayMEM is a class used to tie together various measurement error submodels
    due to different modes of measurement at the participant-day level;
    participant days will be viewed as independent of each other.

    The following are attributes of ParticipantDayMEM:
        - participant ID
        - day of the study
        - dictionary with latent smoking event times
        - dictionary with smoking events reported in EMA (Self-Report and Random)
        - dictionary with smoking events reported in end-of-day survey
        - dictionary with puffmarker identified smoking events
    
    Design notes:
        - ParticipantDayMEM will first be instantiated prior to instantiating subclasses below
        - All updates to latent data will be done through ParticipantDayMEM and not the subclasses
          to ensure that all subclasses are working off the same latent data
        - All updates to the params will be done through the individual subclasses
    """
    
    def __init__(self, 
                 participant = None, day = None, 
                 latent_data = None,
                 observed_ema_data = None, observed_eod_survey_data = None, observed_puffmarker_data = None):

        self.participant = participant
        self.day = day
        self.latent_data = latent_data
        self.observed_ema_data = observed_ema_data
        self.observed_eod_survey_data = observed_eod_survey_data
        self.observed_puffmarker_data = observed_puffmarker_data

    def inherit_all_data(self,
                         InstanceLatent = None,
                         InstanceSelfReport = None, InstanceRandomEMA = None,
                         InstanceEODSurvey = None, InstanceHTMG = None):
        """
        Initialize latent and observed data in all subcomponents in one fell swoop
        """

        # Setting default value of each argument to none allows the end-user
        # to pick and choose which particular subcomponents to use together

        if InstanceLatent is not None:
            InstanceLatent.participant = self.participant
            InstanceLatent.day = self.day
            InstanceLatent.latent_data = copy.deepcopy(self.latent_data)

        if InstanceSelfReport is not None:
            InstanceSelfReport.participant = self.participant
            InstanceSelfReport.day = self.day
            InstanceSelfReport.latent_data = copy.deepcopy(self.latent_data)
            InstanceSelfReport.observed_data = copy.deepcopy(self.observed_ema_data)

        if InstanceRandomEMA is not None:
            InstanceRandomEMA.participant = self.participant
            InstanceRandomEMA.day = self.day
            InstanceRandomEMA.latent_data = copy.deepcopy(self.latent_data)
            InstanceRandomEMA.observed_data = copy.deepcopy(self.observed_ema_data)

        if InstanceEODSurvey is not None:
            InstanceEODSurvey.participant = self.participant
            InstanceEODSurvey.day = self.day
            InstanceEODSurvey.latent_data = copy.deepcopy(self.latent_data)
            InstanceEODSurvey.observed_data = copy.deepcopy(self.observed_eod_survey_data)

        if InstanceHTMG is not None:
            InstanceHTMG.participant = self.participant
            InstanceHTMG.day = self.day
            InstanceHTMG.latent_data = copy.deepcopy(self.latent_data)
            InstanceHTMG.observed_data = copy.deepcopy(self.observed_puffmarker_data)

    def update_latent_data(self, 
                           new_latent_data,
                           InstanceLatent = None,
                           InstanceSelfReport = None,
                           InstanceRandomEMA = None,
                           InstanceEODSurvey = None,
                           InstanceHTMG = None):
        """
        Update latent data in all subcomponents in one fell swoop by applying a method to ParticipantDayMEM
        """

        self.latent_data = new_latent_data

        # Setting default value of each argument to none allows the end-user
        # to pick and choose which particular subcomponents to use together

        if InstanceLatent is not None:
            InstanceLatent.latent_data = copy.deepcopy(self.latent_data)
        
        if InstanceSelfReport is not None:
            InstanceSelfReport.latent_data = copy.deepcopy(self.latent_data)
        
        if InstanceRandomEMA is not None:
            InstanceRandomEMA.latent_data = copy.deepcopy(self.latent_data)
        
        if InstanceEODSurvey is not None:
            InstanceEODSurvey.latent_data = copy.deepcopy(self.latent_data)
        
        if InstanceHTMG is not None:
            InstanceHTMG.latent_data = copy.deepcopy(self.latent_data)

    # Define classes corresponding to subcomponent of MEM
    # Each class simply inherits the latent and observed data of ParticipantDayMEM
    # Additionally, each of the following classes would have an attribute for parameter values

    class Latent:
        def __init__(self):
            self.participant = None
            self.day = None
            self.latent_data = None
            self.params = None
        
        def calc_loglik(self):
            '''
            Calculate loglikelihood for latent process subcomponent
            '''   

            # Inputs to be checked ------------------------------------------------------------------------
            m = len(self.latent_data['latent_event_order'])

            if m==0:
                raise ValueError('Total number of latent events for current participant-day is: ', m)

            # Begin after checks on inputs have been passed -----------------------------------------------
            lambda_prequit = self.params['lambda_prequit']
            lambda_postquit = self.params['lambda_postquit']
            smoking_times = self.latent_data['hours_since_start_day']

            if self.day <4:
                total_loglik = m*np.log(lambda_prequit) - lambda_prequit*np.sum(smoking_times)
            else:
                total_loglik = m*np.log(lambda_postquit) - lambda_postquit*np.sum(smoking_times)

            return total_loglik

    class SelfReport:
        def __init__(self):
            self.participant = None
            self.day = None
            self.latent_data = None
            self.observed_data = None
            self.params = None

        def match(self):
            '''
            Call the method match after SelfReport inherits all data from ParticipantDayMEM
            '''

            # Inputs to be checked ----------------------------------------------------------------------------
            m = len(self.latent_data['latent_event_order'])
            tot_sr = np.sum(self.observed_data['assessment_type']=='selfreport')

            if m==0:
                raise ValueError('Total number of latent events for current participant-day is: ', m)
            elif tot_sr==0:
                raise ValueError('Total number of selfreport EMA for current participant-day is: ', tot_sr)
            else:
                pass

            # Begin after checks on inputs have been passed ---------------------------------------------------
            all_latent_times = self.latent_data['hours_since_start_day']
            tot_observed = len(self.observed_data['assessment_order'])
            self.observed_data['matched_latent_time'] = np.repeat(np.nan, tot_observed)

            for i in range(0, tot_observed):
                current_lb = self.observed_data['assessment_begin_shifted'][i]
                current_ub = self.observed_data['assessment_begin'][i]
                current_assessment_type = self.observed_data['assessment_type'][i]

                if current_assessment_type=='selfreport':
                    which_within = (all_latent_times >= current_lb) & (all_latent_times < current_ub)
                    which_idx = np.where(which_within)
                    matched_idx = np.max(which_idx)
                    matched_latent_time = all_latent_times[matched_idx]
                    self.observed_data['matched_latent_time'][i] = matched_latent_time

        def calc_loglik(self):
            '''
            Call the method calc_loglik after the method match has been called
            Calculate loglikelihood corresponding to self report EMA subcomponent
            '''

            # Inputs to be checked ----------------------------------------------------------------------------
            m = len(self.latent_data['latent_event_order'])
            tot_sr = np.sum(self.observed_data['assessment_type']=='selfreport')

            if m==0:
                raise ValueError('Total number of latent events for current participant-day is: ', m)
            elif tot_sr==0:
                raise ValueError('Total number of selfreport EMA for current participant-day is: ', tot_sr)
            else:
                pass

            # Begin after checks on inputs have been passed ---------------------------------------------------
            prob_reporting = 0.9
            lambda_delay = 12

            all_latent_times = self.latent_data['hours_since_start_day']
            tot_latent_events = len(all_latent_times)
            tot_reported = np.sum(self.observed_data['assessment_type']=='selfreport')

            # Subcomponent due to propensity to self-report
            total_loglik = tot_reported * np.log(prob_reporting) + (tot_latent_events - tot_reported) * np.log(1-prob_reporting)

            # Subcomponent due to delay
            self.observed_data['delay'] = self.observed_data['assessment_begin'] - self.observed_data['matched_latent_time']
            total_loglik += tot_reported * np.log(lambda_delay) - lambda_delay * np.nansum(self.observed_data['delay'])

            # Subcomponent due to recall
            tot_ema = len(self.observed_data['assessment_order'])
            self.observed_data['prob_bk'] = np.repeat(np.nan, tot_ema)
            self.observed_data['log_prob_bk'] = np.repeat(np.nan, tot_ema)

            for i in range(0, tot_ema):
                if self.observed_data['assessment_type'][i]=='selfreport':
                    current_lb = self.observed_data['assessment_begin_shifted'][i]
                    current_ub = self.observed_data['assessment_begin'][i] 
                    curr_true_time = self.observed_data['matched_latent_time'][i]

                    # Calculate denominator of bk
                    use_scale = (current_ub - current_lb)*1
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
            return total_loglik 


    class RandomEMA:
        def __init__(self):
            self.participant = None
            self.day = None
            self.latent_data = None
            self.observed_data = None
            self.params = None

        def match(self):
            '''
            Call the method match after RandomEMA inherits all data from ParticipantDayMEM
            '''

            # Inputs to be checked ----------------------------------------------------------------------------
            tot_random_ema = np.sum(self.observed_data['assessment_type']=='random_ema')
            m = len(self.latent_data['latent_event_order'])
            tot_random_ema_yes = (1*(self.observed_data['assessment_type']=='random_ema'))*(1*(self.observed_data['smoke']=='Yes'))
            tot_random_ema_yes = np.sum(tot_random_ema_yes)

            if tot_random_ema==0:
                raise ValueError('Total number of Ranodm EMA for current participant-day is: ', tot_random_ema)
            elif (m==0) and (tot_random_ema_yes>0):
                raise ValueError('Total number of latent events for current participant-day is: ', m, 
                                 'but number of Random EMA where participant reported Yes is ', tot_random_ema_yes)
            else:
                pass

            # Begin after checks on inputs have been passed ---------------------------------------------------
            all_latent_times = self.latent_data['hours_since_start_day']

            tot_observed = len(self.observed_data['assessment_order'])
            self.observed_data['matched_latent_time'] = np.repeat(np.nan, tot_observed)
            self.observed_data['any_matched_latent_time'] = np.repeat(np.nan, tot_observed)

            for i in range(0, tot_observed):
                current_assessment_type = self.observed_data['assessment_type'][i]

                if current_assessment_type == 'random_ema':
                    # First, calculate value of any_matched_latent_time
                    current_lb = self.observed_data['assessment_begin_shifted'][i]
                    current_ub = self.observed_data['assessment_begin'][i]
                    which_within = (all_latent_times >= current_lb) & (all_latent_times < current_ub)
                    self.observed_data['any_matched_latent_time'][i] = np.where(np.sum(which_within) > 0, 1, 0)

                    # Next, if participant reported 'yes' in a Random EMA, determine which latent times are matched
                    current_smoking_indicator = self.observed_data['smoke'][i]

                    if current_smoking_indicator == 'Yes':
                        which_idx = np.where(which_within)
                        matched_idx = np.max(which_idx)
                        matched_latent_time = all_latent_times[matched_idx]
                        self.observed_data['matched_latent_time'][i] = matched_latent_time

        
        def calc_loglik(self):
            '''
            Call the method calc_loglik after the method match has been called
            Calculate loglikelihood corresponding to Random EMA subcomponent
            '''

            # Inputs to be checked ----------------------------------------------------------------------------
            tot_random_ema = np.sum(self.observed_data['assessment_type']=='random_ema')
            m = len(self.latent_data['latent_event_order'])
            tot_random_ema_yes = (1*(self.observed_data['assessment_type']=='random_ema'))*(1*(self.observed_data['smoke']=='Yes'))
            tot_random_ema_yes = np.sum(tot_random_ema_yes)

            if tot_random_ema==0:
                raise ValueError('Total number of Random EMA for current participant-day is: ', tot_random_ema)
            elif (m==0) and (tot_random_ema_yes>0):
                raise ValueError('Total number of latent events for current participant-day is: ', m, 
                                 'but number of Random EMA where participant reported Yes is ', tot_random_ema_yes)
            else:
                pass

            # Begin after checks on inputs have been passed ---------------------------------------------------
            all_latent_times = self.latent_data['hours_since_start_day']
            tot_latent_events = len(all_latent_times)
            
            # Subcomponent due to recall
            tot_ema = len(self.observed_data['assessment_order'])
            self.observed_data['prob_bk'] = np.repeat(np.nan, tot_ema)
            self.observed_data['log_prob_bk'] = np.repeat(np.nan, tot_ema)

            for i in range(0, tot_ema):
                if (self.observed_data['assessment_type'][i]=='random_ema') and (self.observed_data['smoke'][i]=='Yes'):
                    current_lb = self.observed_data['assessment_begin_shifted'][i]
                    current_ub = self.observed_data['assessment_begin'][i] 
                    curr_true_time = self.observed_data['matched_latent_time'][i]

                    # Calculate denominator of bk
                    use_scale = use_scale = (current_ub - current_lb)*1
                    total_prob_constrained_lb = norm.cdf(x = current_lb, loc = curr_true_time, scale = use_scale)
                    total_prob_constrained_ub = norm.cdf(x = current_ub, loc = curr_true_time, scale = use_scale)
                    tot_prob_constrained = total_prob_constrained_ub - total_prob_constrained_lb

                    # Calculate numerator of bk
                    windowtag = self.observed_data['windowtag'][i]
                    
                    # Note: each value of windowtag corresponds to a response option in hours
                    # use_this_window_max will be based on time when prevous EMA was delivered
                    use_this_window_min = {1: 0/60, 2: 20/60, 3: 40/60, 4: 60/60, 5: 80/60, 6: 100/60}
                    use_this_window_max = {1: 20/60, 2: 40/60, 3: 60/60, 4: 80/60, 5: 100/60, 6: np.nan}
                     
                    # lower limit of integration
                    if windowtag == 6:
                        current_lk = self.observed_data['assessment_begin_shifted'][i] 
                    else:
                        current_lk = self.observed_data['assessment_begin'][i] - use_this_window_max[windowtag] 

                    # upper limit of integration
                    current_uk = self.observed_data['assessment_begin'][i] - use_this_window_min[windowtag]
                    
                    prob_constrained_lk = norm.cdf(x = current_lk, loc = curr_true_time, scale = use_scale)
                    prob_constrained_uk = norm.cdf(x = current_uk, loc = curr_true_time, scale = use_scale)

                    self.observed_data['prob_bk'][i] = (prob_constrained_uk - prob_constrained_lk)/tot_prob_constrained
                    self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                
                elif (self.observed_data['assessment_type'][i]=='random_ema') and (self.observed_data['smoke'][i]=='No'):
                    any_matched = self.observed_data['any_matched_latent_time'][i]
                    
                    if any_matched == 0:
                        self.observed_data['prob_bk'][i] = 1
                        self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                    else:
                        # A latent smoking event occurring between previous and current assessment was detected
                        # even if, at the current assessment, the participant reported 'No' smoking
                        self.observed_data['prob_bk'][i] = 0
                        self.observed_data['log_prob_bk'][i] = np.log(self.observed_data['prob_bk'][i])
                else:
                    pass

            total_loglik = np.nansum(self.observed_data['log_prob_bk'])
            return total_loglik 


    class EODSurvey:
        def __init__(self):
            self.participant = None
            self.day = None
            self.latent_data = None
            self.observed_data = None
            self.params = None

        def calc_loglik(self):
            '''
            Calculate loglikelihood corresponding to end-of-day EMA subcomponent
            '''
            
            # Inputs to be checked ----------------------------------------------------------------------------
            any_eod_ema = len(self.observed_data['assessment_begin'])

            if any_eod_ema == 0:
                raise ValueError('Total number of end-of-day EMA current participant-day is: ', any_eod_ema)

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
                    recall_epsilon = 2 # in hours

                    true_smoke_times = all_true_smoke_times[(all_true_smoke_times > curr_lk - recall_epsilon) * (all_true_smoke_times < curr_uk + recall_epsilon)]

                    # Specify covariance matrix based on an exchangeable correlation matrix
                    rho = 0.6
                    use_cormat = np.eye(len(true_smoke_times)) + rho*(np.ones((len(true_smoke_times),1)) * np.ones((1,len(true_smoke_times))) - np.eye(len(true_smoke_times)))
                    use_sd = 20/60 # in hours
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

            return loglik

    class HTMG:
        def __init__(self):
            self.participant = None
            self.day = None
            self.latent_data = None
            self.observed_data = None
            self.params = None

        # Add methods to class object:
        # - calculate log-likelihood using existing values of params



# %%
###############################################################################
# Execute workflow for one particular participant-day
###############################################################################

current_participant = None
current_day = None

# Instantiate object for a particular participant day
this_object = ParticipantDayMEM(participant = current_participant, 
                                day = current_day,
                                latent_data = init_latent_data[current_participant][current_day],
                                observed_ema_data = dict_observed_ema[current_participant][current_day],
                                observed_eod_survey_data = dict_observed_eod_survey[current_participant][current_day],
                                observed_puffmarker_data = dict_observed_puffmarker[current_participant][current_day])

# Instantiate subcomponent objects for a particular participant day
latent_obj = this_object.Latent()
selfreport_obj = this_object.SelfReport()
randomema_obj = this_object.RandomEMA()
eodsurvey_obj = this_object.EODSurvey()

# Subcomponent objects inherit all data from this_object
this_object.inherit_all_data(InstanceLatent = latent_obj,
                             InstanceSelfReport = selfreport_obj,
                             InstanceRandomEMA = randomema_obj,
                             InstanceEODSurvey = eodsurvey_obj)

# Specify parameters to be estimated
latent_obj.params = {'lambda_prequit':0.45, 'lambda_postquit':0.30}

# Prepare to calculate total loglikelihood
curr_tot_latent_events = len(latent_obj.latent_data['latent_event_order'])
curr_tot_selfreport = np.sum(selfreport_obj.observed_data['assessment_type']=='selfreport')
curr_tot_random_ema = np.sum(randomema_obj.observed_data['assessment_type']=='random_ema')
curr_tot_random_ema_yes = (1*(randomema_obj.observed_data['assessment_type']=='random_ema'))*(1*(randomema_obj.observed_data['smoke']=='Yes'))
curr_tot_random_ema_yes = np.sum(curr_tot_random_ema_yes)
curr_any_eod_ema = len(eodsurvey_obj.observed_data['assessment_begin'])

# Begin calculating total loglikelihood
total_loglik = 0

if curr_tot_latent_events != 0:
    total_loglik += latent_obj.calc_loglik()

if (curr_tot_latent_events != 0) and (curr_tot_selfreport != 0):
    selfreport_obj.match()
    total_loglik += selfreport_obj.calc_loglik()

if (curr_tot_random_ema != 0) and ~((curr_tot_random_ema_yes > 0) and (curr_tot_latent_events == 0)):
    randomema_obj.match()
    total_loglik += randomema_obj.calc_loglik()

if curr_any_eod_ema != 0:
    total_loglik += eodsurvey_obj.calc_loglik()

# Print total loglikelihood
print(total_loglik)


