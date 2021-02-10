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
                    which_idx = np.where(which_within)
                    matched_idx = np.max(which_idx)
                    matched_latent_time = all_latent_times[matched_idx]
                    self.observed_data['matched_latent_time'][i] = matched_latent_time
        
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

# %%

for idx_sim in range(0,30):
    current_latent_smoking_times = latent_obj.latent_data['hours_since_start_day']
    current_grid = latent_obj.construct_grid()
    current_grid_loglik = np.array([])

    for idx_grid in range(0, len(current_grid)):
        total_loglik = 0
        try_new_point = current_grid[idx_grid]
        candidate_latent_smoking_times = np.sort(np.append(current_latent_smoking_times, try_new_point))
        latent_obj.update_latent_data(latent_smoking_times=candidate_latent_smoking_times)
        selfreport_obj.update_latent_data(latent_smoking_times=candidate_latent_smoking_times)
        # Latent Class ########################################################
        loglik_contribution_latent = latent_obj.calc_loglik()
        total_loglik += loglik_contribution_latent
        # SelfReport Class ####################################################
        selfreport_obj.match()
        loglik_contribution_selfreport = selfreport_obj.calc_loglik()
        total_loglik += loglik_contribution_selfreport
        # Print intermediate caluclations #####################################
        current_grid_loglik = np.append(current_grid_loglik, total_loglik)

    current_grid_lik = np.exp(current_grid_loglik)
    current_pdf = current_grid_lik/np.sum(current_grid_lik)
    
    # Construct distribution
    new_point = np.random.choice(current_grid, 1, p=current_pdf)
    new_latent_smoking_times = np.sort(np.append(current_latent_smoking_times, new_point))
    latent_obj.update_latent_data(latent_smoking_times=new_latent_smoking_times)
    selfreport_obj.update_latent_data(latent_smoking_times=new_latent_smoking_times)

    if idx_sim in np.array([0,29]):
        print(new_point)
        print(new_latent_smoking_times)
        plt.figure(clear=True)
        plt.ylim(bottom=0.0, top=0.009)
        plt.plot(current_grid, current_pdf)
        plt.xlabel("Hours Elapsed Since Start of Day")
        plt.ylabel("Density")


# %%





