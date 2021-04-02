# %%
from multiprocessing import Pool
import time
import numpy as np
from scipy.stats import mvn
import os
import pickle
import copy

exec(open('../../env_vars.py').read())
dir_picklejar = os.environ['dir_picklejar']

filename = os.path.join(os.path.realpath(dir_picklejar), 'init_latent_data_small')
infile = open(filename,'rb')
init_latent_data = pickle.load(infile)
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

def construct_grid(increment, day_length):
    
    if day_length <= increment:
        grid = np.array([0, day_length])
    else:
        grid = np.arange(0, day_length, increment)
    
    return grid

def get_sets_along_grid(grid, current_latent_data):

    grid = np.setdiff1d(ar1 = grid, ar2 = current_latent_data)

    M = len(grid)

    sets_along_grid = {}
    for idx_grid in range(0,M):
        new_latent_data = np.append(current_latent_data, grid[idx_grid])
        new_latent_data = np.sort(new_latent_data)
        sets_along_grid.update({idx_grid:new_latent_data})

    return sets_along_grid

# %%

class EODSurvey:
    def __init__(self, participant = None, day = None, latent_data = None, observed_data = None, params = None):
        self.participant = participant
        self.day = day
        self.latent_data = copy.deepcopy(latent_data)
        self.observed_data = copy.deepcopy(observed_data)
        self.params = copy.deepcopy(params)

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
                        limits_of_integration = GrowTree(depth=len(true_smoke_times))

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
            # If participant did not complete EOD survey, then this measurement type should not contribute to the loglikelihood
            loglik = 0

        return loglik


# %%
def parallelize_class_methods(x):
  '''
  x is an instance of EODSurveyClass which has been instantiated prior to calling 
  the function parallelize_class_methods
  '''
  loglik = x.calc_loglik()
  return loglik


# %%
current_participant = None
current_day = None

# Initialize EODSurvey object
# Do not consider birthing new points yet
init_eodsurvey_obj = EODSurvey(participant = current_participant, 
                               day = current_day, 
                               latent_data = init_latent_data[current_participant][current_day],
                               observed_data = dict_observed_eod_survey[current_participant][current_day],
                               params = {'recall_epsilon':3, 'sd': 30/60, 'rho':0.8, 'budget':10})

# %%
eodsurvey_grid = construct_grid(increment = 60/60, day_length = init_eodsurvey_obj.latent_data['day_length'])
eodsurvey_grid_sets = get_sets_along_grid(grid = eodsurvey_grid, current_latent_data = init_eodsurvey_obj.latent_data['hours_since_start_day'])

# %%
total_grid_sets = len(eodsurvey_grid_sets)

my_list = []
for idx_set in range(0, total_grid_sets):
    current_latent_data = copy.deepcopy(init_latent_data[current_participant][current_day])
    current_latent_data['hours_since_start_day'] = eodsurvey_grid_sets[idx_set]
    my_list.append(EODSurvey(participant = current_participant, 
                             day = current_day, 
                             latent_data = current_latent_data,
                             observed_data = dict_observed_eod_survey[current_participant][current_day],
                             params = {'recall_epsilon':3, 'sd': 30/60, 'rho':0.8, 'budget':10}))

# %%
if __name__ == '__main__':
    with Pool(processes = 8) as p:

        start_time = time.time()
        my_output = p.map(parallelize_class_methods, my_list)
        end_time = time.time()

        print(f"Time taken {end_time - start_time} seconds")
        print(my_output)

# %%

