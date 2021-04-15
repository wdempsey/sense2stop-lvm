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
def f(x):
    loglik = x.calc_loglik()
    return(x.index, loglik)


def parallelize_class_method(list_objects, num_processes = 8):
    '''
    list_objects is a list containing instances of classes
    '''

    with Pool(processes = num_processes) as p:
        my_output = p.map(f, list_objects)
        return my_output


def grid_likelihood_latent(current_participant, current_day, latent_params, dict_latent_data):

    '''
    Calculate the likelihood at each point of a grid
    '''
    # Initialize Latent object
    init_latent_obj = Latent(participant = current_participant,
                                day = current_day,
                                latent_data = dict_latent_data[current_participant][current_day],
                                params = copy.deepcopy(latent_params))


    # Construct grid
    latent_grid = construct_grid(increment = 1/60, day_length = init_latent_obj.latent_data['day_length'])
    latent_grid_sets = get_sets_along_grid(grid = latent_grid, current_latent_data = init_latent_obj.latent_data['hours_since_start_day'])

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

def grid_likelihood_eodsurvey(current_participant, current_day, latent_params, eodsurvey_params, dict_latent_data, dict_observed_eod_survey):

    '''
    Calculate the likelihood at each point of a grid
    '''
    # Initialize EODSurvey object
    init_eodsurvey_obj = EODSurvey(participant = current_participant, 
                                    day = current_day, 
                                    latent_data = dict_latent_data[current_participant][current_day],
                                    observed_data = dict_observed_eod_survey[current_participant][current_day],
                                    params = copy.deepcopy(eodsurvey_params))

    # Construct grid
    fine_grid = construct_grid(increment = 1/60, day_length = init_eodsurvey_obj.latent_data['day_length'])
    eodsurvey_grid = construct_grid(increment = 30/60, day_length = init_eodsurvey_obj.latent_data['day_length'])
    eodsurvey_grid_sets = get_sets_along_grid(grid = eodsurvey_grid, current_latent_data = init_eodsurvey_obj.latent_data['hours_since_start_day'])

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
    if len(candidate_latent_data['hours_since_start_day']) < 6:
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
    # Note: interpolate likelihood instead og loglikelihood to avoid having to interpolate over -inf values. This will produce an error.
    f = interpolate.interp1d(x = eodsurvey_grid, y = eodsurvey_grid_lik, fill_value="extrapolate")
    interpolated_eodsurvey_grid_lik = f(fine_grid)
        
    return interpolated_eodsurvey_grid_lik


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
    init_latent_data = pickle.load(infile)
    infile.close()

    filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_eod_survey')
    infile = open(filename,'rb')
    dict_observed_eod_survey = pickle.load(infile)
    infile.close()

    filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_all_ema')
    infile = open(filename,'rb')
    dict_observed_ema = pickle.load(infile)
    infile.close()

    # Enumerate all unique participant ID's and study days
    all_participant_ids = data_day_limits['participant_id'].unique()
    all_days = data_day_limits['study_day'].unique()
    

    latent_params = {'lambda_prequit':1, 'lambda_postquit':1}
    eodsurvey_params = {'recall_epsilon':3, 'sd': 30/60, 'rho':0.8, 'budget':10}
    dict_latent_data = copy.deepcopy(init_latent_data)


# %%
    # Latent model: Likelihood corresponding to each point on the grid
    dict_latent_model_likelihood = {}
    for current_participant in all_participant_ids: 
        current_dict = {}
        for current_day in all_days:  # all_days here
            v = grid_likelihood_latent(current_participant = current_participant, 
                                       current_day = current_day, 
                                       latent_params = latent_params, 
                                       dict_latent_data = dict_latent_data)
            current_dict.update({current_day:v})
        dict_latent_model_likelihood.update({current_participant:current_dict})
# %%
    start_time = time.time()
    # MEM -- end of day survey subcomponent: Likelihood corresponding to each point on the grid
    dict_mem_eodsurvey_likelihood = {}
    for current_participant in all_participant_ids:
        current_dict = {}
        for current_day in all_days:  # all_days here
            v = grid_likelihood_eodsurvey(current_participant = current_participant, 
                                          current_day = current_day, 
                                          latent_params = latent_params, 
                                          eodsurvey_params = eodsurvey_params, 
                                          dict_latent_data = dict_latent_data, 
                                          dict_observed_eod_survey = dict_observed_eod_survey)
            current_dict.update({current_day:v})
        dict_mem_eodsurvey_likelihood.update({current_participant:current_dict})

        # For now, plot CDF using only the controbution of EOD survey
        this_participant_eodsurvey_list = dict_mem_eodsurvey_likelihood[current_participant]
        for current_day in all_days:
            # Preparation for plotting smart birth CDF
            current_element_wise_lik = this_participant_eodsurvey_list[current_day] 
            current_denominator_pdf_smart_birth = np.sum(current_element_wise_lik)
            current_pdf_smart_birth = current_element_wise_lik/current_denominator_pdf_smart_birth
            current_cdf_smart_birth = np.cumsum(current_pdf_smart_birth)
            current_grid = construct_grid(increment = 1/60, day_length = dict_latent_data[current_participant][current_day]['day_length'])
            # Preparation for plotting current set of latent smoking times
            current_latent_smoking_times = dict_latent_data[current_participant][current_day]['hours_since_start_day']
            # Preparation for plotting observed measurements -- end of day survey
            current_checked_boxes_eod_survey = dict_observed_eod_survey[current_participant][current_day]['ticked_box_scaled']
            # Show plot
            current_day_length = np.max(current_grid)
            plt.xticks(np.arange(0, current_day_length+1, 1.0))
            plt.yticks(np.arange(0,1.1,0.1))
            plt.ylim(bottom=-0.20, top=1.05)
            plt.step(current_grid, current_cdf_smart_birth, 'r-', where='post') 

            if len(current_latent_smoking_times)>0:
                plt.scatter(current_latent_smoking_times, np.repeat(-0.05, len(current_latent_smoking_times)), s=10, marker = 'o', label='Current Latent Smoking Times')
            
            if len(current_checked_boxes_eod_survey)>0:
                list_seg = []
                for idx in range(0, len(current_checked_boxes_eod_survey)):
                    lower_lim = current_checked_boxes_eod_survey[idx]
                    upper_lim = lower_lim + 1

                    plt.scatter(lower_lim, -.1, marker = '|', s=30, c='g')
                    plt.scatter(upper_lim, -.1, marker = '|', s=30, c='g')

                    list_seg.append((lower_lim, upper_lim))
                    list_seg.append((-.1,-.1))
                    list_seg.append('g')
                
                plt.plot(*list_seg)

            plt.xlabel('Hours Elapsed Since Start of Day')
            plt.ylabel('Cumulative Density')
            plt.savefig(os.path.join(os.path.realpath(dir_picklejar), 'smart_birth_cdf_plot', '{}_{}_cdf.jpg'.format(current_participant, current_day)))
            plt.clf()

            print(current_participant, current_day)
    
    end_time = time.time()
    overall_time = end_time - start_time
    print(f"Time taken {overall_time} seconds")

# %%

