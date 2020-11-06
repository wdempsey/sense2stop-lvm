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
from scipy.stats import mvn

exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']
dir_code_methods = os.environ['dir_code_methods']

# %%
class Tree:
    def __init__(self, data = None):
        self.data = data
    
    def GrowTree(self, depth):
        if depth==1:
            self.data = list([0,1])
            return self

        elif depth > 1:
            curr_level = 1
            self.data = list([0,1])

            curr_level = 2
            while curr_level <= depth:
                # Sweep through all leaves at the current level
                list_curr_level = list(np.repeat(np.nan, repeats=2**curr_level))
                for i in range(0, len(self.data)):
                    left_leaf = np.append(np.array(self.data[i]), 0)
                    right_leaf = np.append(np.array(self.data[i]), 1)
                    list_curr_level[2*i] = list(left_leaf)
                    list_curr_level[2*i + 1] = list(right_leaf)
                    #print(list_curr_level)
                    
                # Go one level below
                self.data = list_curr_level
                curr_level += 1
            return self

        else:
            return 0

# %%
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

filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_eod_survey')
infile = open(filename,'rb')
dict_eod_survey = pickle.load(infile)
infile.close()

# %%
# Demo for one participant-day
current_participant = None
current_day = None
tmp_dict_eod = copy.deepcopy(dict_eod_survey[current_participant][current_day])
tmp_dict_latent = copy.deepcopy(latent_data[current_participant][current_day])

print('Boxes checked:', tmp_dict_eod['ticked_box_raw'])
print('True latent smoking times', tmp_dict_latent['hours_since_start_day'] + tmp_dict_eod['start_time_scaled'])

# %%
tot_ticked = len(tmp_dict_eod['ticked_box_raw'])
start_day = 0
end_day = 24
all_boxes = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20])
arr_ticked = tmp_dict_eod['ticked_box_raw']
arr_not_ticked = np.setdiff1d(all_boxes, arr_ticked)
true_smoke_times = tmp_dict_latent['hours_since_start_day'] + tmp_dict_eod['start_time_scaled']
tmp_dict_latent['true_smoke_times'] = true_smoke_times
rho = 0.6
# Exchangeable correlation matrix
use_cormat = np.eye(len(true_smoke_times)) + rho*(np.ones((len(true_smoke_times),1)) * np.ones((1,len(true_smoke_times))) - np.eye(len(true_smoke_times)))
use_sd = 90/60 # in hours
use_covmat = (use_sd**2)*use_cormat
mytree = Tree()
mytree.GrowTree(depth=len(true_smoke_times))
limits_of_integration = mytree.data

total_possible_prob, error_code_total_possible_prob = mvn.mvnun(lower=np.repeat(start_day,len(true_smoke_times)),upper=np.repeat(end_day,len(true_smoke_times)),means=true_smoke_times,covar=use_covmat)

k=0  # this is the current Box k; k=0 is 8am - 9am; k=6 is 2pm; k=10 is 6pm
curr_lk = all_boxes[k] # lower limit of Box k
curr_uk = curr_lk + 1 # upper limit of Box k
collect_edge_probabilities = np.array([])

for j in range(0, len(limits_of_integration)):
    curr_limits = np.array(limits_of_integration[j])
    curr_lower_limits = np.where(curr_limits==0, start_day, curr_uk)
    curr_upper_limits = np.where(curr_limits==0, curr_lk, end_day)
    edge_probabilities, error_code_edge_probabilities = mvn.mvnun(lower=curr_lower_limits,upper=curr_upper_limits,means=true_smoke_times,covar=use_covmat)
    collect_edge_probabilities = np.append(collect_edge_probabilities, edge_probabilities)

total_edge_probabilities = np.sum(collect_edge_probabilities)
prob_none_recalled_within_current_box = total_edge_probabilities/total_possible_prob
prob_at_least_one_recalled_within_box = 1-prob_none_recalled_within_current_box

print('Current Box: ', k+8, 'am to ', k+9, 'am')
#print('Total possible prob:', total_possible_prob)
#print('Total edge probability:', total_edge_probabilities)
print('Prob at least one recalled within Current Box:', prob_at_least_one_recalled_within_box)
print('Prob at none recalled within Current Box:', prob_none_recalled_within_current_box)
print('Boxes Ticked within EOD survey', arr_ticked)
print('True latent smoking times', true_smoke_times)


# %%
# Now go through each box one by one
tot_ticked = len(tmp_dict_eod['ticked_box_raw'])
start_day = 0
end_day = 24
all_boxes = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20])
arr_ticked = tmp_dict_eod['ticked_box_raw']
arr_not_ticked = np.setdiff1d(all_boxes, arr_ticked)
true_smoke_times = tmp_dict_latent['hours_since_start_day'] + tmp_dict_eod['start_time_scaled']
tmp_dict_latent['true_smoke_times'] = true_smoke_times
rho = 0.6
# Exchangeable correlation matrix
use_cormat = np.eye(len(true_smoke_times)) + rho*(np.ones((len(true_smoke_times),1)) * np.ones((1,len(true_smoke_times))) - np.eye(len(true_smoke_times)))
use_sd = 90/60  # in hours
use_covmat = (use_sd**2)*use_cormat
mytree = Tree()
mytree.GrowTree(depth=len(true_smoke_times))
limits_of_integration = mytree.data

total_possible_prob, error_code_total_possible_prob = mvn.mvnun(lower=np.repeat(start_day,len(true_smoke_times)),upper=np.repeat(end_day,len(true_smoke_times)),means=true_smoke_times,covar=use_covmat)

collect_box_probs = np.array([])
for k in range(0, len(all_boxes)):
    curr_box = all_boxes[k] # lower limit of Box k; setting curr_lk and curr_box to be separate variables in case change of scale is needed for curr_lk
    curr_lk = all_boxes[k] # lower limit of Box k
    curr_uk = curr_lk + 1 # upper limit of Box k
    collect_edge_probabilities = np.array([])

    for j in range(0, len(limits_of_integration)):
        curr_limits = np.array(limits_of_integration[j])
        curr_lower_limits = np.where(curr_limits==0, start_day, curr_uk)
        curr_upper_limits = np.where(curr_limits==0, curr_lk, end_day)
        edge_probabilities, error_code_edge_probabilities = mvn.mvnun(lower=curr_lower_limits,upper=curr_upper_limits,means=true_smoke_times,covar=use_covmat)
        collect_edge_probabilities = np.append(collect_edge_probabilities, edge_probabilities)

    total_edge_probabilities = np.sum(collect_edge_probabilities)
    prob_none_recalled_within_current_box = total_edge_probabilities/total_possible_prob
    prob_at_least_one_recalled_within_box = 1-prob_none_recalled_within_current_box

    if curr_box in arr_ticked:
        collect_box_probs = np.append(collect_box_probs, prob_at_least_one_recalled_within_box)
    else:
        collect_box_probs = np.append(collect_box_probs, prob_none_recalled_within_current_box)

prob_observed_box_checking_pattern = np.prod(collect_box_probs)

print('True latent smoking times', true_smoke_times)
print('Boxes checked: ', arr_ticked)
print('Boxes not checked: ', arr_not_ticked)
print('Probability for observing box checking pattern for specific Box k given true latent smoking times', np.round(collect_box_probs, 3))
print('Probability of observing box checking pattern in EOD EMA given true latent smoking times', prob_observed_box_checking_pattern)

# %%

