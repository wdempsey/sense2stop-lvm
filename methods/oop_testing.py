# -*- coding: utf-8 -*-
"""
A RJMCMC code-base to fit recurrent-event models
where events are measured with uncertainty.
@author: Walter Dempsey and Jamie Yap
"""

#%%

## Import modules
import numpy as np
import os
import pickle
import scipy.special as sc

## List down file paths
exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']


#%%

###############################################################################
# Read in preparation: data_dates data frame
###############################################################################
filename = os.path.join(os.path.realpath(dir_picklejar), 'save_all_dict')
infile = open(filename,'rb')
clean_data = pickle.load(infile)
infile.close()

#%% 

'''
Delete all times > 1hr before start time. 
Extend day to handle all other times and remove duplicates
Need to move this part of code to pre-processing at some point
'''

for key in clean_data.keys():
    temp = clean_data[key]
    for days in temp.keys():
        day_temp = temp[days]
        if len(day_temp['hours_since_start_day']) > 0:
            ## Check if any times < or > 1hr 
            loc_temp = np.where(day_temp['hours_since_start_day'] > -1)
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'].iloc[loc_temp]
            day_temp['delta'] = day_temp['delta'].iloc[loc_temp]
            loc_temp = np.where(day_temp['day_length'] - day_temp['hours_since_start_day'] > -1)
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'].iloc[loc_temp]
            day_temp['delta'] = day_temp['delta'].iloc[loc_temp]
            if day_temp['hours_since_start_day'].size > 0:
                day_min = np.min(day_temp['hours_since_start_day'])
                day_max = np.max(day_temp['hours_since_start_day'])
            else:
                day_min = 0
                day_max = day_temp['day_length']
            day_min = np.min([day_min,0])
            day_max = np.max([day_max, day_temp['day_length']])
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'] - day_min
            unique_temp = np.unique(day_temp['hours_since_start_day'], return_index=True)[1]
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'].iloc[unique_temp]
            day_temp['delta'] = day_temp['delta'].iloc[unique_temp]
            day_temp['day_length'] = day_max - day_min

#%%

class measurement_model(object):
    '''
    This class constructs a measurement error subcomponent
    Attributes: 
        Data: Must provide the observed data
        Model: Computes prob of measurements given latent variables
    '''
    def __init__(self, data=0, model=0, latent = 0):
        self.data = data
        self.latent = latent
        self.model = model
        
    def compute_total_mem(self):
        total = 0 
        for id in self.data.keys():
            for days in self.data[id].keys():
                observed = self.data[id][days]
                latent = self.latent[id][days]
                total += self.model(observed,latent)
        return total

    def compute_mem_userday(self, id, days):
        total = 0 
        observed = self.data[id][days]
        latent = self.latent[id][days]
        total += np.log(self.model(observed,latent))
        return total
    
    def compute_mem(self, observed, latent):
        total = 0 
        total += np.log(self.model(observed,latent))
        return total
    
    def update_latent(self, new_latent):
        self.latent = new_latent
        return 0

#%%
'''
    Building a measurement-error model for self-report
    Input: Daily observed data, daily latent smoking times
    Output: log-likelihood for fixed MEM parameters
'''

def convert_windowtag(windowtag):
    if windowtag == 1:
        window_max = 5./60.; window_min = 0./60.
    elif windowtag == 2:
        window_max = 15./60.; window_min = 5./60.
    elif windowtag == 3:
        window_max = 30./60.; window_min = 15./60.
    else:
        window_max = 60./60.; window_min = 30./60.
    return window_min, window_max

def normal_cdf(x, mu=0, sd=1):
    '''Use scipy.special to compute cdf'''
    z = (x-mu)/sd
    return (sc.erf(z/np.sqrt(2))+1)/2 

def matching(observed_dict, latent_dict):
    ''' 
    For each obs, looks backward to see if there is a matching
    latent time (that is not taken by a prior obs).  
    Reports back matched pairs and non-matched times.
    '''
    latent = np.sort(np.array(latent_dict['hours_since_start_day']))
    obs_order = np.argsort(observed_dict['hours_since_start_day'])
    observed = np.array(observed_dict['hours_since_start_day'])[obs_order]
    delta = np.array(observed_dict['delta'])[obs_order]
    match = np.empty(shape = (1,3))
    for iter in range(len(observed)):
        current_obs = observed[iter]
        current_delta = delta[iter]
        if len(np.where(latent < current_obs)) > 0:
            temp = np.max(np.where(latent < current_obs))
            match = np.concatenate((match, [[latent[temp], current_obs, current_delta]]), axis = 0)
            latent = np.delete(latent, temp, axis = 0 )
    return match[1:], latent

def selfreport_mem(x, t, winmin, winmax):
    ''' Measurement model for self-report '''
    gap = t - x
    mem_scale = 5
    upper = normal_cdf(winmax, mu = gap, sd = mem_scale)
    lower = normal_cdf(winmin, mu = gap, sd = mem_scale)
    return np.log(upper-lower)

def selfreport_mem_total(observed_dict, latent_dict):
    '''
    observed: Observed self report times
    latent: Vector of latent smoking events
    '''
    latent_matched, latent_notmatched = matching(observed_dict, latent_dict)
    total = 1.0
    if latent_matched.shape[0] != observed_dict['hours_since_start_day'].size:
        total = -np.inf
    else: 
        total = latent_matched.size*np.log(0.9) + latent_notmatched.size*np.log(0.1)
    for row in latent_matched:
        windowmin, windowmax = convert_windowtag(row[2])
        total += selfreport_mem(row[0], row[1], windowmin, windowmax)
    return total


#%% 
'''
Making latent initial estimate for now
'''   
import copy

latent_data = copy.deepcopy(clean_data)

for key in latent_data.keys():
    for days in latent_data[key].keys(): 
        result = np.array([])
        for delta in latent_data[key][days]['delta']:
            result = np.append(result, np.mean(convert_windowtag(delta)))
        temp = latent_data[key][days]['hours_since_start_day'] - result
        latent_data[key][days]['hours_since_start_day'] = temp

#%% 
'''
Testing measurement model output
'''
sr_mem = measurement_model(data=clean_data, model=selfreport_mem_total, latent = latent_data)
sr_mem.compute_total_mem()

#%%
        
class latent(object):
    '''
    This class defines the latent process
    Attributes:
        Initial data: a first initialization of the latent process
        Model: For  
    '''
    
    def __init__(self, data=0, model=0, params=0):
        self.data = data
        self.model = model
        self.params = params        
    
    def update_params(self, new_params):
        self.params = new_params
        return 0
    
    def compute_total_pp(self):
        total = 0 
        for id in self.data.keys():
            for days in self.data[id].keys():
                latent = self.data[id][days]
                total += self.model(latent, self.params)
        return total
    
    
    def adapMH_times(self, covariance_list):
        '''
        Builds an adaptive MH for updating model parameter
        '''
        return 0
    
#%%
'''
    Building a latent poisson process model for smoking times
    Input: Daily latent smoking times
    Output: log-likelihood for fixed parameters
'''

def latent_poisson_process(latent_dict, params):
    '''
    latent: Vector of latent smoking events
    parameters: vector of parameters
    '''
    daylength = latent_dict['day_length']
    total = latent_dict['hours_since_start_day'].size * np.log(params) - params * daylength - sc.gammaln(latent_dict['hours_since_start_day'].size+1)
    return total

lat_pp = latent(data=latent_data, model=latent_poisson_process, params = 1.0)

lat_pp.compute_total_pp()
        
#%%
'''
Define the model as a latent object and a list of mem objects
'''

class model(object):
    '''
    This class defines the latent process
    Attributes:
        Initial data: a first initialization of the latent process
        Model: For  
    '''
    
    def __init__(self, init=0, latent=0, model=0):
        self.data = init # Initial smoking estimates
        self.latent = latent
        self.memmodel = model
    
    def birth_death(self, p = 0.5, smartdumb = False):
        '''
        Building a birth-death module that updates
        the latent events.
        Inputs:
            p = probability of birth-death; default is symmetric
            smartdumb = Logical variable indicating if smart-dumb proposals 
            are to be used.  Default is False.
        '''
        for participant in self.data.keys():
            for days in self.data[participant].keys():
                print("On Participant %s and day %s" % (participant, days))
                smoke = self.latent.data[participant][days]
                sr = self.memmodel.data[participant][days]
                llik_mem_current = self.memmodel.model(sr, smoke)
                llik_current= self.latent.model(smoke, params = 1.0)
                new_smoke = smoke.copy()
                birthdeath = np.random.binomial(1,0.5)
                if (birthdeath == 1 and smoke['day_length'] > 0.0):
                    #print("Proposing a birth")
                    birth = np.random.uniform(low=0.0, high = smoke['day_length'])    
                    new_smoke['hours_since_start_day'] = np.sort(np.append(new_smoke['hours_since_start_day'], birth)) 
                    logtrans_birth = np.log(p) + np.log(smoke['day_length'])
                    logtrans_death = np.log(1-p) + np.log(smoke['hours_since_start_day'].size)
                    llik_birth = self.latent.model(new_smoke, params = 1.0)
                    llik_mem_birth = self.memmodel.model(sr, new_smoke)
                    log_acceptprob = (llik_birth-llik_current) + (logtrans_death-logtrans_birth)  + (llik_mem_birth-llik_mem_current)
                    acceptprob = np.exp(log_acceptprob)
                    temp = np.random.binomial(1, p = np.min([acceptprob,1]))
                    if temp == 1:
                        print("Accepted the birth for participant %s on day %s" % (id, days))
                        smoke['hours_since_start_day'] = new_smoke['hours_since_start_day']
                elif (smoke['hours_since_start_day'].size > 0 and smoke['day_length'] > 0.0): 
                    death = np.random.randint(smoke['hours_since_start_day'].size, size = 1)
                    new_smoke['hours_since_start_day'] = np.delete(np.array(smoke['hours_since_start_day']), death, axis = 0)
                    logtrans_birth = np.log(p) + np.log(smoke['day_length'])
                    logtrans_death = np.log(1-p) + np.log(smoke['hours_since_start_day'].size)
                    llik_death = self.latent.model(new_smoke, params = 1.0)
                    llik_mem_death = self.memmodel.model(sr, new_smoke)
                    log_acceptprob = (llik_death-llik_current) + (logtrans_birth-logtrans_death) + (llik_mem_death-llik_mem_current)
                    acceptprob = np.exp(log_acceptprob)
                    temp = np.random.binomial(1, p = np.min([acceptprob,1]))
                    if temp == 1:
                        print("Accepted the death for participant %s on day %s" % (id, days))
                        smoke['hours_since_start_day'] = new_smoke['hours_since_start_day']
        return 0
    
    def adapMH_times(self, covariance_list):
        '''
        Builds an adaptive MH for updating the latent
        smoking times (account for highly irregular 
        covariance)
        '''
        return 0


#%%

test_model = model(init = clean_data,  latent = lat_pp, model = sr_mem)
test_model.birth_death()