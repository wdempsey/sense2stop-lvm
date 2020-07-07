# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:04:54 2020

@author: Balthazar
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
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'].iloc[np.where(day_temp['hours_since_start_day'] > -1)]
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'].iloc[np.where(day_temp['day_length'] - day_temp['hours_since_start_day'] > -1)]
            if day_temp['hours_since_start_day'].size > 0:
                day_min = np.min(day_temp['hours_since_start_day'])
                day_max = np.max(day_temp['hours_since_start_day'])
            else:
                day_min = 0
                day_max = day_temp['day_length']
            day_min = np.min([day_min,0])
            day_max = np.max([day_max, day_temp['day_length']])
            day_temp['hours_since_start_day'] = day_temp['hours_since_start_day'] - day_min
            day_temp['hours_since_start_day'] = np.unique(day_temp['hours_since_start_day'])
            day_temp['day_length'] = day_max - day_min



#%%

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
                total += np.log(self.model(observed,latent))
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

def selfreport_mem(observed_dict, latent_dict):
    '''
    observed: Observed self report times
    latent: Vector of latent smoking events
    '''
    observed = observed_dict['hours_since_start_day']
    latent = latent_dict['hours_since_start_day']
    total = 1.0
    if not np.all(np.isin(observed,latent)):
        total = -np.inf
    else: 
        total = np.prod(np.isin(latent,observed)*0.9 + (1-np.isin(latent,observed))*0.1)
    return total

sr_mem = measurement_model(data=clean_data, model=selfreport_mem, latent = clean_data)
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

lat_pp = latent(data=clean_data, model=latent_poisson_process, params = 1.0)

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
        for id in self.data.keys():
            for days in self.data[id].keys():
                smoke = self.data[id][days]
                sr = self.data[id][days]
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
                    llik_mem_birth = selfreport_mem(sr, new_smoke)
                    log_acceptprob = (llik_birth-llik_current) + (logtrans_death-logtrans_birth)  + (llik_mem_birth-llik_mem_current)
                    acceptprob = np.exp(log_acceptprob)
                    temp = np.random.binomial(1, p = np.min([acceptprob,1]))
                    if temp == 1:
                        print("Accepted the birth for participant %s on day %s" % (id, days))
                        smoke['hours_since_start_day'] = new_smoke['hours_since_start_day']
                elif (smoke['hours_since_start_day'].size > 0 and smoke['day_length'] > 0.0): 
                   # print("Proposing a death")
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