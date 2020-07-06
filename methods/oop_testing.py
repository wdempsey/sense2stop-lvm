# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:04:54 2020

@author: Balthazar
"""

#%%

## Import modules
import pandas as pd
import numpy as np
import os
import pickle

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

latent_data = clean_data

#%%

class measurement_model(object):
    '''
    This class constructs a measurement error subcomponent
    Attributes: 
        Data: Must provide the observed data
        Model: Computes prob of measurements given latent variables
    '''
    def __init__(self, data=0, model=0):
        self.data = data
        self.model = model
        
    def compute_mem(self, latent_dict):
        for id in self.data.keys():
            for days in self.data[id].keys():
                observed = self.data[id][days]
                latent = latent_dict[id][days]
                print(np.log(self.model(observed,latent)))
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

sr_mem = measurement_model(data=clean_data, model=selfreport_mem)
sr_mem.compute_mem(latent_data)

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
    
    def compute_pp(self):
        for id in self.data.keys():
            for days in self.data[id].keys():
                latent = self.data[id][days]
                print(self.model(latent, self.params))
        return 0
    
    def adapMH_times(self, covariance_list):
        '''
        Builds an adaptive MH for updating the latent
        smoking times (account for highly irregular 
        covariance)
        '''
        return 0
    
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
    latent = latent_dict['hours_since_start_day']
    daylength = latent_dict['day_length']
    total = len(latent)* np.log(params) - params * daylength
    return total

lat_pp = latent(data=latent_data, model=latent_poisson_process, params = 1.0)

lat_pp.compute_pp()
        
#%%
'''
Define the model as a latent object and a list of mem objects
'''

def model(object):
    '''
    This class defines the latent process
    Attributes:
        Initial data: a first initialization of the latent process
        Model: For  
    '''
    
    def __init__(self, init=0, latent=0, model=0):
        self.data = init # Initial smoking estimates
        self.latent = latent
        self.model = model
        return 0
    
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
            for days in self.smoke[id].keys():
                latent = self.data[id][days]
                birthdeath = np.random.binomial(1,0.5)
                if (birthdeath == 1):
                    birth = np.random.uniform(low=0.0, high = latent['day_length'])
                    new_latent = np.sort(np.append(latent['hours_since_start_day'], birth)) 
                    trans_birth = p * latent['day_length']
                    trans_death = (1-p) * latent['hours_since_start_day'].size
                
                
        return 0
