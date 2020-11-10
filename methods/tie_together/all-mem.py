# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import mvn
from datetime import datetime
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
data_day_limits['start_time_scaled'] = data_day_limits['start_time'].apply(lambda x: x.hour + (x.minute)/60 + (x.second)/3600) 
data_day_limits['end_time_scaled'] = data_day_limits['end_time'].apply(lambda x: x.hour + (x.minute)/60 + (x.second)/3600)

# %%
filename = os.path.join(os.path.realpath(dir_picklejar), 'init_latent_data')
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
current_participant = None
current_day = None

current_dict_observed_ema = copy.deepcopy(dict_observed_ema[current_participant][current_day])
current_dict_observed_eod_survey = copy.deepcopy(dict_observed_eod_survey[current_participant][current_day])
current_dict_observed_puffmarker = copy.deepcopy(dict_observed_puffmarker[current_participant][current_day])

# %%

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

    # Define classes corresponding to subcomponent of MEM
    # Each class simply inherits the latent and observed data of ParticipantDayMEM
    # Additionally, each of the following classes would have an attribute for parameter values

    class Latent:
        def __init__(self, outer_instance, params = None):
            self.participant = outer_instance.participant
            self.day = outer_instance.day
            self.latent_data = outer_instance.latent_data
            self.params = params

    class SelfReport:
        def __init__(self, outer_instance, params = None):
            self.participant = outer_instance.participant
            self.day = outer_instance.day
            self.latent_data = outer_instance.latent_data
            self.observed_data = outer_instance.observed_ema_data
            self.params = params
    
    class RandomEMA:
        def __init__(self, outer_instance, params = None):
            self.participant = outer_instance.participant
            self.day = outer_instance.day
            self.latent_data = outer_instance.latent_data
            self.observed_data = outer_instance.observed_ema_data
            self.params = params

    class EODSurvey:
        def __init__(self, outer_instance, params = None):
            self.participant = outer_instance.participant
            self.day = outer_instance.day
            self.latent_data = outer_instance.latent_data
            self.observed_data = outer_instance.observed_eod_survey_data
            self.params = params

    class HTMG:
        def __init__(self, outer_instance, params = None):
            self.participant = outer_instance.participant
            self.day = outer_instance.day
            self.latent_data = outer_instance.latent_data
            self.observed_data = outer_instance.observed_puffmarker_data
            self.params = params

# %%
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'tie_together', 'helper-classes.py')).read())

# %%
# Instantiate object for a particular participant day
this_object = ParticipantDayMEM(participant = current_participant, 
                                day = current_day,
                                latent_data = init_latent_data[current_participant][current_day],
                                observed_ema_data = dict_observed_ema[current_participant][current_day],
                                observed_eod_survey_data = dict_observed_eod_survey[current_participant][current_day],
                                observed_puffmarker_data = dict_observed_puffmarker[current_participant][current_day])

# Instantiate subcomponent objects for a particular participant day
latent_obj = this_object.Latent(outer_instance = this_object)
selfreport_obj = this_object.SelfReport(outer_instance = this_object)
randomema_obj = this_object.RandomEMA(outer_instance = this_object)
eodsurvey_obj = this_object.EODSurvey(outer_instance = this_object)

# %%
print(eodsurvey_obj.participant)
print(eodsurvey_obj.observed_data)

