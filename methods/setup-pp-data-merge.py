#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt

exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']
dir_code_methods = os.environ['dir_code_methods']

#%%
###############################################################################
# Dictionaries for latent variable models
###############################################################################

filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_selfreport')
infile = open(filename,'rb')
dict_selfreport = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_random_ema')
infile = open(filename,'rb')
dict_random_ema = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_puffmarker')
infile = open(filename,'rb')
dict_puffmarker = pickle.load(infile)
infile.close()

#%%
###############################################################################
# Create a data frame with records of start & end of day timestamps
# for each participant-day
###############################################################################

# output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'setup-day-limits.py')).read())
data_reference = data_day_limits.loc[:,['participant_id','study_day']].groupby('participant_id').count().reset_index()
data_reference = data_reference.rename(columns = {'study_day':'max_study_day'})

# SANITY CHECK
#data_reference['max_study_day'].value_counts()  # this is equal to 14

#%%
###############################################################################
# Knit together various data streams
###############################################################################

all_participant_id = data_hq_episodes['id'].drop_duplicates()
all_participant_id.index = np.array(range(0,len(all_participant_id.index)))
all_dict = {}

for i in range(0, len(all_participant_id)):
  current_participant = all_participant_id[i]
  current_dict = {}

  for j in range(1, 15):
    this_study_day = j
    
    # Lets work with selfeport first ##########################################
    current_dict_selfreport = dict_selfreport[current_participant][j]
    if len(current_dict_selfreport['hours_since_start_day'])==0:
      tmp_selfreport = pd.DataFrame({})
    else:
      tmp_selfreport = pd.DataFrame({'assessment_type':'selfreport',
                                   'hours_since_start_day': current_dict_selfreport['hours_since_start_day'],
                                   'smoke': 'Yes',
                                   'when_smoke': current_dict_selfreport['message'],
                                   'delta': current_dict_selfreport['delta']
                                  })

    # Now let's work with Random EMA ##########################################
    current_dict_random_ema = dict_random_ema[current_participant][j]
    if len(current_dict_random_ema['hours_since_start_day'])==0:
      tmp_random_ema = pd.DataFrame({})
    else:
      tmp_random_ema = pd.DataFrame({'assessment_type':'random_ema',
                                    'hours_since_start_day': current_dict_random_ema['hours_since_start_day'],
                                    'smoke': current_dict_random_ema['smoke'],
                                    'when_smoke': current_dict_random_ema['when_smoke'],
                                    'delta': current_dict_random_ema['delta']
                                    })
    
    # Now, let's concatanate ##################################################
    frames = [tmp_selfreport, tmp_random_ema]
    result = pd.concat(frames)

    if len(result.index) > 0: 
      result.sort_values(by=['hours_since_start_day'], inplace=True)
      result['hours_since_start_day_shifted'] = result['hours_since_start_day'].shift(periods=+1)
      result['hours_since_start_day_shifted'] = np.where(pd.isna(result['hours_since_start_day_shifted']), 0, result['hours_since_start_day_shifted'])
      result['time_between'] = result['hours_since_start_day'] - result['hours_since_start_day_shifted']
      # Let's create a time variable that depends on the value of 'smoke' #######
      result['puff_time'] = np.where(result['smoke']=='Yes', result['hours_since_start_day']-result['delta'], pd.NA)
      # Rearrange columns #######################################################
      result = result.loc[:, ['assessment_type', 'smoke','hours_since_start_day_shifted','hours_since_start_day','time_between','puff_time']]

    # Combine information into a dictionary ###################################
    new_dict = {this_study_day: result}
    current_dict.update(new_dict)
  
  # Update participant ########################################################
  all_dict.update({current_participant:current_dict})  

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_knitted')
outfile = open(filename, 'wb')
pickle.dump(all_dict, outfile)
outfile.close()


