#%%
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt
import copy

exec(open('../../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']
dir_code_methods = os.environ['dir_code_methods']

#%%
###############################################################################
# Dictionaries for latent variable models
###############################################################################

filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_selfreport')
infile = open(filename,'rb')
dict_selfreport = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_random_ema')
infile = open(filename,'rb')
dict_random_ema = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'observed_dict_puffmarker')
infile = open(filename,'rb')
dict_puffmarker = pickle.load(infile)
infile.close()

#%%
###############################################################################
# Create a data frame with records of start & end of day timestamps
# for each participant-day
###############################################################################

# output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'tie_together', 'setup-day-limits.py')).read())
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

# %%
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
      # important step to sort according to hours_since_start_day
      result.sort_values(by=['hours_since_start_day'], inplace=True)
      result['hours_since_start_day_shifted'] = result['hours_since_start_day'].shift(periods=+1)
      result['hours_since_start_day_shifted'] = np.where(pd.isna(result['hours_since_start_day_shifted']), 0, result['hours_since_start_day_shifted'])
      result['time_between'] = result['hours_since_start_day'] - result['hours_since_start_day_shifted']

      which_not_duplicate = (result['time_between']!=0)
      which_idx = np.where(which_not_duplicate)
      result = result.iloc[which_idx]

      # Let's create a time variable that depends on the value of 'smoke' #######
      # Note: in setup-pp-data-selfreport.py if when_smoke=4, delta is set to .5 hours
      result['delta'] = np.where(np.logical_and(result['assessment_type']=='selfreport', result['when_smoke']==4), result['delta'] + (result['hours_since_start_day'] - result['delta'] - result['hours_since_start_day_shifted'])/2, result['delta'])
      result['delta'] = np.where(np.logical_and(result['assessment_type']=='random_ema', result['when_smoke']==6), result['delta'] + (result['hours_since_start_day'] - result['delta'] - result['hours_since_start_day_shifted'])/2, result['delta'])

      # Assign a puff time only if participant reported 'Yes'
      result['puff_time'] = np.where(result['smoke']=='Yes', result['hours_since_start_day']-result['delta'], np.nan)
      # Rearrange columns #######################################################
      #result = result.loc[:, ['assessment_type', 'smoke','hours_since_start_day_shifted','hours_since_start_day','time_between','puff_time']]

    # Combine information into a dictionary ###################################
    new_dict = {this_study_day: result}
    current_dict.update(new_dict)
  
  # Update participant ########################################################
  all_dict.update({current_participant:current_dict})  

# %%
###############################################################################
# Do checks on dict_knitted
###############################################################################

dict_knitted = all_dict

all_participant_id = data_hq_episodes['id'].drop_duplicates()
all_participant_id.index = np.array(range(0,len(all_participant_id.index)))

total_count_flagged = 0
total_count_rows = 0

#%%

for i in range(0, len(all_participant_id)):
  current_participant = all_participant_id[i]

  for j in range(1, 15):
    this_study_day = j

    dat = dict_knitted[current_participant][this_study_day]

    if len(dat.index)>0:
      dat['flag'] = np.where(dat['puff_time'] < dat['hours_since_start_day_shifted'], 1, 0)
      dat['flag'] = np.where(pd.isna(dat['puff_time']), np.nan, dat['flag'])
      total_count_flagged += dat['flag'].sum()
      total_count_rows += len(dat.index)

      if dat['flag'].sum() > 0:
        #print(dat)
        print(current_participant, this_study_day)
    
    else:
      next

print(total_count_flagged/total_count_rows)

#%%
collect_dat = pd.DataFrame({})

for i in range(0, len(all_participant_id)):
  current_participant = all_participant_id[i]

  for j in range(1, 15):
    this_study_day = j

    dat = dict_knitted[current_participant][this_study_day]

    if len(dat.index)>0:
      subset_dat = dat.loc[dat['flag'] == 1]
    
    else:
      next

    # Now, let's concatanate ##################################################
    collect_dat = [collect_dat, subset_dat]
    collect_dat = pd.concat(collect_dat)

# Print out for checking
#collect_dat.loc[collect_dat['hours_since_start_day_shifted']==0]
#collect_dat.loc[collect_dat['hours_since_start_day_shifted']>0]
# %%

###############################################################################
# Work with flagged cases
###############################################################################

all_participant_id = data_hq_episodes['id'].drop_duplicates()
all_participant_id.index = np.array(range(0,len(all_participant_id.index)))

for i in range(0, len(all_participant_id)):
  current_participant = all_participant_id[i]

  for j in range(1, 15):
    this_study_day = j

    dat = dict_knitted[current_participant][this_study_day]

    if len(dat.index)>0:
      tmp_length = dat['hours_since_start_day'] - dat['hours_since_start_day_shifted']
      # When the very first smoking event occurs before start of day
      # That is, dat['puff_time']<0 occurs when dat['flag']==1 and and dat['hours_since_start_day_shifted']==0
      dat['puff_time'] = np.where((dat['flag']==1) & (dat['hours_since_start_day_shifted']==0), 0.5*(dat['hours_since_start_day_shifted'] + dat['hours_since_start_day']) , dat['puff_time'])

      # When the participant reported a smoking event in the current EMA to occur BEFORE
      # the previous EMA, then set puff_time to be the midpoint between the previous and current EMA
      dat['puff_time'] = np.where((dat['flag']==1) & (dat['hours_since_start_day_shifted']>0), 0.5*(dat['hours_since_start_day_shifted'] + dat['hours_since_start_day']) , dat['puff_time'])
      # Update dictionary
      dict_knitted[current_participant][this_study_day] = dat
    
    else:
      next


# %%
#########################################################################################################
# Initialize using Self-Report and Random EMA only. Do not use puffmarker.
#########################################################################################################

small_latent_data = {}

for participant in dict_knitted.keys():
    current_participant_dict = {}
    for days in dict_knitted[participant].keys():
        current_data = dict_knitted[participant][days]
        all_puff_time = []
        if len(current_data.index)==0:
            next
        else:
            current_data_yes = current_data[current_data['smoke']=='Yes']
            if len(current_data_yes)==0:
                next
            else:
                for this_row in range(0, len(current_data_yes.index)):
                    all_puff_time.append(current_data_yes['puff_time'].iloc[this_row])

        # The output of the next line should be one number only, but it will be an array object
        # Hence, the .iloc[0] converts the array object into a float
        current_day_length = data_day_limits[(data_day_limits['participant_id']==participant) & (data_day_limits['study_day']==days)]['day_length'].iloc[0]
        new_dict = {'participant_id':participant, 
                    'study_day':days, 
                    'day_length':current_day_length, 
                    'latent_event_order': (np.arange(len(all_puff_time))),  # begins with zero (not 1)
                    'hours_since_start_day': np.array(all_puff_time)}
        current_participant_dict.update({days: new_dict})
    # Add this participant's data to dictionary
    small_latent_data.update({participant:current_participant_dict})

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'init_latent_data_small')  
outfile = open(filename, 'wb')
pickle.dump(small_latent_data, outfile)
outfile.close()





