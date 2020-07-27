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

#%%
###############################################################################
# Create a data frame with records of start & end of day timestamps
# for each participant-day
###############################################################################
dir_code_methods = os.environ['dir_code_methods']

# output of this script is the data frame data_day_limits
exec(open(os.path.join(os.path.realpath(dir_code_methods), 'setup-day-limits.py')).read())


###############################################################################
# Dictionaries for latent variable models
###############################################################################

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_selfreport')
infile = open(filename,'rb')
dict_selfreport = pickle.load(infile)
infile.close()

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_random_ema')
infile = open(filename,'rb')
dict_random_ema = pickle.load(infile)
infile.close()

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_puffmarker')
infile = open(filename,'rb')
dict_puffmarker = pickle.load(infile)
infile.close()

#%%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_knitted')
infile = open(filename,'rb')
dict_knitted = pickle.load(infile)
infile.close()


###############################################################################
# Do checks on dict_knitted
###############################################################################

#%%
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
    
    else:
      next

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
      dat['puff_time'] = np.where((dat['flag']==1) & (dat['hours_since_start_day_shifted']==0), 0 , dat['puff_time'])
      dat['puff_time'] = np.where((dat['flag']==1) & (dat['hours_since_start_day_shifted']>0), 0.5*(dat['hours_since_start_day_shifted'] + dat['hours_since_start_day']) , dat['puff_time'])
      # Update dictionary
      dict_knitted[current_participant][this_study_day] = dat
    
    else:
      next

# %%

###############################################################################
# Now, knit puffmarker data into the fabric of random EMA & self-report data
###############################################################################

all_participant_id = data_hq_episodes['id'].drop_duplicates()
all_participant_id.index = np.array(range(0,len(all_participant_id.index)))

for i in range(0, len(all_participant_id)):
  current_participant = all_participant_id[i]

  for j in range(1, 15):
    this_study_day = j
    # dat_survey contains both Random EMA & self-report data
    dat_survey = dict_knitted[current_participant][this_study_day]
    # create an indicator for whether an adjustment using puffmarker timestamp was made
    dat_survey['adjusted'] = 0
    # dat_puffmarker only contains puffmarker data
    dat_puffmarker = dict_puffmarker[current_participant][this_study_day]
    # Now, go through the rows of dat_survey one by one
    if len(dat_survey.index)>0:
      for k in range(0, len(dat_survey.index)):
        if dat_survey['smoke'].iloc[k]=='Yes':
          LB = dat_survey['hours_since_start_day_shifted'].iloc[k]
          UB = dat_survey['hours_since_start_day'].iloc[k]
          # Check whether puffmarker timestamp is within two consecutive assessments
          # and revise puff_time accordingly
          subset_dat_puffmarker = dat_puffmarker['hours_since_start_day'][(dat_puffmarker['hours_since_start_day']>=LB) & (dat_puffmarker['hours_since_start_day']<=UB)]
          if len(subset_dat_puffmarker)==1:
            subset_dat_puffmarker = subset_dat_puffmarker.to_dict()
            dat_survey['adjusted'] = 1
            dat_survey['puff_time'].iloc[k] = [subset_dat_puffmarker]
          elif len(subset_dat_puffmarker)>=2:
            subset_dat_puffmarker = subset_dat_puffmarker.to_dict()
            dat_survey['adjusted'] = 1
            dat_survey['puff_time'].iloc[k] = [subset_dat_puffmarker]
          else:
            dat_survey['adjusted'] = 0
        else:
          dat_survey['adjusted'] = np.nan

      # Update dictionary
      dict_knitted[current_participant][this_study_day] = dat_survey
    
    else:
      next


###############################################################################
# Do checks on dict_knitted
###############################################################################

#%%

all_participant_id = data_hq_episodes['id'].drop_duplicates()
all_participant_id.index = np.array(range(0,len(all_participant_id.index)))

total_count_flagged = 0
total_count_rows = 0
total_count_adjusted = 0


for i in range(0, len(all_participant_id)):
  current_participant = all_participant_id[i]

  for j in range(1, 15):
    this_study_day = j

    dat = dict_knitted[current_participant][this_study_day]

    if len(dat.index)>0:
      total_count_flagged += dat['flag'].sum()
      total_count_rows += len(dat.index)
      total_count_adjusted += dat['adjusted'].sum()
    
    else:
      next

#%%
collect_dat = pd.DataFrame({})

for i in range(0, len(all_participant_id)):
  current_participant = all_participant_id[i]

  for j in range(1, 15):
    this_study_day = j

    dat = dict_knitted[current_participant][this_study_day]

    if len(dat.index)>0:
      subset_dat = dat.loc[dat['adjusted'] == 1]
    
    else:
      next

    # Now, let's concatanate ##################################################
    collect_dat = [collect_dat, subset_dat]
    collect_dat = pd.concat(collect_dat)

#%%

