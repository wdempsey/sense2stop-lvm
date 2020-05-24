#%%
import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

# List down file paths
#dir_data = "../smoking-lvm-cleaned-data/final"
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']

# Read in data
data_dates = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'participant-dates.csv'))
data_selfreport = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'self-report-smoking-final.csv'))

#%%
# Define functions
def calculate_delta(message):
    sr_accptresponse = ['Smoking Event(less than 5 minutes ago)', 
                        'Smoking Event(5 - 15 minutes ago)', 
                        'Smoking Event(15 - 30 minutes ago)']
    sr_dictionary = {'Smoking Event(less than 5 minutes ago)': 2.5, 
                     'Smoking Event(15 - 30 minutes ago)': 17.5, 
                     'Smoking Event(5 - 15 minutes ago)': 10} 

    if message in sr_accptresponse:
        # Convert time from minutes to seconds
        use_delta = sr_dictionary[message]*60  
    else:
        # If participant reported smoking more than 30 minutes ago,
        # then we consider time s/he smoked as missing
        use_delta = pd.NA  
    return use_delta

def round_day(raw_day):
    if pd.isna(raw_day):
        # Missing values for raw_day can occur
        # if participant reported smoking more than 30 minutes ago
        out_day = pd.NA
    else:
        # This takes care of the instances when participant reported to smoke 
        # less than 30 minutes ago
        if raw_day >= 0:
            # If on or after Quit Date, round down to the nearest integer
            # e.g., floor(2.7)=2
            out_day = np.floor(raw_day)
        else:
            # If before Quit Date, round up to the nearest integer
            # e.g., ceil(-2.7)=-2
            out_day = np.ceil(raw_day)
        
    return out_day

#%%

###############################################################################
# Data preparation: data_dates data frame
###############################################################################
# Create unix timestamps corresponding to 12AM of a given human-readable date
data_dates["start_date_unixts"] = (
data_dates["start_date"]
    .apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    .apply(lambda x: datetime.timestamp(x))
)

data_dates["quit_date_unixts"] = (
data_dates["quit_date"]
    .apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    .apply(lambda x: datetime.timestamp(x))
)

data_dates["expected_end_date_unixts"] = (
data_dates["expected_end_date"]
    .apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    .apply(lambda x: datetime.timestamp(x))
)

data_dates["actual_end_date_unixts"] = (
data_dates["actual_end_date"]
    .apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
    .apply(lambda x: datetime.timestamp(x))
)

# More tidying up
data_dates = (
    data_dates
        .rename(columns={"participant": "participant_id"})
        .loc[:, ["participant_id", 
                 "start_date_unixts", "quit_date_unixts",
                 "expected_end_date_unixts","actual_end_date_unixts"]]
)
#%%
###############################################################################
# Data preparation: longdf_dates data frame
###############################################################################
longdf_dates = pd.DataFrame({'participant_id':[], 
                             'start_date_unixts':[], 
                             'quit_date_unixts':[],
                             'expected_end_date_unixts':[],
                             'actual_end_date_unixts':[],
                             'quit_day':[], 
                             'study_day':[]})
n_participants = len(data_dates.index)

for i in range(0, n_participants):
    this_participant_dates = data_dates.iloc[i]
    total_study_days = this_participant_dates["actual_end_date_unixts"] - this_participant_dates["start_date_unixts"]
    total_study_days = total_study_days/(60*60*24)
    total_study_days = np.int64(total_study_days)

    if total_study_days > 0:
        participant_id = np.int64(this_participant_dates["participant_id"])
        start_date_unixts = np.int64(this_participant_dates["start_date_unixts"])
        quit_date_unixts = np.int64(this_participant_dates["quit_date_unixts"])
        expected_end_date_unixts = np.int64(this_participant_dates["expected_end_date_unixts"])
        actual_end_date_unixts = np.int64(this_participant_dates["actual_end_date_unixts"])

        quit_day = this_participant_dates["quit_date_unixts"] - this_participant_dates["start_date_unixts"]
        quit_day = quit_day/(60*60*24)
        quit_day = np.int64(quit_day)

        this_participant_df = {'participant_id':np.repeat(participant_id, total_study_days),
                               'start_date_unixts':np.repeat(start_date_unixts, total_study_days),
                               'quit_date_unixts':np.repeat(quit_date_unixts, total_study_days),
                               'expected_end_date_unixts':np.repeat(expected_end_date_unixts, total_study_days),
                               'actual_end_date_unixts':np.repeat(actual_end_date_unixts, total_study_days),
                               'quit_day':np.repeat(quit_day, total_study_days),
                               'study_day':np.array(range(0, total_study_days))
                              }
        this_participant_df = pd.DataFrame(this_participant_df)
        longdf_dates = longdf_dates.append(this_participant_df)
    else:
        pass

#%%
longdf_dates["day_since_quit"] = (
    longdf_dates
    .loc[:, ["study_day", "quit_day"]]
    .pipe(lambda x: (x["study_day"] - x["quit_day"]))
)

longdf_dates["is_post_quit"] = (
    longdf_dates
    .loc[:, ["day_since_quit"]]
    .pipe(lambda x: 1*(x["day_since_quit"]>=0))
)

longdf_dates["day_within_period"] = (
    longdf_dates.loc[:, ["study_day","day_since_quit","is_post_quit"]]
    .pipe(lambda x: x["is_post_quit"]*x["day_since_quit"] + (1-x["is_post_quit"])*x["study_day"])
)

#%%

###############################################################################
# Merge data_selfreport with data_dates
###############################################################################
data_selfreport = data_selfreport.merge(data_dates, how = 'left', on = 'participant_id')

###############################################################################
# Data preparation: data_selfreport data frame
###############################################################################
data_selfreport["begin_unixts"] = data_selfreport["timestamp"]/1000
data_selfreport["delta"] = data_selfreport["message"].apply(lambda x: calculate_delta(x))
data_selfreport["smoked_unixts"] = data_selfreport["begin_unixts"] - data_selfreport["delta"]

# Create a new variable, study_day: number of days since participant entered
# the study
data_selfreport["study_day"] = (
        data_selfreport
        .loc[:, ["start_date_unixts","smoked_unixts"]]
        .pipe(lambda x: (x["smoked_unixts"]-x["start_date_unixts"])/(60*60*24))
        .apply(lambda x: round_day(x))
)

# Create a new variable, day_since_quit: number of days before or after 
# 12AM on Quit Date
data_selfreport["day_since_quit"] = (
    data_selfreport
        .loc[:, ["quit_date_unixts","smoked_unixts"]]
        .pipe(lambda x: (x["smoked_unixts"]-x["quit_date_unixts"])/(60*60*24))
        .apply(lambda x: round_day(x))
)

# Drop columns with missing values in the smoked_unixts variable
data_selfreport = data_selfreport.dropna(how = 'any', subset=['smoked_unixts'])
data_selfreport["study_day"] = data_selfreport["study_day"].apply(lambda x: np.int(x))
data_selfreport["day_since_quit"] = data_selfreport["day_since_quit"].apply(lambda x: np.int(x))

# Create a new variable, is_post_quit: whether a given day falls before or on/after 12AM on Quit Date
data_selfreport["is_post_quit"] = data_selfreport["day_since_quit"].apply(lambda x: 0 if x < 0 else 1)

# Create a new variable, day_within_period: 
# if is_post_quit<0, number of days after 12AM on start of study
# if is_post_quit>=0, number of days after 12AM on Quit Date
# hence day_within_period is a count variable with ZERO as minimum value
data_selfreport["day_within_period"] = np.where(data_selfreport["is_post_quit"]==0,
                                                data_selfreport["study_day"], 
                                                data_selfreport["day_since_quit"])

# Finally, select subset of columns
use_these_columns = ["participant_id", "start_date_unixts", "quit_date_unixts",
                     "expected_end_date_unixts","actual_end_date_unixts",
                     "is_post_quit", "study_day", "day_since_quit", "day_within_period",
                     "begin_unixts", "message", "delta", "smoked_unixts"]
data_selfreport = data_selfreport.loc[:, use_these_columns]

#%%
###############################################################################
# For each row in longdf_dates, count number of recorded selfreport timestamps
###############################################################################

# Loop through longdf_dates and count number of reported smoking events 
# for each person day
collect_count = []
for i in range(0, len(longdf_dates.index)):
    df_person_day = longdf_dates.iloc[i]
    this_participant_id = df_person_day["participant_id"]
    this_study_day = df_person_day["study_day"]
    grabbed_data = data_selfreport[(data_selfreport.participant_id == this_participant_id) & (data_selfreport.study_day == this_study_day)]
    this_count = len(grabbed_data.index)
    collect_count.append(this_count)

df_counts = longdf_dates
df_counts = df_counts.assign(counts_sr = collect_count)

#%%
###############################################################################
# In df_counts, identify which individuals have "trailing zeros"
###############################################################################

# Note: All rows belonging to individuals whose last 7 or erlier days consist
# of "trailing zeros" are excluded

aggregations = {"counts_sr": lambda x: 1 if sum(x.tail(7))>0 else 0}
df_trailing_zeros = (
    df_counts
        .loc[:, ["participant_id","counts_sr"]]
        .groupby("participant_id")
        .agg(aggregations)
        .reset_index(drop=False)
        .rename(columns={"counts_sr": "keep_participant"})
)

df_counts_subset_trailingzeros = df_counts.merge(df_trailing_zeros, how = 'left', on = 'participant_id')
df_counts_subset_trailingzeros = df_counts_subset_trailingzeros[df_counts_subset_trailingzeros.keep_participant==1].drop("keep_participant", axis=1)

# Exclude this participant
df_counts_subset_trailingzeros = df_counts_subset_trailingzeros[df_counts_subset_trailingzeros["participant_id"] != 227]

#%%
###############################################################################
# In df_counts, identify which individuals have zeros on all days
###############################################################################

# Note: All rows belonging to individuals whose last 7 or erlier days consist
# of "trailing zeros" are excluded

aggregations = {"counts_sr": lambda x: 1 if sum(x)>0 else 0}
df_all_zeros = (
    df_counts
        .loc[:, ["participant_id","counts_sr"]]
        .groupby("participant_id")
        .agg(aggregations)
        .reset_index(drop=False)
        .rename(columns={"counts_sr": "keep_participant"})
)

df_counts_subset_allzeros = df_counts.merge(df_all_zeros, how = 'left', on = 'participant_id')
df_counts_subset_allzeros = df_counts_subset_allzeros[df_counts_subset_allzeros.keep_participant==1].drop("keep_participant", axis=1)

# Exclude this participant
df_counts_subset_allzeros = df_counts_subset_allzeros[df_counts_subset_allzeros["participant_id"] != 227]

#%%
###############################################################################
# Write out csv file for prepared data if write_out==True
###############################################################################
write_out = True

if write_out:
    df_counts.to_csv(os.path.join(os.path.realpath(dir_data), 'work_with_counts_all.csv'), index=False)
    df_counts_subset_trailingzeros.to_csv(os.path.join(os.path.realpath(dir_data), 'work_with_counts_subset_trailingzeros.csv'), index=False)
    df_counts_subset_allzeros.to_csv(os.path.join(os.path.realpath(dir_data), 'work_with_counts_subset_allzeros.csv'), index=False)

#%%
###############################################################################
# Data preparation: Create data to be used as input to pymc3
###############################################################################

# Collect data to be used in analyses in a dictionary
collect_data_analysis = {}
collect_data_analysis['df_counts'] = df_counts
collect_data_analysis['df_counts_subset_trailingzeros'] = df_counts_subset_trailingzeros
collect_data_analysis['df_counts_subset_allzeros'] = df_counts_subset_allzeros
#%%
# collect_results is a dictionary that will collect results across all models
collect_results={}

#%%
###############################################################################
# Estimation using pymc3
###############################################################################
use_this_data = collect_data_analysis['df_counts_subset_allzeros']

# Create new participant id's
participant_names = use_this_data['participant_id'].unique()
n_participants = len(participant_names)
d = {'participant_id':participant_names, 'participant_idx':np.array(range(0,n_participants))}
reference_df = pd.DataFrame(d)
use_this_data = use_this_data.merge(reference_df, how = 'left', on = 'participant_id')
participant_idx = use_this_data['participant_idx'].values

#%%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    # Outcome Data
    Y_observed = pm.Data('counts_sr', use_this_data['counts_sr'].values)

    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta = pm.Normal('beta', mu=0, sd=10)
    gamma = pm.Normal('gamma', mu=0, sd=10, shape=n_participants)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    logmu = beta + gamma[participant_idx]
    mu = np.exp(logmu)
    Y_hat = pm.Poisson('Y_hat', mu=mu, observed=Y_observed)

#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=10000, tune=5000, cores=1, target_accept=0.90)

#%%
# Calculate 95% credible interval
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]

# Produce trace plots
pm.traceplot(posterior_samples)

# Transform coefficients and recover mu value
model_summary_expscale = np.exp(model_summary_logscale)
model_summary_expscale = model_summary_expscale.rename(index=lambda x: 'exp('+x+')') 

# Round up to 3 decimal places
model_summary_logscale = model_summary_logscale.round(3)
model_summary_expscale = model_summary_expscale.round(3)

# Collect results
collect_results['0'] = {'model':model, 
                        'posterior_samples':posterior_samples,
                        'model_summary_logscale':model_summary_logscale,
                        'model_summary_expscale':model_summary_expscale}

# Remove variable from workspace
del model, posterior_samples, model_summary_logscale, model_summary_expscale

#%%

###############################################################################
# Estimation using pymc3
###############################################################################
use_this_data = collect_data_analysis['df_counts_subset_allzeros']

# Create new participant id's
participant_names = use_this_data['participant_id'].unique()
n_participants = len(participant_names)
d = {'participant_id':participant_names, 'participant_idx':np.array(range(0,n_participants))}
reference_df = pd.DataFrame(d)
use_this_data = use_this_data.merge(reference_df, how = 'left', on = 'participant_id')
participant_idx = use_this_data['participant_idx'].values

#%%

with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    # Outcome Data
    Y_observed = pm.Data('counts_sr', use_this_data['counts_sr'].values)
    
    # Covariate Data
    is_post_quit = pm.Data('is_post_quit', use_this_data['is_post_quit'].values)
    day_within_period = pm.Data('day_within_period', use_this_data['day_within_period'].values)

    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    gamma_prequit = pm.Normal('gamma_prequit', mu=0, sd=10, shape = n_participants)
    gamma_postquit = pm.Normal('gamma_postquit', mu=0, sd=10, shape = n_participants)
    
    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    logmu_prequit = beta_prequit + gamma_prequit[participant_idx]
    mu_prequit = np.exp(logmu_prequit)

    logmu_postquit = beta_postquit + gamma_postquit[participant_idx]
    mu_postquit = np.exp(logmu_postquit)
    mu = (1-is_post_quit)*mu_prequit + is_post_quit*mu_postquit

    Y_hat = pm.Poisson('Y_hat', mu=mu, observed=Y_observed)

#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=15000, tune=5000, cores=1, target_accept=0.90)

#%%
# Calculate 95% credible interval
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]

# Produce trace plots
pm.traceplot(posterior_samples)

# Transform coefficients and recover mu value
model_summary_expscale = np.exp(model_summary_logscale)
model_summary_expscale = model_summary_expscale.rename(index=lambda x: 'exp('+x+')') 

# Round up to 3 decimal places
model_summary_logscale = model_summary_logscale.round(3)
model_summary_expscale = model_summary_expscale.round(3)

# Collect results
collect_results['1'] = {'model':model, 
                        'posterior_samples':posterior_samples,
                        'model_summary_logscale':model_summary_logscale,
                        'model_summary_expscale':model_summary_expscale}
#%%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale, model_summary_expscale
#%%

###############################################################################
# Print results from all models
###############################################################################
import matplotlib.pyplot as plt

# Model 0
pm.traceplot(collect_results['0']['posterior_samples'])
print(collect_results['0']['model_summary_logscale'])
print(collect_results['0']['model_summary_expscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['0']['posterior_samples'], var_names=['beta'], credible_interval=0.95)
pm.forestplot(collect_results['0']['posterior_samples'], var_names=['gamma'], credible_interval=0.95)

#%%
# Model 1
pm.traceplot(collect_results['1']['posterior_samples'])
print(collect_results['1']['model_summary_logscale'])
print(collect_results['1']['model_summary_expscale'])

plt.figure(figsize=(4,8))
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_prequit'], credible_interval=0.95)
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['beta_postquit'], credible_interval=0.95)
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['gamma_prequit'], credible_interval=0.95)
pm.forestplot(collect_results['1']['posterior_samples'], var_names=['gamma_postquit'], credible_interval=0.95)

# %%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_randeff_models_allzeros')
outfile = open(filename, 'wb')
pickle.dump(collect_results, outfile)
outfile.close()

# %%
