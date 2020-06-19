# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Summary
# 
# * ADD LATER
# * ADD LATER
# * ADD LATER
# %% [markdown]
# # Estimation

# %%
import pymc3 as pm
import arviz as az
import pandas as pd
import numpy as np
from datetime import datetime
import os

exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']

# %% [markdown]
# Only self-report data will be used to estimate time between events for now.

# %%
data_selfreport = pd.read_csv(os.path.join(os.path.realpath(dir_data), 'work_with_datapoints.csv'))
use_this_data = data_selfreport

# %% [markdown]
# Let's define the distribution of censored data.

# %%
def exponential_log_complementary_cdf(x, lam):
    ''' log complementary CDF of exponential distribution '''
    return -lam*x

# %% [markdown]
# Let's pull out variables that will be used in all models.

# %%
censored = use_this_data['censored'].values.astype(bool)
time_to_next_event = use_this_data['time_to_next_event'].values.astype(float)
is_post_quit = use_this_data['is_post_quit'].values.astype(float)

# %% [markdown]
# Let's pull out features we have constructed.

# %%
# Features applicable to pre- and post-quit periods
day_within_period = use_this_data['day_within_period'].values.astype(float)
hours_since_previous_sr_within_day = use_this_data['hours_since_previous_sr_within_day'].values.astype(float)
hours_since_previous_sr_within_period = use_this_data['hours_since_previous_sr_within_period'].values.astype(float)
is_first_sr_within_day = use_this_data['is_first_sr_within_day'].values.astype(float)
is_first_sr_within_period = use_this_data['is_first_sr_within_period'].values.astype(float)
order_within_day = use_this_data['order_within_day'].values.astype(float)
order_within_period = use_this_data['order_within_period'].values.astype(float)
hours_since_start_of_study = use_this_data['hours_since_start_of_study'].values.astype(float)
hours_since_start_of_period = use_this_data['hours_since_start_of_period'].values.astype(float)
hour_of_day = use_this_data['hour_of_day'].values.astype(float)
sleep = use_this_data['sleep'].values.astype(float)  # 1=if between 1am to 6am, 0=outside of this time

# Features applicable only to the post-quit period
is_within24hours_quit = use_this_data['is_within24hours_quit'].values.astype(float)
is_within48hours_quit = use_this_data['is_within48hours_quit'].values.astype(float)
is_within72hours_quit = use_this_data['is_within72hours_quit'].values.astype(float)

# %% [markdown]
# ## Model 1

# %%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    beta_prequit_day = pm.Normal('beta_prequit_day', mu=0, sd=10)
    beta_postquit_day = pm.Normal('beta_postquit_day', mu=0, sd=10)
    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = (
        beta_prequit*(1-is_post_quit[~censored]) + beta_prequit_day*day_within_period[~censored]*(1-is_post_quit[~censored])
        + beta_postquit*is_post_quit[~censored] + beta_postquit_day*day_within_period[~censored]*is_post_quit[~censored]
        )

    lamb_observed = np.exp(loglamb_observed)

    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

    loglamb_censored = (
        beta_prequit*(1-is_post_quit[censored]) + beta_prequit_day*day_within_period[censored]*(1-is_post_quit[censored]) 
        + beta_postquit*is_post_quit[censored] + beta_postquit_day*day_within_period[censored]*is_post_quit[censored]
    )

    lamb_censored = np.exp(loglamb_censored)

    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))

# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=1000, tune=1000, cores=1, init='adapt_diag', target_accept=0.90, max_treedepth=50)


# %%
# Calculate 95% credible interval
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]
model_summary_logscale


# %%
summary_expscale = {'mean': [np.mean(np.exp(posterior_samples['beta_prequit_day'])), np.mean(np.exp(posterior_samples['beta_postquit_day']))],
                    'LB': [np.quantile(np.exp(posterior_samples['beta_prequit_day']), q=.125), np.quantile(np.exp(posterior_samples['beta_postquit_day']), q=.125)],
                    'UB': [np.quantile(np.exp(posterior_samples['beta_prequit_day']), q=.975), np.quantile(np.exp(posterior_samples['beta_postquit_day']), q=.975)]}

summary_expscale = pd.DataFrame(summary_expscale)
summary_expscale.index = ['exp_beta_prequit_day','exp_beta_postquit_day']
summary_expscale


# %%
pm.traceplot(posterior_samples)


# %%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale

# %% [markdown]
# ## Model 2

# %%
feature1 = hours_since_previous_sr_within_period


# %%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    beta_prequit_feature1 = pm.Normal('beta_prequit_feature1', mu=0, sd=10)
    beta_postquit_feature1 = pm.Normal('beta_postquit_feature1', mu=0, sd=10)


    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = (
        beta_prequit*(1-is_post_quit[~censored]) + beta_postquit*is_post_quit[~censored]
        )

    loglamb_observed_features = (
        beta_prequit_feature1*feature1[~censored]*(1-is_post_quit[~censored]) +
        beta_postquit_feature1*feature1[~censored]*is_post_quit[~censored]
    )

    lamb_observed = np.exp(loglamb_observed + loglamb_observed_features)

    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

    loglamb_censored = (
        beta_prequit*(1-is_post_quit[censored]) + beta_postquit*is_post_quit[censored]
    )
    
    loglamb_censored_features = (
        beta_prequit_feature1*feature1[censored]*(1-is_post_quit[censored]) +
        beta_postquit_feature1*feature1[censored]*is_post_quit[censored]
    )

    lamb_censored = np.exp(loglamb_censored + loglamb_censored_features)

    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))

#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=1000, tune=1000, cores=1, init='adapt_diag', target_accept=0.90, max_treedepth=50)


# %%
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]
model_summary_logscale


# %%
posterior_samples_expscale_prequit_feature1 = np.exp(posterior_samples['beta_prequit_feature1'])
posterior_samples_expscale_postquit_feature1 = np.exp(posterior_samples['beta_postquit_feature1'])

model_summary_expscale = {'mean': [np.mean(posterior_samples_expscale_prequit_feature1), np.mean(posterior_samples_expscale_postquit_feature1)],
                          'LB': [np.quantile(posterior_samples_expscale_prequit_feature1, q=.125), np.quantile(posterior_samples_expscale_postquit_feature1, q=.125)],
                          'UB': [np.quantile(posterior_samples_expscale_prequit_feature1, q=.975), np.quantile(posterior_samples_expscale_postquit_feature1, q=.975)]}

model_summary_expscale = pd.DataFrame(model_summary_expscale)
model_summary_expscale.index = ['exp_beta_prequit_feature1', 'exp_beta_postquit_feature1']
model_summary_expscale


# %%
diff_prepost_feature1 = posterior_samples['beta_postquit_feature1'] - posterior_samples['beta_prequit_feature1']
exp_diff_prepost_feature1 = np.exp(diff_prepost_feature1)

diff_summary_expscale = {'mean': [np.mean(exp_diff_prepost_feature1)],
                          'LB': [np.quantile(exp_diff_prepost_feature1, q=.125)],
                          'UB': [np.quantile(exp_diff_prepost_feature1, q=.975)]}

diff_summary_expscale = pd.DataFrame(diff_summary_expscale)
diff_summary_expscale.index = ['exp_diff_prepost_feature1']
diff_summary_expscale


# %%
pm.traceplot(posterior_samples)

# %% [markdown]
# ## Model 3

# %%
feature1 = is_within48hours_quit
feature2 = hours_since_previous_sr_within_period


# %%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    beta_postquit_feature1 = pm.Normal('beta_postquit_feature1', mu=0, sd=10)
    beta_prequit_feature2 = pm.Normal('beta_prequit_feature2', mu=0, sd=10)
    beta_postquit_feature2 = pm.Normal('beta_postquit_feature2', mu=0, sd=10)
    beta_postquit_feature_product = pm.Normal('beta_postquit_feature_product', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = (
        beta_prequit*(1-is_post_quit[~censored]) + beta_postquit*is_post_quit[~censored] 
        )

    loglamb_observed_features1 = (
        beta_postquit_feature1*feature1[~censored]*is_post_quit[~censored] +
        beta_prequit_feature2*feature2[~censored]*(1-is_post_quit[~censored]) +
        beta_postquit_feature2*feature2[~censored]*is_post_quit[~censored] +
        beta_postquit_feature_product*feature1[~censored]*feature2[~censored]*is_post_quit[~censored]
    )

    lamb_observed = np.exp(loglamb_observed + loglamb_observed_features1)

    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

    loglamb_censored = (
        beta_prequit*(1-is_post_quit[censored]) + beta_postquit*is_post_quit[censored] 
    )
    
    loglamb_censored_features1 = (
        beta_postquit_feature1*feature1[censored]*is_post_quit[censored] +
        beta_prequit_feature2*feature2[censored]*(1-is_post_quit[censored]) +
        beta_postquit_feature2*feature2[censored]*is_post_quit[censored] +
        beta_postquit_feature_product*feature1[censored]*feature2[censored]*is_post_quit[censored]
    )

    lamb_censored = np.exp(loglamb_censored + loglamb_censored_features1)

    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))

with model:
    posterior_samples = pm.sample(draws=1000, tune=1000, cores=1, init='adapt_diag', target_accept=0.90, max_treedepth=50)


# %%
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]
model_summary_logscale


# %%
# Slope of hours since previous self-report within period:

# Difference between within first 48 hours in post-quit period vs. after first 48 hours in post-quit period
diff_feature_postquitwithin48_postquitafter48 = posterior_samples['beta_postquit_feature_product']
exp_diff_feature_postquitwithin48_postquitafter48 = np.exp(diff_feature_postquitwithin48_postquitafter48)

# Difference between within first 48 hours in post-quit period vs. pre-quit
diff_feature_postquitwithin48_prequit = posterior_samples['beta_postquit_feature2'] + posterior_samples['beta_postquit_feature_product'] - posterior_samples['beta_prequit_feature2']
exp_diff_feature_postquitwithin48_prequit = np.exp(diff_feature_postquitwithin48_prequit)

# Difference between after 48 hours in post-quit period vs. pre-quit
diff_feature_postquitafter48_prequit = posterior_samples['beta_postquit_feature2'] - posterior_samples['beta_prequit_feature2']
exp_diff_feature_postquitafter48_prequit = np.exp(diff_feature_postquitafter48_prequit)

diff_summary_expscale = {'mean': [np.mean(exp_diff_feature_postquitwithin48_postquitafter48), np.mean(exp_diff_feature_postquitwithin48_prequit), np.mean(exp_diff_feature_postquitafter48_prequit)],
                          'LB': [np.quantile(exp_diff_feature_postquitwithin48_postquitafter48, q=.125), np.quantile(exp_diff_feature_postquitwithin48_prequit, q=.125), np.quantile(exp_diff_feature_postquitafter48_prequit, q=.125)],
                          'UB': [np.quantile(exp_diff_feature_postquitwithin48_postquitafter48, q=.975), np.quantile(exp_diff_feature_postquitwithin48_prequit, q=.975), np.quantile(exp_diff_feature_postquitafter48_prequit, q=.975)]}

diff_summary_expscale = pd.DataFrame(diff_summary_expscale)
diff_summary_expscale.index = ['exp_diff_feature_postquitwithin48_postquitafter48','exp_diff_feature_postquitwithin48_prequit','exp_diff_feature_postquitafter48_prequit']
diff_summary_expscale


# %%
pm.traceplot(posterior_samples)


# %%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale


# %%
## Model 4


# %%
feature1 = order_within_day


# %%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    beta_prequit_feature1 = pm.Normal('beta_prequit_feature1', mu=0, sd=10)
    beta_postquit_feature1 = pm.Normal('beta_postquit_feature1', mu=0, sd=10)


    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = (
        beta_prequit*(1-is_post_quit[~censored]) + beta_postquit*is_post_quit[~censored]
        )

    loglamb_observed_features = (
        beta_prequit_feature1*feature1[~censored]*(1-is_post_quit[~censored]) +
        beta_postquit_feature1*feature1[~censored]*is_post_quit[~censored]
    )

    lamb_observed = np.exp(loglamb_observed + loglamb_observed_features)

    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

    loglamb_censored = (
        beta_prequit*(1-is_post_quit[censored]) + beta_postquit*is_post_quit[censored]
    )
    
    loglamb_censored_features = (
        beta_prequit_feature1*feature1[censored]*(1-is_post_quit[censored]) +
        beta_postquit_feature1*feature1[censored]*is_post_quit[censored]
    )

    lamb_censored = np.exp(loglamb_censored + loglamb_censored_features)

    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))

#%%
# Sample from posterior distribution
with model:
    posterior_samples = pm.sample(draws=1000, tune=1000, cores=1, init='adapt_diag', target_accept=0.90, max_treedepth=50)


# %%
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]
model_summary_logscale


# %%
posterior_samples_expscale_prequit_feature1 = np.exp(posterior_samples['beta_prequit_feature1'])
posterior_samples_expscale_postquit_feature1 = np.exp(posterior_samples['beta_postquit_feature1'])

model_summary_expscale = {'mean': [np.mean(posterior_samples_expscale_prequit_feature1), np.mean(posterior_samples_expscale_postquit_feature1)],
                          'LB': [np.quantile(posterior_samples_expscale_prequit_feature1, q=.125), np.quantile(posterior_samples_expscale_postquit_feature1, q=.125)],
                          'UB': [np.quantile(posterior_samples_expscale_prequit_feature1, q=.975), np.quantile(posterior_samples_expscale_postquit_feature1, q=.975)]}

model_summary_expscale = pd.DataFrame(model_summary_expscale)
model_summary_expscale.index = ['exp_beta_prequit_feature1', 'exp_beta_postquit_feature1']
model_summary_expscale


# %%
# Difference between pre-quit and post-quit periods:
# time to first self-report
diff_prepost_feature1 = posterior_samples['beta_postquit_feature1'] - posterior_samples['beta_prequit_feature1']
exp_diff_prepost_feature1 = np.exp(diff_prepost_feature1)

diff_summary_expscale = {'mean': [np.mean(exp_diff_prepost_feature1)],
                          'LB': [np.quantile(exp_diff_prepost_feature1, q=.125)],
                          'UB': [np.quantile(exp_diff_prepost_feature1, q=.975)]}

diff_summary_expscale = pd.DataFrame(diff_summary_expscale)
diff_summary_expscale.index = ['exp_diff_prepost_feature1']
diff_summary_expscale


# %%
pm.traceplot(posterior_samples)

# %% [markdown]
# ## Model 5

# %%
feature1 = is_within48hours_quit
feature2 = order_within_day


# %%
with pm.Model() as model:
    # -------------------------------------------------------------------------
    # Priors
    # -------------------------------------------------------------------------
    beta_prequit = pm.Normal('beta_prequit', mu=0, sd=10)
    beta_postquit = pm.Normal('beta_postquit', mu=0, sd=10)
    beta_postquit_feature1 = pm.Normal('beta_postquit_feature1', mu=0, sd=10)
    beta_prequit_feature2 = pm.Normal('beta_prequit_feature2', mu=0, sd=10)
    beta_postquit_feature2 = pm.Normal('beta_postquit_feature2', mu=0, sd=10)
    beta_postquit_feature_product = pm.Normal('beta_postquit_feature_product', mu=0, sd=10)

    # -------------------------------------------------------------------------
    # Likelihood
    # -------------------------------------------------------------------------
    loglamb_observed = (
        beta_prequit*(1-is_post_quit[~censored]) + beta_postquit*is_post_quit[~censored] 
        )

    loglamb_observed_features1 = (
        beta_postquit_feature1*feature1[~censored]*is_post_quit[~censored] +
        beta_prequit_feature2*feature2[~censored]*(1-is_post_quit[~censored]) +
        beta_postquit_feature2*feature2[~censored]*is_post_quit[~censored] +
        beta_postquit_feature_product*feature1[~censored]*feature2[~censored]*is_post_quit[~censored]
    )

    lamb_observed = np.exp(loglamb_observed + loglamb_observed_features1)

    Y_hat_observed = pm.Exponential('Y_hat_observed', lam = lamb_observed, observed=time_to_next_event[~censored])

    loglamb_censored = (
        beta_prequit*(1-is_post_quit[censored]) + beta_postquit*is_post_quit[censored] 
    )
    
    loglamb_censored_features1 = (
        beta_postquit_feature1*feature1[censored]*is_post_quit[censored] +
        beta_prequit_feature2*feature2[censored]*(1-is_post_quit[censored]) +
        beta_postquit_feature2*feature2[censored]*is_post_quit[censored] +
        beta_postquit_feature_product*feature1[censored]*feature2[censored]*is_post_quit[censored]
    )

    lamb_censored = np.exp(loglamb_censored + loglamb_censored_features1)

    Y_hat_censored = pm.Potential('Y_hat_censored', exponential_log_complementary_cdf(x = time_to_next_event[censored], lam = lamb_censored))

with model:
    posterior_samples = pm.sample(draws=1000, tune=1000, cores=1, init='adapt_diag', target_accept=0.90, max_treedepth=50)


# %%
model_summary_logscale = az.summary(posterior_samples, credible_interval=.95)
model_summary_logscale = model_summary_logscale[['mean','hpd_2.5%','hpd_97.5%']]
model_summary_logscale


# %%
posterior_samples_expscale_postquit_feature1 = np.exp(posterior_samples['beta_postquit_feature1'])
posterior_samples_expscale_prequit_feature2 = np.exp(posterior_samples['beta_prequit_feature2'])
posterior_samples_expscale_postquit_feature2 = np.exp(posterior_samples['beta_postquit_feature2'])
posterior_samples_expscale_postquit_feature_product = np.exp(posterior_samples['beta_postquit_feature_product'])

model_summary_expscale = {'mean': [np.mean(posterior_samples_expscale_postquit_feature1), 
                                   np.mean(posterior_samples_expscale_prequit_feature2), 
                                   np.mean(posterior_samples_expscale_postquit_feature2), 
                                   np.mean(posterior_samples_expscale_postquit_feature_product)],
                          'LB': [np.quantile(posterior_samples_expscale_postquit_feature1, q=.125), 
                                 np.quantile(posterior_samples_expscale_prequit_feature2, q=.125), 
                                 np.quantile(posterior_samples_expscale_postquit_feature2, q=.125),
                                 np.quantile(posterior_samples_expscale_postquit_feature_product, q=.125)],
                          'UB': [np.quantile(posterior_samples_expscale_postquit_feature1, q=.975), 
                                 np.quantile(posterior_samples_expscale_prequit_feature2, q=.975), 
                                 np.quantile(posterior_samples_expscale_postquit_feature2, q=.975),
                                 np.quantile(posterior_samples_expscale_postquit_feature_product, q=.975)]}

model_summary_expscale = pd.DataFrame(model_summary_expscale)
model_summary_expscale.index = ['exp_beta_postquit_feature1','exp_beta_prequit_feature2', 'exp_beta_postquit_feature2','exp_beta_postquit_feature_product']
model_summary_expscale


# %%
# Time to first self-report within period:

# Difference between within first 48 hours in post-quit period vs. after first 48 hours in post-quit period
diff_feature_postquitwithin48_postquitafter48 = posterior_samples['beta_postquit_feature_product']
exp_diff_feature_postquitwithin48_postquitafter48 = np.exp(diff_feature_postquitwithin48_postquitafter48)

# Difference between within first 48 hours in post-quit period vs. pre-quit
diff_feature_postquitwithin48_prequit = posterior_samples['beta_postquit_feature2'] + posterior_samples['beta_postquit_feature_product'] - posterior_samples['beta_prequit_feature2']
exp_diff_feature_postquitwithin48_prequit = np.exp(diff_feature_postquitwithin48_prequit)

# Difference between after 48 hours in post-quit period vs. pre-quit
diff_feature_postquitafter48_prequit = posterior_samples['beta_postquit_feature2'] - posterior_samples['beta_prequit_feature2']
exp_diff_feature_postquitafter48_prequit = np.exp(diff_feature_postquitafter48_prequit)

diff_summary_expscale = {'mean': [np.mean(exp_diff_feature_postquitwithin48_postquitafter48), np.mean(exp_diff_feature_postquitwithin48_prequit), np.mean(exp_diff_feature_postquitafter48_prequit)],
                          'LB': [np.quantile(exp_diff_feature_postquitwithin48_postquitafter48, q=.125), np.quantile(exp_diff_feature_postquitwithin48_prequit, q=.125), np.quantile(exp_diff_feature_postquitafter48_prequit, q=.125)],
                          'UB': [np.quantile(exp_diff_feature_postquitwithin48_postquitafter48, q=.975), np.quantile(exp_diff_feature_postquitwithin48_prequit, q=.975), np.quantile(exp_diff_feature_postquitafter48_prequit, q=.975)]}

diff_summary_expscale = pd.DataFrame(diff_summary_expscale)
diff_summary_expscale.index = ['exp_diff_feature_postquitwithin48_postquitafter48','exp_diff_feature_postquitwithin48_prequit','exp_diff_feature_postquitafter48_prequit']
diff_summary_expscale


# %%
# Remove variable from workspace
del model, posterior_samples, model_summary_logscale


# %%


