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

# %%
exec(open('../env_vars.py').read())
dir_data = os.environ['dir_data']
dir_picklejar = os.environ['dir_picklejar']
dir_code_methods = os.environ['dir_code_methods']

# %%
filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_knitted')
infile = open(filename,'rb')
dict_knitted = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_puffmarker')
infile = open(filename,'rb')
dict_puffmarker = pickle.load(infile)
infile.close()

filename = os.path.join(os.path.realpath(dir_picklejar), 'dict_eod_survey')
infile = open(filename,'rb')
dict_eod_survey = pickle.load(infile)
infile.close()

# %%
# Differing conclusions
current_participant = None
current_day = None

pm_times = dict_puffmarker[current_participant][current_day]['hours_since_start_day'].to_list()
eod_times = dict_eod_survey[current_participant][current_day]['ticked_box_scaled']
random_ema_pufftime = dict_knitted[current_participant][current_day][(dict_knitted[current_participant][current_day]['assessment_type']=='random_ema') & (dict_knitted[current_participant][current_day]['smoke']=='Yes')]['puff_time'].to_list()
sr_ema_pufftime = dict_knitted[current_participant][current_day][dict_knitted[current_participant][current_day]['assessment_type']=='selfreport']['puff_time'].to_list()
random_ema_nopuff = dict_knitted[current_participant][current_day][(dict_knitted[current_participant][current_day]['assessment_type']=='random_ema') & (dict_knitted[current_participant][current_day]['smoke']=='No')]['hours_since_start_day']

print(pm_times)
print(eod_times)
print(sr_ema_pufftime)
print(random_ema_pufftime)
print(random_ema_nopuff)

fig1 = plt.figure(facecolor='white')
ax1 = plt.axes(frameon=True)
ax1.axes.get_yaxis().set_visible(False)
plt.ylim(bottom=-5, top=+5)

plt.scatter(eod_times, np.repeat(0, len(eod_times)), s=50, label="End-of-Day EMA")
plt.scatter(sr_ema_pufftime, np.repeat(-1, len(sr_ema_pufftime)), s=50, label="SR")
plt.scatter(pm_times, np.repeat(-2, len(pm_times)), s=50, label="Puffmarker")
plt.scatter(random_ema_pufftime, np.repeat(-3, len(random_ema_pufftime)), s=50, label="Random EMA (Yes)")
plt.scatter(random_ema_nopuff, np.repeat(-4, len(random_ema_nopuff)), s=50, label="Random EMA (No)")

plt.xlabel("Hours Elapsed Since Start of Day")
plt.legend(loc='upper left')
plt.show()

# %%
