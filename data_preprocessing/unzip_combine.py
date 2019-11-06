import os; os.chdir("C:/Users/wdem/Documents/GitHub/sense2stop-lvm/data_preprocessing")
from unzip_combine_functions import *
import sys

global_dir = "../cleaned-data/"
python_version = int(sys.version[0])

if python_version == 3:
	all_participant_ids = list(range(201, 271))
else:
	all_participant_ids = range(201, 271)

# Combine the HQ Windows to a CSV file
combine_study_days('hq-episodes', all_participant_ids)
# Combine puffMarker probability
combine_smoking_episode('puff-episode', all_participant_ids)
# Combine puffMaker episode
combine_puff_probability('puff-probability', all_participant_ids)
# Combine Random EMA
combine_random_ema('random-ema', all_participant_ids)
# Combine End-of-Day EMA
combine_end_of_day_ema('eod-ema', all_participant_ids)
# Combine Event-Contingent
combine_event_contingent_ema('eventcontingent-ema', all_participant_ids)
# Combine self report
combine_self_report_smoking('self-report-smoking', all_participant_ids)