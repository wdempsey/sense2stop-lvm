import glob
from zipfile import ZipFile
import pandas as pd
from unzip_alternative_functions import *

# Insert the directory name where the bz2 file is kept
dir = '../data-streams/Participants 223-227'

participant_dates = pd.read_csv('../cleaned-data/participant-dates-v2.csv')

if python_version == 3:
        all_participant_ids = list(range(223, 228))
else:
        all_participant_ids = range(223, 228)

for participant_id in all_participant_ids:
	print('Now on participant ' + str(participant_id))
	# File name we are dealing with
	zip_name = '/'+str(participant_id)+'*.zip'
	file_name = glob.glob(dir + zip_name)
	participant_zip = ZipFile(file_name[0])
	# Add the HQ Windows to a CSV file
	study_days(participant_zip, participant_id, participant_dates)
	# Add to puffMarker probability
	smoking_episode(participant_zip, participant_id)
	# # Add to puffMaker episode
	puff_probability(participant_zip, participant_id)
	# # Add to Random EMA
	random_ema(participant_zip, participant_id)
	# # # Add to End-of-Day EMA
	end_of_day_ema(participant_zip, participant_id)
	# # Add to Event-Contingent
	event_contingent_ema(participant_zip, participant_id)
	# Add to self report
	self_report_smoking(participant_zip, participant_id)