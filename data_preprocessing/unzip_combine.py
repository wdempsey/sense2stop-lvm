import glob
from zipfile import ZipFile
import pandas as pd
from unzip_functions import *

# Insert the directory name where the bz2 file is kept
dir = "../data-streams/"

participant_dates = pd.read_csv("../cleaned-data/participant-dates-v2.csv")

# No cloud data for 255-270
if python_version == 3:
	all_participant_ids = list(range(201, 223)) + list(range(228,254))
else:
	all_participant_ids = range(201, 223) + range(228,254)

for participant_id in all_participant_ids:
	print('Now on participant ' + str(participant_id))
	# File name we are dealing with
	zip_name = '/'+str(participant_id)+'*.zip'
	file_name = glob.glob(dir + zip_name)
	participant_zip = ZipFile(file_name[0])
	# Add the HQ Windows to a CSV file
	study_days(participant_zip, participant_id, participant_dates)
	# Add to cStress episodes
	stress_episodes(participant_zip, participant_id)
	# Add to cStress
	cstress(participant_zip, participant_id)
	# Add to puffMarker probability
	smoking_episode(participant_zip, participant_id)
	# Add to puffMaker episode
	puff_probability(participant_zip, participant_id)
	# Add to Random EMA
	random_ema(participant_zip, participant_id)
	# Add to End-of-Day EMA
	end_of_day_ema(participant_zip, participant_id)
	# Add to Event-Contingent
	event_contingent_ema(participant_zip, participant_id)
	# Add to self report
	self_report_smoking(participant_zip, participant_id)