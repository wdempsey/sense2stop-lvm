import glob
from zipfile import ZipFile
from unzip_backup_functions import *

dir = '../data-streams-backup/Processed Phone Backup Files'

if python_version == 3:
	all_participant_ids = list(range(201, 223)) + list(range(228,271))
else:
	all_participant_ids = range(201, 223) + range(228,271)

participant_dates = pd.read_csv('../cleaned-data/participant-dates-v2.csv')

print('About to start the loop')

for participant_id in all_participant_ids:
	print('Now on participant ' + str(participant_id))
	zip_name = '/'+str(participant_id)+'*.zip'
	file_name = glob.glob(dir + zip_name)
	file_exists = False
	try:
		participant_zip = ZipFile(file_name[0])
		file_exists = True
	except:
		print('participant '+ str(participant_id) + ' does not have cloud back up data!')
	if not file_exists:
		continue
	study_days(participant_zip, participant_id, participant_dates)
	smoking_episode(participant_zip, participant_id)
	puff_probability(participant_zip, participant_id)
	end_of_day_ema(participant_zip, participant_id)
	random_ema(participant_zip, participant_id)
	event_contingent_ema(participant_zip, participant_id)
	self_report_smoking(participant_zip, participant_id)
