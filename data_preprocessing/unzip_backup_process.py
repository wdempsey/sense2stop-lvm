import glob
import zipfile
from unzip_backup_functions import *

dir = './data-streams-backup/Processed Phone Backup Files'

all_participant_ids = range(201, 223) + range(228,254)

participant_dates = pd.read_csv('./cleaned-data/participant-dates.csv')

for participant_id in all_participant_ids:
        print('Now on participant ' + str(participant_id))
        # File name we are dealing with
        zip_name = '/'+str(participant_id)+'*.zip'
        file_name = glob.glob(dir + zip_name)
        file_exists = False
        try:
        	participant_zip = zipfile.ZipFile(file_name[0])
        	file_exists = True
        except:
        	print('participant '+ str(participant_id) + ' does not have cloud back up data!')
        if not file_exists:
        	continue
        # ADD HQ PARTITION TO BACKUP FILE
        study_days(participant_zip, participant_id, participant_dates)
        '''
    	smoking_episode(participant_zip, participant_id)

    	puff_probability(participant_zip, participant_id)

    	end_of_day_ema(participant_zip, participant_id)

    	random_ema(participant_zip, participant_id)

    	event_contingent_ema(participant_zip, participant_id)
        '''
