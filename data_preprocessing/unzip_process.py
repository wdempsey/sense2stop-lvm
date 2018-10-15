import glob
import zipfile
from unzip_functions import *

# Insert the directory name where the bz2 file is kept
dir = '/Users/walterdempsey/Box/MD2K Processed Data/Data Streams'

all_participant_ids = range(201, 223) + range(228,238)

for participant_id in all_participant_ids:
        print('Now on participant ' + str(participant_id))
        # File name we are dealing with
        zip_name = '/'+str(participant_id)+'*.zip'
        file_name = glob.glob(dir + zip_name)
        participant_zip = zipfile.ZipFile(file_name[0])
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
