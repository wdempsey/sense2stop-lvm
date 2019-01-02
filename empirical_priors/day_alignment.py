import pandas as pd
import numpy as np
import datetime as datetime
import glob
import zipfile
import os
import os.path
import bz2
from alignment import *

# directory
# dir = "/Volumes/dav/MD2K Processed Data/Data Streams/"
dir = "/Volumes/dav/MD2K Processed Data/Data Streams/Participants 223-227"
# all_participant_ids = list(range(201,223))+list(range(228,238))
all_participant_ids = list(range(223, 228))
for participant_id in all_participant_ids:
        print('Now on participant ' + str(participant_id))
        # File name we are dealing with
        zip_name = '/'+str(participant_id)+'*.zip'
        file_name = glob.glob(dir + zip_name)
        participant_zip = zipfile.ZipFile(file_name[0])
        process_alignment(participant_zip, participant_id, True)
