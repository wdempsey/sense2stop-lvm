import pandas as pd
import numpy as np
import json, ast, csv
from datetime import datetime
import os
import os.path
from bz2 import BZ2File as bzopen
import bz2
import glob
import zipfile

# Insert the directory name where the bz2 file is kept
dir = '/Users/walterdempsey/Box/MD2K Processed Data/Data Streams'

all_participant_ids = range(201,223,1) + range(228,238,1)

for participant_id in all_participant_ids:

        # File name we are dealing with
        zip_name = '/'+str(participant_id)+'*.zip'
        file_name = glob.glob(dir + zip_name)
        
        participant_zip = zipfile.ZipFile(file_name[0])
        
        zip_namelist = participant_zip.namelist()
        
        puffMarker_marker = 'PUFF_PROBABILITY+PHONE.csv.bz2'
        zip_matching = [s for s in zip_namelist if puffMarker_marker in s]
        
        puffMarker_file = participant_zip.open(zip_matching[0])
        
        newfile = pd.read_csv(puffMarker_file, compression = 'bz2', header = None)        
        df = pd.DataFrame(np.array(newfile),
                          columns=['time', 'offset', 'prob'])
        
        df['id'] = participant_id
        
        print(df)

        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'puffMarker-probability.csv'
        
        if os.path.isfile(save_dir + save_filename):
                append_write = 'a' # append if already exists
        else:
                append_write = 'w' # make a new file if not
                
        puffMarker = open(save_dir+save_filename, append_write)

        
        df.to_csv(puffMarker, header=False)

        puffMarker.close()
