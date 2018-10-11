import pandas as pd
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

# File name we are dealing with
file_name = glob.glob(dir + '/203*.zip')

participant_zip = zipfile.ZipFile(file_name[0])

zip_namelist = participant_zip.namelist()

puffMarker_marker = 'PUFF_PROBABILITY+PHONE.csv.bz2'

zip_matching = [s for s in zip_namelist if puffMarker_marker in s]

puffMarker_file = participant_zip.open(zip_matching[0])

newfile = bz2.decompress(puffMarker_file.read())

newfile = newfile.replace("\r", "")
newfile = newfile.replace("\n", ",")
newfile = newfile.split(",")
newfile.pop()



csvfile = pd.read_csv(newfile)


for fname in glob.glob(dir + '*PUFFMARKER_FEATURE_VECTOR+*.csv.bz2'): #The reg_ex will select all the bz2 files containing the word PUFFMARKER_FEATURE_VECTOR
	with bzopen(fname, 'r') as csvfile:
		for l in csvfile:
			time, offset, col1, col2, col3 = l.rstrip().split(',') #This line splits each line by ','. The first column is time, next is offset,
                                                                   #rest are other columns in the file which can be parsed as col1, col2 and so on.			
