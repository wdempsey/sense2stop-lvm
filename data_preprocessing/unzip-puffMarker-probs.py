import pandas as pd
import json, ast, csv
from datetime import datetime
import os
import os.path
from bz2 import BZ2File as bzopen
import glob

dir = '' #Insert the directory name where the bz2 file is kept
for fname in glob.glob(dir + '*PUFFMARKER_FEATURE_VECTOR+*.csv.bz2'): #The reg_ex will select all the bz2 files containing the word PUFFMARKER_FEATURE_VECTOR
	with bzopen(fname, 'r') as csvfile:
		for l in csvfile:
			time, offset, col1, col2, col3 = l.rstrip().split(',') #This line splits each line by ','. The first column is time, next is offset,
                                                                   #rest are other columns in the file which can be parsed as col1, col2 and so on.			