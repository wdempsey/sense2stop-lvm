import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import os.path
import bz2

def unix_date(intime):
	try:
	    return (datetime.fromtimestamp(int(intime)/
	                                   1000).strftime('%Y-%m-%d %H:%M:%S'))
	except:
		return None

def end_minus_start(end, start):
	try:
		return (int(end) / 1000 - int(start) / 1000) / 3600
	except:
		return None

def delete_duplicates(df, hours):
	'''
	delete 'duplicate' timestamps defined as two timestamps within the hours parameter
	'''
	if df.shape[0] == 0:
		return df, []

	days = [0]
	timestamps = [df.iloc[0][0]] # store timestamps
	for index, row in df.iterrows():
		if index == 0:
			prev_timestamp = row[0]
			continue
		current_timestamp = row[0]

		# require timestamp difference to be greater hours
		if abs(int(current_timestamp) / 1000 - int(prev_timestamp) / 1000) / 3600 > hours:
			days.append(index)
			prev_timestamp = row[0] # only reset prev if the current one is good
			timestamps.append(row[0])
	return df.ix[days], timestamps

def line_up(start_stamps, end_stamps, participant_id):
	'''
	line up the timestamps
	'''
	i = 0
	j = 0

	# ignore the bug in the dataset
	if participant_id == 224 or participant_id == 225:
		i = 1

	starts = []
	ends = [] 
	while i < len(start_stamps) and j < len(end_stamps):
		start = int(start_stamps[i]) / 1000
		end = int(end_stamps[j]) / 1000
		if end < start:
			j += 1
			continue
		elif end - start > 15 * 3600:
			i += 1
			continue
		else:
			starts.append(i)
			ends.append(j)
			i += 1
			j += 1
	return starts, ends

def process_alignment(participant_zip, participant_id, back_up=False):
	'''
	Inputs: zipfile, participant_id
	Output: add to csv the start time of a participant
	'''

	# extract files
	if not back_up:
		zip_namelist = participant_zip.namelist()
		start_marker = 'DAY_START+PHONE.csv.bz2'
		end_marker = 'DAY_END+PHONE.csv.bz2'
		start_matching = [s for s in zip_namelist if start_marker in s]
		end_matching  = [s for s in zip_namelist if end_marker in s]

		start_file = participant_zip.open(start_matching[0])
		end_file = participant_zip.open(end_matching[0])

		day_start = pd.read_csv(start_file, compression='bz2', header=None, 
			usecols = [0], names = ['start_timestamp'])
		day_end = pd.read_csv(end_file, compression='bz2', header=None, 
			usecols = [0], names = ['end_timestamp'])
	if back_up:
		zip_namelist = participant_zip.namelist()
		start_marker = 'null_DAY_START_null'
		end_marker = 'null_DAY_END_null'

		start_matching = [s for s in zip_namelist if start_marker in s]
		end_matching  = [s for s in zip_namelist if end_marker in s]
		
		start_file = participant_zip.open(start_matching[1])
		end_file = participant_zip.open(end_matching[1])
		day_start = pd.read_csv(start_file, header=None, 
			usecols = [0], names = ['start_timestamp'])
		day_end = pd.read_csv(end_file, header=None, 
			usecols = [0], names = ['end_timestamp'])
		print(day_start, day_end)

	# delete duplicates 
	day_start, start_stamps = delete_duplicates(day_start, 12)
	day_end, end_stamps = delete_duplicates(day_end, 12)

	# line up the start and the end timestamps
	starts, ends = line_up(start_stamps, end_stamps, participant_id)
	day_start = day_start.iloc[starts].reset_index(drop=True)
	day_end = day_end.iloc[ends].reset_index(drop=True)

	df = pd.concat([day_start, day_end], axis=1)

	# create dataframe
	try:
		df['participant_id'] = participant_id
		df['start_time'] = df['start_timestamp'].apply(unix_date)
		df['end_time'] = df['end_timestamp'].apply(unix_date)
		df['alignment'] = df.apply(lambda x: end_minus_start(x['end_timestamp'], x['start_timestamp']), axis=1)
		df = df[df.alignment != None] # delete extra days 
		print(df)
	except ValueError as e:
		print(e)
		print('{} has error and program exited prematurely!'.format(participant_id))
		return

	# write to the file
	save_dir = '/Users/jasonma/dropbox/research/sense2stop-lvm/empirical_priors/'
	save_filename = 'alignment_user.csv'
	if os.path.isfile(save_dir + save_filename):
		append_write = 'a'  # append if already exists
		header_binary = False
	else:
		append_write = 'w'  # make a new file if not
		header_binary = True
	temp_csv_file = open(save_dir+save_filename, append_write)
	df.to_csv(temp_csv_file, header=header_binary, index=False)
	temp_csv_file.close()
	print('Added to allignment file!')
	return None

def compute_alignment():
	return