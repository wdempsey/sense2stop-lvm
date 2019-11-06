import pandas as pd
import numpy as np
import sys
import os

global_dir = "../cleaned-data/"
python_version = int(sys.version[0])

def combine_smoking_episode(filename, all_participant_ids):
# Inputs: participant ids
	# Output: add to csv (prints when done)
	cloud_data = pd.read_csv(global_dir+filename+'.csv')
	backup_data = pd.read_csv(global_dir+filename+'-backup.csv')
	alternate_data = pd.read_csv(global_dir+filename+'-alternative.csv')
	for participant_id in all_participant_ids:
		print('On participant ' + str(participant_id))
		if any(backup_data['participant_id'] == participant_id):
			participant_data = backup_data[backup_data['participant_id'].isin([participant_id])]
		elif any(cloud_data['participant_id'] == participant_id):
			participant_data = cloud_data[cloud_data['participant_id'].isin([participant_id])]
			participant_data = participant_data.drop('offset', axis = 1)
		elif any(alternate_data['participant_id'] == participant_id):
			participant_data = alternate_data[alternate_data['participant_id'].isin([participant_id])]
		else:
			participant_data = None
			print ("No files contain that participant")
		if participant_data is not None:
			save_dir = global_dir
			save_filename = 'final/puff-episode-final.csv'
			if os.path.isfile(save_dir + save_filename):
				append_write = 'a'  # append if already exists
				header_binary = False
			else:
				append_write = 'w'  # make a new file if not
				header_binary = True
			temp_csv_file = open(save_dir+save_filename, append_write)
			participant_data.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
			temp_csv_file.close()
		print('Added to final puff episode file!')
	return None

def combine_puff_probability(filename, all_participant_ids):
	# Inputs: participant ids
	# Output: add to csv (prints when done)
	cloud_data = pd.read_csv(global_dir+filename+'.csv')
	backup_data = pd.read_csv(global_dir+filename+'-backup.csv')
	alternate_data = pd.read_csv(global_dir+filename+'-alternative.csv')
	for participant_id in all_participant_ids:
		print('On participant ' + str(participant_id))
		if any(backup_data['participant_id'] == participant_id):
			participant_data = backup_data[backup_data['participant_id'].isin([participant_id])]
		elif any(cloud_data['participant_id'] == participant_id):
			participant_data = cloud_data[cloud_data['participant_id'].isin([participant_id])]
			participant_data = participant_data.drop('offset', axis = 1)
		elif any(alternate_data['participant_id'] == participant_id):
			participant_data = alternate_data[alternate_data['participant_id'].isin([participant_id])]
		else:
			participant_data = None
			print ("No files contain that participant")
		if participant_data is not None:
			save_dir = global_dir
			save_filename = 'final/puff-probability-final.csv'
			if os.path.isfile(save_dir + save_filename):
				append_write = 'a'  # append if already exists
				header_binary = False
			else:
				append_write = 'w'  # make a new file if not
				header_binary = True
			temp_csv_file = open(save_dir+save_filename, append_write)
			participant_data.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
			temp_csv_file.close()
		print('Added to final puff-probability file!')
	return None

def combine_random_ema(filename, all_participant_ids):
	# Inputs: participant ids
	# Output: add to csv (prints when done)
	cloud_data = pd.read_csv(global_dir+filename+'.csv')
	backup_data = pd.read_csv(global_dir+filename+'-backup.csv')
	alternate_data = pd.read_csv(global_dir+filename+'-alternative.csv')
	for participant_id in all_participant_ids:
		print('On participant ' + str(participant_id))
		if any(backup_data['participant_id'] == participant_id):
			participant_data = backup_data[backup_data['participant_id'].isin([participant_id])]
		elif any(cloud_data['participant_id'] == participant_id):
			participant_data = cloud_data[cloud_data['participant_id'].isin([participant_id])]
			participant_data = participant_data.drop('offset', axis = 1)
		elif any(alternate_data['participant_id'] == participant_id):
			participant_data = alternate_data[alternate_data['participant_id'].isin([participant_id])]
		else:
			participant_data = None
			print ("No files contain that participant")
		if participant_data is not None:
			save_dir = global_dir
			save_filename = 'final/random-ema-final.csv'
			if os.path.isfile(save_dir + save_filename):
				append_write = 'a'  # append if already exists
				header_binary = False
			else:
				append_write = 'w'  # make a new file if not
				header_binary = True
			temp_csv_file = open(save_dir+save_filename, append_write)
			participant_data.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
			temp_csv_file.close()
		print('Added to final random-ema file!')
	return None

def combine_end_of_day_ema(filename, all_participant_ids):
	# Inputs: participant ids
	# Output: add to csv (prints when done)
	cloud_data = pd.read_csv(global_dir+filename+'.csv')
	backup_data = pd.read_csv(global_dir+filename+'-backup.csv')
	alternate_data = pd.read_csv(global_dir+filename+'-alternative.csv')
	for participant_id in all_participant_ids:
		print('On participant ' + str(participant_id))
		if any(backup_data['participant_id'] == participant_id):
			participant_data = backup_data[backup_data['participant_id'].isin([participant_id])]
		elif any(cloud_data['participant_id'] == participant_id):
			participant_data = cloud_data[cloud_data['participant_id'].isin([participant_id])]
			participant_data = participant_data.drop('offset', axis = 1)
		elif any(alternate_data['participant_id'] == participant_id):
			participant_data = alternate_data[alternate_data['participant_id'].isin([participant_id])]
		else:
			participant_data = None
			print ("No files contain that participant")
		if participant_data is not None:
			save_dir = global_dir
			save_filename = 'final/eod-ema-final.csv'
			if os.path.isfile(save_dir + save_filename):
				append_write = 'a'  # append if already exists
				header_binary = False
			else:
				append_write = 'w'  # make a new file if not
				header_binary = True
			temp_csv_file = open(save_dir+save_filename, append_write)
			participant_data.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
			temp_csv_file.close()
		print('Added to final eod-ema file!')
	return None


def combine_event_contingent_ema(filename, all_participant_ids):
	# Inputs: participant ids
	# Output: add to csv (prints when done)
	cloud_data = pd.read_csv(global_dir+filename+'.csv')
	backup_data = pd.read_csv(global_dir+filename+'-backup.csv')
	alternate_data = pd.read_csv(global_dir+filename+'-alternative.csv')
	for participant_id in all_participant_ids:
		print('On participant ' + str(participant_id))
		if any(backup_data['participant_id'] == participant_id):
			participant_data = backup_data[backup_data['participant_id'].isin([participant_id])]
		elif any(cloud_data['participant_id'] == participant_id):
			participant_data = cloud_data[cloud_data['participant_id'].isin([participant_id])]
			participant_data = participant_data.drop('offset', axis = 1)
		elif any(alternate_data['participant_id'] == participant_id):
			participant_data = alternate_data[alternate_data['participant_id'].isin([participant_id])]
		else:
			participant_data = None
			print ("No files contain that participant")
		if participant_data is not None:
			save_dir = global_dir
			save_filename = 'final/eventcontingent-ema-final.csv'
			if os.path.isfile(save_dir + save_filename):
				append_write = 'a'  # append if already exists
				header_binary = False
			else:
				append_write = 'w'  # make a new file if not
				header_binary = True
			temp_csv_file = open(save_dir+save_filename, append_write)
			participant_data.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
			temp_csv_file.close()
		print('Added to final eventcontingent-ema file!')
	return None

def combine_self_report_smoking(filename, all_participant_ids):
	# Inputs: participant ids
	# Output: add to csv (prints when done)
	cloud_data = pd.read_csv(global_dir+filename+'.csv')
	backup_data = pd.read_csv(global_dir+filename+'-backup.csv')
	alternate_data = pd.read_csv(global_dir+filename+'-alternative.csv')
	for participant_id in all_participant_ids:
		print('On participant ' + str(participant_id))
		if any(backup_data['participant_id'] == participant_id):
			participant_data = backup_data[backup_data['participant_id'].isin([participant_id])]
		elif any(cloud_data['participant_id'] == participant_id):
			participant_data = cloud_data[cloud_data['participant_id'].isin([participant_id])]
			participant_data = participant_data.drop('offset', axis = 1)
		elif any(alternate_data['participant_id'] == participant_id):
			participant_data = alternate_data[alternate_data['participant_id'].isin([participant_id])]
		else:
			participant_data = None
			print ("No files contain that participant")
		if participant_data is not None:
			save_dir = global_dir
			save_filename = 'final/self-report-smoking-final.csv'
			if os.path.isfile(save_dir + save_filename):
				append_write = 'a'  # append if already exists
				header_binary = False
			else:
				append_write = 'w'  # make a new file if not
				header_binary = True
			temp_csv_file = open(save_dir+save_filename, append_write)
			participant_data.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
			temp_csv_file.close()
		print('Added to final self-report-smoking file!')
	return None

def combine_study_days(filename, all_participant_ids):
	# Inputs: participant ids
	# Output: add to csv (prints when done)
	cloud_data = pd.read_csv(global_dir+filename+'.csv')
	backup_data = pd.read_csv(global_dir+filename+'-backup.csv')
	alternate_data = pd.read_csv(global_dir+filename+'-alternative.csv')
	for participant_id in all_participant_ids:
		print('On participant ' + str(participant_id))
		if any(backup_data['id'] == participant_id):
			participant_data = backup_data[backup_data['id'].isin([participant_id])]
		elif any(cloud_data['id'] == participant_id):
			participant_data = cloud_data[cloud_data['id'].isin([participant_id])]
		elif any(alternate_data['id'] == participant_id):
			participant_data = alternate_data[alternate_data['id'].isin([participant_id])]
		else:
			participant_data = None
			print ("No files contain that participant")
		if participant_data is not None:
			save_dir = global_dir
			save_filename = 'final/hq-episodes-final.csv'
			if os.path.isfile(save_dir + save_filename):
				append_write = 'a'  # append if already exists
				header_binary = False
			else:
				append_write = 'w'  # make a new file if not
				header_binary = True
			temp_csv_file = open(save_dir+save_filename, append_write)
			participant_data.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
			temp_csv_file.close()
		print('Added to final hq-episode file!')
	return None
