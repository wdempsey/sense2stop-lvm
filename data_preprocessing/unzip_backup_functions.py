import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import os.path
from helpers import *

# global_dir = "/Volumes/dav/MD2K Processed Data/smoking-lvm-cleaned-data/"
global_dir = "/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/"

def smoking_episode(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    csv_marker = 'PUFFMARKER_SMOKING_EPISODE'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    if csv_matching == []:
        print("No PUFFMARKER_SMOKING_EPISODE data for participant " + str(participant_id))
        return

    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('Empty file for smoking episode')
    else:
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 2),
                          columns=['timestamp', 'event'])
        df['participant_id'] = participant_id
        df['date'] = df['timestamp'].apply(unix_date)
        df['hour'] = df['timestamp'].apply(hour_of_day)
        df['minute'] = df['timestamp'].apply(minute_of_day)
        df['day_of_week'] =  df['timestamp'].apply(day_of_week)
        save_dir = global_dir
        save_filename = 'puff-episode-backup.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a'  # append if already exists
            header_binary = False
        else:
            append_write = 'w'  # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        df.to_csv(temp_csv_file, header=header_binary, index=False)
        temp_csv_file.close()
        print('Added to episode file!')
        return None

def puff_probability(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    csv_marker = 'PUFF_PROBABILITY' 
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    if csv_matching == []:
        print("No PUFF_PROBABILITY data for participant " + str(participant_id))
        return
    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('Empty file for smoking episode')
    else:
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 2),
                          columns=['timestamp', 'event'])
        df['participant_id'] = participant_id
        df['date'] = df['timestamp'].apply(unix_date)
        df['hour'] = df['timestamp'].apply(hour_of_day)
        df['minute'] = df['timestamp'].apply(minute_of_day)
        df['day_of_week'] =  df['timestamp'].apply(day_of_week)
        save_dir = global_dir
        save_filename = 'puff-probability-backup.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a'  # append if already exists
            header_binary = False
        else:
            append_write = 'w'  # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        df.to_csv(temp_csv_file, header=header_binary, index=False)
        temp_csv_file.close()
        print('Added to puff probability file!')

def random_ema(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    csv_marker = "RANDOM_EMA"
    zip_matching = [s for s in zip_namelist if csv_marker in s]
    zip_matching = [s for s in zip_matching if 'csv' in s]
    if not zip_matching:
        print("No RANDOM_EMA for participant " + str(participant_id))
        return      
    tempfile = participant_zip.open(zip_matching[0])
    tempfile = tempfile.readlines()
    ts_list = []
    json_list = []
    for line in tempfile:
        line = line.replace("\n", "")
        ts, values = line.rstrip().split(',', 1)
        ts_list.append(ts)
        #values = values.replace("\'", "")
        json_data = json.loads(values)
        stripped_json = strip_random_ema_json(json_data)
        json_list.append(stripped_json)

    json_df = pd.DataFrame(json_list,
                           columns=['status', 'smoke', 'when_smoke', 'eat',
                                    'when_eat', 'drink', 'when_drink',
                                    'urge', 'cheerful', 'happy', 'angry',
                                    'stress', 'sad', 'see_or_smell',
                                    'access', 'smoking_location'])
    json_df['participant_id'] = participant_id
    json_df['timestamp'] = ts_list
    json_df['date'] = json_df['timestamp'].apply(unix_date)
    json_df['hour'] = json_df['timestamp'].apply(hour_of_day)
    json_df['minute'] = json_df['timestamp'].apply(minute_of_day)
    json_df['day_of_week'] = json_df['timestamp'].apply(day_of_week)
    save_dir = global_dir
    save_filename = 'random-ema-backup.csv'
    if os.path.isfile(save_dir + save_filename):
        append_write = 'a' # append if already exists
        header_binary = False
    else:
        append_write = 'w' # make a new file if not
        header_binary = True
    temp_csv_file = open(save_dir+save_filename, append_write)
    json_df.to_csv(temp_csv_file, header=header_binary, index=False)
    temp_csv_file.close()
    print('Added to random ema file!')

def end_of_day_ema(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    csv_marker = 'END_OF_DAY_EMA'
    zip_matching = [s for s in zip_namelist if csv_marker in s]
    zip_matching = [s for s in zip_matching if 'csv' in s]
    if not zip_matching:
        print("No END_OF_DAY_EMA for participant " + str(participant_id))
        return None
    else:
        csv_file = participant_zip.open(zip_matching[0])
        tempfile = csv_file.readlines()
        ts_list = []
        json_list = []
        for line in tempfile:
            line = line.replace("\n", "")
            ts, values = line.rstrip().split(',', 1)
            ts_list.append(ts)
            json_data = json.loads(values)
            stripped_json = strip_end_of_day_ema_json(json_data)
            json_list.append(stripped_json)

        json_df = pd.DataFrame(json_list,
                               columns=['status', '8to9', '9to10', '10to11',
                                        '11to12', '12to13', '13to14',
                                        '14to15', '15to16', '16to17', '17to18',
                                        '18to19', '19to20'])
        json_df['participant_id'] = participant_id
        json_df['timestamp'] = ts_list
        json_df['date'] = json_df['timestamp'].apply(unix_date)
        json_df['hour'] = json_df['timestamp'].apply(hour_of_day)
        json_df['minute'] = json_df['timestamp'].apply(minute_of_day)
        json_df['day_of_week'] = json_df['timestamp'].apply(day_of_week)
        save_dir = global_dir
        save_filename = 'eod-ema-backup.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a' # append if already exists
            header_binary = False
        else:
            append_write = 'w' # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        json_df.to_csv(temp_csv_file, header=header_binary, index=False)
        temp_csv_file.close()
        print('Added to end of day ema file!')
        return None

def event_contingent_ema(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    csv_marker = 'SMOKING_EMA'
    zip_matching = [s for s in zip_namelist if csv_marker in s]
    zip_matching = [s for s in zip_matching if 'csv' in s]
    if not zip_matching:
        print("No SMOKING_EMA for participant " + str(participant_id))
        return 
    else: 
        csv_file = participant_zip.open(zip_matching[0])
        tempfile = csv_file.readlines()
        ts_list = []
        json_list = []
        for line in tempfile:
            line = line.replace("\n", "")
            ts, values = line.rstrip().split(',', 1)
            ts_list.append(ts)
            json_data = json.loads(values)
            stripped_json = strip_event_contingent_json(json_data)
            json_list.append(stripped_json)

        json_df = pd.DataFrame(json_list,
                               columns=['status', 'smoke', 'when_smoke',
                                        'urge', 'cheerful', 'happy',
                                        'angry', 'stress', 'sad',
                                        'see_or_smell',
                                        'access', 'smoking_location'])
        json_df['participant_id'] = participant_id
        json_df['timestamp'] = ts_list
        json_df['date'] = json_df['timestamp'].apply(unix_date)
        json_df['hour'] = json_df['timestamp'].apply(hour_of_day)
        json_df['minute'] = json_df['timestamp'].apply(minute_of_day)
        json_df['day_of_week'] = json_df['timestamp'].apply(day_of_week)
        save_dir = global_dir
        save_filename = 'eventcontingent-ema-backup.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a' # append if already exists
            header_binary = False
        else:
            append_write = 'w' # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        json_df.to_csv(temp_csv_file, header=header_binary, index=False)
        temp_csv_file.close()
        print('Added to event contingent ema file!')
        return None