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


def smoking_episode(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    bz2_marker = 'PUFFMARKER_SMOKING_EPISODE+PHONE.csv.bz2'
    zip_matching = [s for s in zip_namelist if bz2_marker in s]
    bz2_file = participant_zip.open(zip_matching[0])
    newfile = bz2.decompress(bz2_file.read())
    newfile = newfile.replace("\r", "")
    newfile = newfile.replace("\n", ",")
    newfile = newfile.split(",")
    newfile.pop()
    df = pd.DataFrame(np.array(newfile).reshape(-1, 3),
                      columns=['time', 'offset', 'event'])
    df['id'] = participant_id    
    print(df)
    save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
    save_filename = 'puffMarker-episode.csv'
    if os.path.isfile(save_dir + save_filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    temp_csv_file = open(save_dir+save_filename, append_write)
    df.to_csv(temp_csv_file, header=False)
    temp_csv_file.close()
    print('Added to file!')


def strip_random_ema_json(json_data):
    data = []
    # 0: smoked?, 1: when smoke,
    # 2: eaten, 3: when eaten,
    # 4: drink, 5: when drink
    # 7: urge, 8: cheerful, 9:happy, 10:angry,
    # 11: stressed, 12: sad, 13:see/smell,
    # 14: access, 15: smoking location,
    htm_questions = range(0, 6)
    ratings_questions = range(7, 16)
    data.extend([json_data['status'].encode('utf-8')])
    if (json_data['status'] == 'COMPLETED' or
        json_data['status'] == 'ABANDONED_BY_TIMEOUT'):
        for i in htm_questions:
            if json_data['question_answers'][i]['response'] is None:
                data.extend(['None'])
            else:
                data.extend([json_data['question_answers'][i]['response'][0].encode('utf-8')])                    
        for i in ratings_questions:
            if json_data['question_answers'][i]['response'] is None:
                data.extend('None')
            else:                    
                data.extend([(to_likert(json_data['question_answers'][i]['response'][0])).encode('utf-8')])                    
    else:
        data.extend(['NA'] * (len(htm_questions) + len(ratings_questions)) )
    return data


def random_ema(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    bz2_marker = 'EMA+RANDOM_EMA+PHONE.csv.bz2'
    zip_matching = [s for s in zip_namelist if bz2_marker in s]
    bz2_file = participant_zip.open(zip_matching[0])
    tempfile = bz2.decompress(bz2_file.read())

    tempfile = tempfile.rstrip('\n').split('\r')
    tempfile.pop()
    df = pd.DataFrame(np.array(newfile).reshape(-1, 3),
                      columns=['time', 'offset', 'event'])

    ts_list = []
    offset_list = []
    json_list = []
    for line in tempfile:
        line = line.replace("\n", "")
        ts, offset, values = line.rstrip().split(',', 2)
        ts_list.append(ts)
        offset_list.append(offset)
        values = values.replace("\'", "")
        json_data = json.loads(values)
        stripped_json = strip_random_ema_json(json_data)
        json_list.append(stripped_json)

    ts_df = pd.DataFrame(ts_list, columns=['time'])
    offset_df = pd.DataFrame(ts_list, columns=['offset'])
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
    json_df['day_of_week'] =  json_df['timestamp'].apply(day_of_week)
    
    save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
    save_filename = 'random-ema.csv'
    if os.path.isfile(save_dir + save_filename):
        append_write = 'a' # append if already exists
        header_binary = False
    else:
        append_write = 'w' # make a new file if not
        header_binary = True
    temp_csv_file = open(save_dir+save_filename, append_write)
    json_df.to_csv(temp_csv_file, header=header_binary, index=False)
    temp_csv_file.close()
    print('Added to file!')
