import pandas as pd
import numpy as np
import json, ast, csv
from datetime import datetime
import os
import os.path
import bz2

def unix_date(intime):
    return ( datetime.fromtimestamp(int(intime)/
                                   1000).strftime('%Y-%m-%d %H:%M:%S'))


def hour_of_day(intime):
    return (datetime.fromtimestamp(int(intime)/1000).strftime("%H"))


def minute_of_day(intime):
    return (datetime.fromtimestamp(int(intime)/1000).strftime("%M"))


def day_of_week(intime):
    return (datetime.fromtimestamp(int(intime)/1000).strftime("%A"))


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


def date_of_month(intime):
    return (datetime.fromtimestamp(int(intime)/1000).strftime('%Y-%m-%d'))

def to_likert(instring):
    if instring=="NO!!!":
        return '1'
    elif instring=="NO":
        return '2'
    elif instring=="no":
        return '3'
    elif instring=="No":
        return '3'
    elif instring=="Yes":
        return '4'
    elif instring=="YES":
        return '5'
    elif instring=="YES!!!":
        return '6'
    else:
        return instring


def day_week(inday):
    if inday == 'Monday':
        return 1
    elif inday == 'Tuesday':
        return 2
    elif inday == 'Wednesday':
        return 3
    elif inday == 'Thursday':
        return 4
    elif inday == 'Friday':
        return 5
    elif inday == 'Saturday':
        return 6
    else:
        return 7


def smoking_episode(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    bz2_marker = 'PUFFMARKER_SMOKING_EPISODE+PHONE.csv.bz2'
    zip_matching = [s for s in zip_namelist if bz2_marker in s]
    bz2_file = participant_zip.open(zip_matching[0])
    newfile = pd.read_csv(bz2_file, compression='bz2', header=None)
    df = pd.DataFrame(np.array(newfile).reshape(-1, 3),
                      columns=['timestamp', 'offset', 'event'])
    df['participant_id'] = participant_id
    df['date'] = df['timestamp'].apply(unix_date)
    df['hour'] = df['timestamp'].apply(hour_of_day)
    df['minute'] = df['timestamp'].apply(minute_of_day)
    df['day_of_week'] =  df['timestamp'].apply(day_of_week)
    print(df)
    save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
    save_filename = 'puffMarker-episode.csv'
    if os.path.isfile(save_dir + save_filename):
        append_write = 'a'  # append if already exists
        header_binary = False
    else:
        append_write = 'w'  # make a new file if not
        header_binary = True
    temp_csv_file = open(save_dir+save_filename, append_write)
    df.to_csv(temp_csv_file, header=header_binary, index=False)
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

    json_df = pd.DataFrame(json_list,
                           columns=['status', 'smoke', 'when_smoke', 'eat',
                                    'when_eat', 'drink', 'when_drink',
                                    'urge', 'cheerful', 'happy', 'angry',
                                    'stress', 'sad', 'see_or_smell',
                                    'access', 'smoking_location'])
    json_df['participant_id'] = participant_id
    json_df['timestamp'] = ts_list
    json_df['offset'] = offset_list
    json_df['date'] = json_df['timestamp'].apply(unix_date)
    json_df['hour'] = json_df['timestamp'].apply(hour_of_day)
    json_df['minute'] = json_df['timestamp'].apply(minute_of_day)
    json_df['day_of_week'] = json_df['timestamp'].apply(day_of_week)
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
