import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import os.path

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
    elif instring=="yes":
        return '4'
    elif instring=="Yes":
        return '4'
    elif instring=="YES":
        return '5'
    elif instring=="YES!!!":
        return '6'
    elif instring is None:
        return 'None'
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
    csv_marker = 'PUFFMARKER_SMOKING_EPISODE'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
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
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'puff-episode-alternative.csv'
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
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'puff-probability-alternative.csv'
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
            if json_data['question_answers'][i]['response'] == []:
                data.extend(['None'])
            else:
                data.extend([json_data['question_answers'][i]['response'][0].encode('utf-8')])                    
        for i in ratings_questions:
            if json_data['question_answers'][i]['response'] == []:
                data.extend(['None'])
            else:                    
                data.extend([(to_likert(json_data['question_answers'][i]['response'][0])).encode('utf-8')])                    
    else:
        data.extend(['NA'] * (len(htm_questions) + len(ratings_questions)) )
    return data


def random_ema(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    csv_marker = "RANDOM_EMA"
    zip_matching = [s for s in zip_namelist if csv_marker in s]
    zip_matching = [s for s in zip_matching if 'csv' in s]
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
    save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
    save_filename = 'random-ema-alternative.csv'
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


def strip_end_of_day_ema_json(json_data):
    hourblocks = ['8:00 am - 9:00 am', '9:00 am - 10:00 am',
                  '10:00 am - 11:00 am', '11:00 am - 12:00 pm',
                  '12:00 pm - 1:00 pm', '1:00 pm - 2:00 pm',
                  '2:00 pm - 3:00 pm', '3:00 pm - 4:00 pm',
                  '4:00 pm - 5:00 pm', '5:00 pm - 6:00 pm',
                  '6:00 pm - 7:00 pm', '7:00 pm - 8:00 pm']
    binary_outcome = []
    binary_outcome.extend([json_data['status'].encode('utf-8')])
    if (json_data['status'] == 'COMPLETED' or
        json_data['status'] == 'ABANDONED_BY_TIMEOUT'):
        response = json_data['question_answers'][0]['response']
        for block in hourblocks:
            if block in response:
                binary_outcome.extend(['1'])
            else:
                binary_outcome.extend(['0'])
    else:
        binary_outcome.extend(['NA'] * len(hourblocks))
    return binary_outcome


def end_of_day_ema(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    csv_marker = 'END_OF_DAY_EMA'
    zip_matching = [s for s in zip_namelist if csv_marker in s]
    zip_matching = [s for s in zip_matching if 'csv' in s]
    if not zip_matching:
        print("No end of day ema")
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
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'eod-ema-alternative.csv'
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
        print("No event contingent ema")
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
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'eventcontingent-ema-alternative.csv'
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


def strip_event_contingent_json(json_data):
    data = []
    # 0: puffed?, 1: when puff,
    # 3: urge, 4: cheerful, 5: happy
    # 6: angry, 7: stress 8: sad
    # 9: see/smell, 10: access, 11: smoking_location
    htm_questions = range(0, 2)
    ratings_questions = range(3, 12)
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
