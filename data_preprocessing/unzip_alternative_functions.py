import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import json
import os
import os.path
import pytz
import sys

global_dir = "../cleaned-data/"
python_version = int(sys.version[0])

def unix_to_datetime(intime):
    local_tz = pytz.timezone('US/Central')
    date = datetime.fromtimestamp(int(intime)/1000, local_tz)
    return date

def unix_date(intime):
    local_tz = pytz.timezone('US/Central')
    date = datetime.fromtimestamp(int(intime)/1000, local_tz)
    return date.strftime('%Y-%m-%d %H:%M:%S')

def unix_date_gmt(intime):
    global_tz = pytz.timezone('GMT')
    date = datetime.fromtimestamp(int(intime)/1000, global_tz)
    return date.strftime('%Y-%m-%d %H:%M:%S')

def hour_of_day(intime):
    local_tz = pytz.timezone('US/Central')
    date = datetime.fromtimestamp(int(intime)/1000, local_tz)
    return date.strftime("%H")


def minute_of_day(intime):
    local_tz = pytz.timezone('US/Central')
    date = datetime.fromtimestamp(int(intime)/1000, local_tz)
    return date.strftime("%M")


def day_of_week(intime):
    local_tz = pytz.timezone('US/Central')
    date = datetime.fromtimestamp(int(intime)/1000, local_tz)
    return date.strftime("%A")


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


def date_of_month(intime):
    local_tz = pytz.timezone('US/Central')
    date = datetime.fromtimestamp(int(intime)/1000, local_tz)
    return date.strftime('%Y-%m-%d')


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
        save_dir = global_dir
        save_filename = 'puff-episode-alternative.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a'  # append if already exists
            header_binary = False
        else:
            append_write = 'w'  # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        df.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
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
        save_dir = global_dir
        save_filename = 'puff-probability-alternative.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a'  # append if already exists
            header_binary = False
        else:
            append_write = 'w'  # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        df.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
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
    if python_version == 3:
        data.extend([json_data['status']])
    else:
        data.extend([json_data['status'].encode('utf-8')])
    if (json_data['status'] == 'COMPLETED' or
        json_data['status'] == 'ABANDONED_BY_TIMEOUT'):
        for i in htm_questions:
            if json_data['question_answers'][i]['response'] == []:
                data.extend(['None'])
            else:
                if python_version == 3:
                    data.extend([json_data['question_answers'][i]['response'][0]])
                else:
                    data.extend([json_data['question_answers'][i]['response'][0].encode('utf-8')])
        for i in ratings_questions:
            if json_data['question_answers'][i]['response'] == []:
                data.extend(['None'])
            else:
                if python_version == 3:
                    data.extend([(to_likert(json_data['question_answers'][i]['response'][0]))])
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
        if python_version == 3:
            line = line.decode('utf-8')
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
    save_filename = 'random-ema-alternative.csv'
    if os.path.isfile(save_dir + save_filename):
        append_write = 'a' # append if already exists
        header_binary = False
    else:
        append_write = 'w' # make a new file if not
        header_binary = True
    temp_csv_file = open(save_dir+save_filename, append_write)
    json_df.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
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
    if python_version == 3:
        binary_outcome.extend([json_data['status']])
    else:
        if python_version == 3:
            binary_outcome.extend([json_data['status']])
        else:
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
            if python_version == 3:
                line = line.decode('utf-8')
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
        save_filename = 'eod-ema-alternative.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a' # append if already exists
            header_binary = False
        else:
            append_write = 'w' # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        json_df.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
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
            if python_version == 3:
                line = line.decode('utf-8')
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
        save_filename = 'eventcontingent-ema-alternative.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a' # append if already exists
            header_binary = False
        else:
            append_write = 'w' # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        json_df.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
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
    if python_version == 3:
        data.extend([json_data['status']])
    else:
        data.extend([json_data['status'].encode('utf-8')])
    if (json_data['status'] == 'COMPLETED' or
        json_data['status'] == 'ABANDONED_BY_TIMEOUT'):
        for i in htm_questions:
            if json_data['question_answers'][i]['response'] is None:
                data.extend(['None'])
            else:
                if python_version == 3:
                    data.extend([json_data['question_answers'][i]['response'][0]])
                else:
                    data.extend([json_data['question_answers'][i]['response'][0].encode('utf-8')])
        for i in ratings_questions:
            if json_data['question_answers'][i]['response'] is None:
                data.extend(['None'])
            else:
                if python_version == 3:
                    data.extend([(to_likert(json_data['question_answers'][i]['response'][0]))])
                else:
                    data.extend([(to_likert(json_data['question_answers'][i]['response'][0])).encode('utf-8')])
    else:
        data.extend(['NA'] * (len(htm_questions) + len(ratings_questions)) )
    return data

def self_report_smoking(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    csv_marker = 'SELF_REPORT_SMOKING'
    zip_matching = [s for s in zip_namelist if csv_marker in s]
    zip_matching = [s for s in zip_matching if 'csv' in s]
    if not zip_matching:
        print("No smoking self report")
        return None
    else:
        csv_file = participant_zip.open(zip_matching[0])
        tempfile = csv_file.readlines()
        ts_list = []
        json_list = []
        for line in tempfile:
            if python_version == 3:
                line = line.decode('utf-8')
            line = line.replace("\n", "")
            ts, values = line.rstrip().split(',', 1)
            ts_list.append(ts)
            json_data = json.loads(values)
            stripped_json = strip_self_report_smoking_json(json_data)
            json_list.append(stripped_json)
        json_df = pd.DataFrame(json_list, columns=['message'])
        json_df['participant_id'] = participant_id
        json_df['timestamp'] = ts_list
        json_df['date'] = json_df['timestamp'].apply(unix_date)
        json_df['hour'] = json_df['timestamp'].apply(hour_of_day)
        json_df['minute'] = json_df['timestamp'].apply(minute_of_day)
        json_df['day_of_week'] = json_df['timestamp'].apply(day_of_week)
        save_dir = global_dir
        save_filename = 'self-report-smoking-alternative.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a' # append if already exists
            header_binary = False
        else:
            append_write = 'w' # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        json_df.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
        temp_csv_file.close()
        print('Added to smoking self report file!')
        return None

def strip_self_report_smoking_json(json_data):
    data = []
    if python_version == 3:
        data.extend([json_data['message']])
    else:
        if python_version == 3:
            data.extend([json_data['message']])
        else:
            data.extend([json_data['message'].encode('utf-8')])
    return data

def stress_episodes(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    csv_marker = 'CSTRESS_STRESS_EPISODE_ARRAY_CLASSIFICATION_FULL_EPISODE'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('Empty file for episode classification')
    else:
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 5),
                          columns=['timestamp', 'start_ts', 'peak_ts', 'end_ts', 'episode_label'])
        df['participant_id'] = participant_id
        df['start_date'] = df['start_ts'].apply(unix_date)
        df['peak_date'] = df['peak_ts'].apply(unix_date)
        df['end_date'] = df['end_ts'].apply(unix_date)
        save_dir = global_dir
        save_filename = 'stress-episodes-alternative.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a'  # append if already exists
            header_binary = False
        else:
            append_write = 'w'  # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        df.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
        temp_csv_file.close()
        print('Added to stress episodes file!')


def wakeup(participant_zip, participant_id):
    zip_namelist = participant_zip.namelist()
    csv_marker = 'WAKEUP'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('No Wakeup file')
    else:
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 2),
                          columns=['timestamp', 'date'])
        local_tz = pytz.timezone('US/Central')
        global_tz = pytz.timezone('GMT')
        wakeup_ts_list = []
        wakeup_date_list = []
        for iter in range(df.shape[0]):
            wakeup_ts_list.append(datetime.fromtimestamp(int(df['timestamp'][iter])/1000, local_tz))
            wakeup_date_list.append(datetime.fromtimestamp(int(df['date'][iter])/1000, global_tz))
        return wakeup_ts_list, wakeup_date_list

def sleep(participant_zip, participant_id):
    zip_namelist = participant_zip.namelist()
    csv_marker = 'SLEEP'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('No SLEEP file')
    else:
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 2),
                          columns=['timestamp', 'date'])
        local_tz = pytz.timezone('US/Central')
        global_tz = pytz.timezone('GMT')
        sleep_ts_list = []
        sleep_date_list = []
        for iter in range(df.shape[0]):
            sleep_ts_list.append(datetime.fromtimestamp(int(df['timestamp'][iter])/1000, local_tz))
            sleep_date_list.append(datetime.fromtimestamp(int(df['date'][iter])/1000, global_tz))
        return sleep_ts_list, sleep_date_list

def daystart(participant_zip, participant_id):
    zip_namelist = participant_zip.namelist()
    csv_marker = 'DAY_START'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('No Day Start file')
    else:
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 2),
                          columns=['timestamp', 'date'])
        local_tz = pytz.timezone('US/Central')
        global_tz = pytz.timezone('GMT')
        daystart_ts_list = []
        daystart_date_list = []
        for iter in range(df.shape[0]):
            daystart_ts_list.append(datetime.fromtimestamp(int(df['timestamp'][iter])/1000, local_tz))
            daystart_date_list.append(datetime.fromtimestamp(int(df['date'][iter])/1000, global_tz))
        return daystart_ts_list, daystart_date_list


def dayend(participant_zip, participant_id):
    zip_namelist = participant_zip.namelist()
    csv_marker = 'DAY_END'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('No Day END file')
    else:
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 2),
                          columns=['timestamp', 'date'])
        local_tz = pytz.timezone('US/Central')
        global_tz = pytz.timezone('GMT')
        dayend_ts_list = []
        dayend_date_list = []
        for iter in range(df.shape[0]):
            dayend_ts_list.append(datetime.fromtimestamp(int(df['timestamp'][iter])/1000, local_tz))
            dayend_date_list.append(datetime.fromtimestamp(int(df['date'][iter])/1000, global_tz))
        return dayend_ts_list, dayend_date_list

def leftwrist_dq(participant_zip, participant_id):
    zip_namelist = participant_zip.namelist()
    csv_marker = 'ACCELEROMETER_DATA_QUALITY_LEFT_WRIST_MOTION_SENSE'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('No left wrist file')
        return None
    else:
        local_tz = pytz.timezone('US/Central')
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 2),
                          columns=['timestamp', 'values'])
        leftwrist_dq_list = []
        df_leftwrist = df[df['values'] == 0]['timestamp']
        df_leftwrist = df_leftwrist.reset_index()
        for iter in range(df_leftwrist.shape[0]):
            leftwrist_dq_list.append(datetime.fromtimestamp(df_leftwrist['timestamp'][iter]/1000, local_tz))
        return leftwrist_dq_list


def rightwrist_dq(participant_zip, participant_id):
    zip_namelist = participant_zip.namelist()
    csv_marker = 'ACCELEROMETER_DATA_QUALITY_RIGHT_WRIST_MOTION_SENSE'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('No right wrist file')
        return None
    else:
        local_tz = pytz.timezone('US/Central')
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 2),
                          columns=['timestamp', 'values'])
        rightwrist_dq_list = []
        df_rightwrist = df[df['values'] == 0]['timestamp']
        df_rightwrist = df_rightwrist.reset_index()
        for iter in range(df_rightwrist.shape[0]):
            rightwrist_dq_list.append(datetime.fromtimestamp(df_rightwrist['timestamp'][iter]/1000, local_tz))
        return rightwrist_dq_list

def respiration_dq(participant_zip, participant_id):
    zip_namelist = participant_zip.namelist()
    csv_marker = 'RESPIRATION_DATA_QUALITY_CHEST_AUTOSENSE_CHEST'
    csv_matching = [s for s in zip_namelist if csv_marker in s]
    csv_matching = [s for s in csv_matching if '.csv' in s]
    csv_file = participant_zip.open(csv_matching[0])
    temp = csv_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        print ('No chest data file')
        return None
    else:
        local_tz = pytz.timezone('US/Central')
        csv_file = participant_zip.open(csv_matching[0])
        newfile = pd.read_csv(csv_file, header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 2),
                          columns=['timestamp', 'values'])
        respiration_dq_list = []
        df_respiration = df[df['values'] == 0]['timestamp']
        df_respiration = df_respiration.reset_index()
        for iter in range(df_respiration.shape[0]):
            respiration_dq_list.append(datetime.fromtimestamp(df_respiration['timestamp'][iter]/1000, local_tz))
        return respiration_dq_list


def currentday_startend(current_date, daystart_ts_list,
                        wakeup_ts_list, wakeup_date_list,
                        dayend_ts_list,
                        sleep_ts_list, sleep_date_list):
    # CHECK IF DAYSTART/DAYEND EXIST FOR THAT DAY
    # ELSE USE WAKEUP/SLEEP
    # IF WAKEUP/SLEEP EMPTY, USE MOST RECENT DAY
    start_time = -1
    for daystart in daystart_ts_list:
        if daystart.date() == current_date.date():
            start_time = daystart
    if start_time == -1:
        start_time = current_date
        if len(wakeup_ts_list) == 0:
            min_gap = 1000
            for daystart in daystart_ts_list:
                current_gap = abs((daystart - start_time).days)
                if current_gap < min_gap:
                    wakeup_hour = int(daystart.strftime('%H'))
                    wakeup_minute = int(daystart.strftime('%M'))
                    start_time = start_time.replace(hour=wakeup_hour,minute=wakeup_minute)
                    min_gap = current_gap
        else:
            start_time = current_date
            for iter in range(0,len(wakeup_ts_list)):
                wakeup_ts = wakeup_ts_list[iter]
                if current_date.date() >= wakeup_ts.date():
                    wakeup_hour = int(wakeup_date_list[iter].strftime('%H'))
                    wakeup_minute = int(wakeup_date_list[iter].strftime('%M'))
                    start_time = start_time.replace(hour=wakeup_hour,minute=wakeup_minute)
    end_time = - 1
    for dayend in dayend_ts_list:
        if dayend.date() == current_date.date():
            end_time = dayend
    if end_time == -1:
        end_time = current_date
        if len(sleep_ts_list) == 0:
            min_gap = 1000
            for dayend in sleep_ts_list:
                current_gap = abs((dayend - end_time).days)
                if current_gap < min_gap:
                    sleep_hour = int(dayend.strftime('%H'))
                    sleep_minute = int(dayend.strftime('%M'))
                    end_time = end_time.replace(hour=sleep_hour,minute=sleep_minute)
                    min_gap = current_gap
        else:
            end_time = current_date
            for iter in range(0,len(sleep_ts_list)):
                sleep_ts = sleep_ts_list[iter]
                if current_date.date() >= sleep_ts.date():
                    sleep_hour = int(sleep_date_list[iter].strftime('%H'))
                    sleep_minute = int(sleep_date_list[iter].strftime('%M'))
                    end_time = end_time.replace(hour=sleep_hour,minute=sleep_minute)
    return start_time, end_time

def leftwrist_day(start_time, end_time, leftwrist_ts_list):
    ## LEFT WRIST
    lw_jumpstart_list = [start_time]
    lw_jumpend_list = []
    first = 0
    any_good_measures = False
    for iter in range(0,len(leftwrist_ts_list)-1):
        lw_ts = leftwrist_ts_list[iter]
        next_lw_ts = leftwrist_ts_list[iter+1]
        if start_time.date() == lw_ts.date() and start_time <= lw_ts and end_time >= lw_ts:
            any_good_measures = True
            if first == 0:
                #print "We hit the first of times"
                diff = lw_ts - start_time
                if diff.seconds > 30.:
                    lw_jumpstart_list[0] = lw_ts
                    #print "And it's a gap"
                    #print start_time, lw_ts
                first = 1
            diff = next_lw_ts - lw_ts
            if diff.seconds > 30.:
                lw_jumpend_list.append(lw_ts)
                if next_lw_ts <= end_time:
                    lw_jumpstart_list.append(next_lw_ts)
    ## END WITH end_time if final window has
    ## good data until that time
    if len(lw_jumpend_list) < len(lw_jumpstart_list):
        lw_jumpend_list.append(end_time)
    else:
        lw_jumpstart_list.append(lw_jumpend_list[len(lw_jumpend_list)-1])
        lw_jumpend_list.append(end_time)
    ## CHECK IF START == END Of Interval and toss if true
    single_point_list = []
    for i in range(len(lw_jumpstart_list)):
        if lw_jumpstart_list[i] == lw_jumpend_list[i]:
            single_point_list.append(i)
    lw_jumpstart_list = np.array(lw_jumpstart_list)
    lw_jumpend_list = np.array(lw_jumpend_list)
    lw_start = np.delete(lw_jumpstart_list, single_point_list)
    lw_end = np.delete(lw_jumpend_list, single_point_list)
    if any_good_measures == True:
        return lw_start, lw_end
    else:
        return [], []

def rightwrist_day(start_time, end_time, rightwrist_ts_list):
    rw_jumpstart_list = [start_time]
    rw_jumpend_list = []
    first = 0
    any_good_measures = False
    for iter in range(0,len(rightwrist_ts_list)-1):
        rw_ts = rightwrist_ts_list[iter]
        next_rw_ts = rightwrist_ts_list[iter+1]
        if start_time.date() == rw_ts.date() and start_time <= rw_ts and end_time >= rw_ts:
            any_good_measures = True
            if first == 0:
                #print "Hit that first time"
                diff = rw_ts - start_time
                if diff.seconds > 30.:
                    rw_jumpstart_list[0] = rw_ts
                    #print "And there's a gap"
                    #print start_time, rw_ts
                first = 1
            diff = next_rw_ts - rw_ts
            if diff.seconds > 30.:
                rw_jumpend_list.append(rw_ts)
                if next_rw_ts <= end_time:
                    rw_jumpstart_list.append(next_rw_ts)
                #print rw_ts, next_rw_ts
    if len(rw_jumpend_list) < len(rw_jumpstart_list):
        rw_jumpend_list.append(end_time)
    else:
        rw_jumpstart_list.append(rw_jumpend_list[len(rw_jumpend_list)-1])
        rw_jumpend_list.append(end_time)
    # CHECK IF
    single_point_list = []
    for i in range(len(rw_jumpstart_list)):
        if rw_jumpstart_list[i] == rw_jumpend_list[i]:
            single_point_list.append(i)
    rw_jumpstart_list = np.array(rw_jumpstart_list)
    rw_jumpend_list = np.array(rw_jumpend_list)
    rw_start = np.delete(rw_jumpstart_list, single_point_list)
    rw_end = np.delete(rw_jumpend_list, single_point_list)
    if any_good_measures == True:
        return rw_start, rw_end
    else:
        return [], []


def respiration_day(start_time, end_time, respiration_ts_list):
    respiration_jumpstart_list = [start_time]
    respiration_jumpend_list = []
    first = 0
    any_good_measures = False
    for iter in range(0, len(respiration_ts_list)-1):
        resp_ts = respiration_ts_list[iter]
        next_resp_ts = respiration_ts_list[iter+1]
        if start_time.date() == resp_ts.date() and start_time <= resp_ts and end_time >= resp_ts:
            any_good_measures = True
            if first == 0:
                #print "At first time"
                diff = resp_ts - start_time
                if diff.seconds > 30.:
                    #print "And there's a gap"
                    respiration_jumpstart_list[0] = resp_ts
                    #print start_time, resp_ts
                first = 1
            diff = next_resp_ts - resp_ts
            if diff.seconds > 30.:
                respiration_jumpend_list.append(resp_ts)
                if next_resp_ts <= end_time:
                    respiration_jumpstart_list.append(next_resp_ts)
                #print resp_ts, next_resp_ts
    if len(respiration_jumpend_list) < len(respiration_jumpstart_list):
        respiration_jumpend_list.append(end_time)
    else:
        respiration_jumpstart_list.append(respiration_jumpend_list[len(respiration_jumpend_list)-1])
        respiration_jumpend_list.append(end_time)
    # CHECK IF START == END Of Interval and toss if true
    single_point_list = []
    for i in range(len(respiration_jumpstart_list)):
        if respiration_jumpstart_list[i] == respiration_jumpend_list[i]:
            single_point_list.append(i)
    respiration_jumpstart_list = np.array(respiration_jumpstart_list)
    respiration_jumpend_list = np.array(respiration_jumpend_list)
    respiration_start = np.delete(respiration_jumpstart_list, single_point_list)
    respiration_end = np.delete(respiration_jumpend_list, single_point_list)
    if any_good_measures == True:
        return respiration_start, respiration_end
    else:
        return [], []

def wrist_union(lw_start, lw_end, rw_start, rw_end):
    if lw_start == [] and rw_start == []:
        return [], []
    else:
        lrw_start_list = []
        lrw_end_list = []
        union_complete = False
        jointwrist_start_list = np.concatenate((lw_start, rw_start))
        jointwrist_end_list = np.concatenate((lw_end, rw_end))
        max_iter = len(jointwrist_start_list)
        iter = 0
        while not union_complete:
            iter +=1
            min_start = min(jointwrist_start_list)
            whichmin = [i for i in range(0,len(jointwrist_start_list)) if jointwrist_start_list[i] == min_start]
            max_end = max(jointwrist_end_list[whichmin])
            move_on = False
            #print iter
            while not move_on:
                #print ("Entered move on sub loop")
                which_in_interval = [i for i in range(0,len(jointwrist_start_list)) if jointwrist_start_list[i] > min_start and jointwrist_start_list[i] <= max_end]
                if len(which_in_interval) == 0:
                    #print("None in interval, move on")
                    move_on = True
                else:
                    if max(jointwrist_end_list[which_in_interval]) > max_end:
                        max_end = max(jointwrist_end_list[which_in_interval])
                    else:
                        #print ("Already got max_end correct, move on")
                        move_on = True
            lrw_start_list.append(min_start)
            lrw_end_list.append(max_end)
            keep_obs = jointwrist_start_list > max(lrw_end_list)
            jointwrist_end_list = jointwrist_end_list[keep_obs]
            jointwrist_start_list = jointwrist_start_list[keep_obs]
            if len(jointwrist_start_list) == 0 or iter > max_iter:
                union_complete = True
        return lrw_start_list, lrw_end_list


def wrist_chest_intersection(lrw_start_list, lrw_end_list, respiration_start, respiration_end, end_time):
    # COMBINE TO GENERATE THE INTERVALS.
    # FIRST, TAKE UNION OF LW/RW
    # NEXT, TAKE INTERSECTION OF UNION WITH RESP LISTS
    # wpc stands for wrist plus chest
    if lrw_start_list == [] or respiration_start == []:
        return [], []
    else:
        wpc_start_list = []
        wpc_end_list = []
        intersection_complete = False
        jointwrist_plus_resp_start_list = np.concatenate((lrw_start_list, respiration_start))
        jointwrist_plus_resp_end_list = np.concatenate((lrw_end_list, respiration_end))
        max_iter = len(jointwrist_plus_resp_start_list)*10
        iter = 0
        min_start = min(jointwrist_plus_resp_start_list)
        while not intersection_complete:
            iter +=1
            whichmin = [i for i in range(0,len(jointwrist_plus_resp_start_list)) if jointwrist_plus_resp_start_list[i] == min_start]
            max_end = max(jointwrist_plus_resp_end_list[whichmin])
            #print iter
            #print "Entered if/then part"
            which_in_interval = [i for i in range(0,len(jointwrist_plus_resp_start_list)) if jointwrist_plus_resp_start_list[i] <= max_end and jointwrist_plus_resp_end_list[i] >= min_start]
            if len(which_in_interval) == 0:
                #print "None in interval, move on"
                whichmin = [i for i in range(0,len(jointwrist_plus_resp_start_list)) if jointwrist_plus_resp_start_list[i] >= max_end]
                min_start = min(jointwrist_plus_resp_start_list[whichmin])
            else:
                #print "Something in interval"
                temp_start = jointwrist_plus_resp_start_list[which_in_interval]
                temp_end = jointwrist_plus_resp_end_list[which_in_interval]
                if len(temp_start[temp_start > min_start]) == 0:
                    min_temp_start = end_time
                else:
                    min_temp_start = min(temp_start[temp_start > min_start])
                # Check if min is start or end time
                if min_temp_start >= min(temp_end):
                    # If min is end, then this is an intersection
                    # window and append to lists!
                    max_end = min(temp_end)
                    #print min_start, max_end
                    wpc_start_list.append(min_start)
                    wpc_end_list.append(max_end)
                    #print "Finished appending"
                    if min_start == max_end:
                        whichmin = [i for i in range(0,len(jointwrist_plus_resp_start_list)) if jointwrist_plus_resp_start_list[i] > max_end]
                    else:
                        whichmin = [i for i in range(0,len(jointwrist_plus_resp_start_list)) if jointwrist_plus_resp_start_list[i] >= max_end]
                    if len(whichmin) == 0:
                        #print "There's nothing left!"
                        min_start = end_time
                    else:
                        min_start = min(jointwrist_plus_resp_start_list[whichmin])
                else:
                    # If min is start, then move
                    # up the start time and move on
                    #print "No appending, move on"
                    min_start = min(temp_start[temp_start > min_start])
            if min_start == end_time or iter > max_iter:
                #print "Intersection complete!"
                intersection_complete = True
        return wpc_start_list, wpc_end_list

def study_days(participant_zip, participant_id, participant_dates):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    # Get wake,sleep, daystart, dayend
    wakeup_ts_list, wakeup_date_list = wakeup(participant_zip, participant_id)
    sleep_ts_list, sleep_date_list = sleep(participant_zip, participant_id)
    daystart_ts_list, daystart_date_list = daystart(participant_zip, participant_id)
    dayend_ts_list, dayend_date_list = dayend(participant_zip, participant_id)
    # Bring in Left, Right, and Resp for participant_id
    leftwrist_ts_list = leftwrist_dq(participant_zip, participant_id)
    rightwrist_ts_list = rightwrist_dq(participant_zip, participant_id)
    respiration_ts_list = respiration_dq(participant_zip, participant_id)
    # Get range of dates from entry to exit
    entry_date = participant_dates['start_date'][np.where(participant_dates['participant'] == participant_id)[0][0]]
    quit_date = participant_dates['quit_date'][np.where(participant_dates['participant'] == participant_id)[0][0]]
    end_date = participant_dates['expected_end_date'][np.where(participant_dates['participant'] == participant_id)[0][0]]
    # Convert to DATETIME object
    local_tz = pytz.timezone('US/Central')
    entry_date = datetime.strptime(entry_date, '%m/%d/%Y')
    quit_date = datetime.strptime(quit_date, '%m/%d/%Y')
    end_date = datetime.strptime(end_date, '%m/%d/%Y')
    entry_date = entry_date.replace(tzinfo=local_tz)
    quit_date = quit_date.replace(tzinfo=local_tz)
    end_date = end_date.replace(tzinfo=local_tz)
    # Setup iteration
    current_date = entry_date
    date_iter = timedelta(days=1)
    date_range_length = (end_date - entry_date).days + 1
    for iter in range(date_range_length):
        # Update current_date and start + end times
        current_date = entry_date + date_iter * iter
        print(current_date.date())
        start_time, end_time = currentday_startend(current_date, daystart_ts_list, wakeup_ts_list, wakeup_date_list, dayend_ts_list, sleep_ts_list, sleep_date_list)
        # Clean raw data for the current_date
        lw_start, lw_end = leftwrist_day(start_time, end_time, leftwrist_ts_list)
        rw_start, rw_end = rightwrist_day(start_time, end_time, rightwrist_ts_list)
        respiration_start, respiration_end = respiration_day(start_time, end_time, respiration_ts_list)
        # Take union of wrist data
        lrw_start_list, lrw_end_list = wrist_union(lw_start, lw_end, rw_start, rw_end)
        # Take intersection with respiration data
        joint_start_list, joint_end_list = wrist_chest_intersection(lrw_start_list, lrw_end_list, respiration_start, respiration_end, end_time)
        if len(joint_start_list) == 0:
            print("Nothing on this day")
        ## Construct DF
        ## Participant id, date, iter, pre/post quit,
        ##
        partition_length = len(joint_start_list)
        if partition_length == 0:
            temp = {'id': [participant_id], 'date': [current_date.date()], 'study_day': [iter+1], 'prequit': [current_date.date() < quit_date.date()], 'hq_start': [-1], 'hq_end': [-1],  'start_time': [start_time], 'end_time': [end_time]}
        else:
            temp = {'id': np.repeat(participant_id, partition_length), 'date': np.repeat(current_date.date(), partition_length), 'study_day': np.repeat(iter+1, partition_length), 'prequit': np.repeat(current_date.date() >= quit_date.date(), partition_length), 'hq_start': joint_start_list, 'hq_end': joint_end_list, 'start_time': np.repeat(start_time, partition_length), 'end_time': np.repeat(end_time, partition_length)}
        df = pd.DataFrame(data = temp)
        save_dir = global_dir
        save_filename = 'hq-episodes-alternative.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a'  # append if already exists
            header_binary = False
        else:
            append_write = 'w'  # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        df.to_csv(temp_csv_file, header=header_binary, index=False, line_terminator = '\n')
        temp_csv_file.close()
        print('Added to hq-episode file!')
    return None