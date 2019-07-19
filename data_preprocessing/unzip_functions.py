import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import json
import os
import os.path
import bz2
import pytz

def unix_date(intime):
    local_tz = pytz.timezone('US/Central')
    date = datetime.fromtimestamp(int(intime)/1000, local_tz)
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
    bz2_marker = 'PUFFMARKER_SMOKING_EPISODE+PHONE.csv.bz2'
    zip_matching = [s for s in zip_namelist if bz2_marker in s]
    bz2_file = participant_zip.open(zip_matching[0])
    temp = bz2_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        return None
    else:
        bz2_file = participant_zip.open(zip_matching[0])
        newfile = pd.read_csv(bz2_file, compression='bz2', header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 3),
                          columns=['timestamp', 'offset', 'event'])
        df['participant_id'] = participant_id
        df['date'] = df['timestamp'].apply(unix_date)
        df['hour'] = df['timestamp'].apply(hour_of_day)
        df['minute'] = df['timestamp'].apply(minute_of_day)
        df['day_of_week'] =  df['timestamp'].apply(day_of_week)
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'puff-episode.csv'
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
    bz2_marker = 'PUFF_PROBABILITY+PHONE.csv.bz2'
    zip_matching = [s for s in zip_namelist if bz2_marker in s]
    bz2_file = participant_zip.open(zip_matching[0])
    temp = bz2_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        return None
    else:
        bz2_file = participant_zip.open(zip_matching[0])
        newfile = pd.read_csv(bz2_file, compression='bz2', header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 3),
                      columns=['timestamp', 'offset', 'event'])
        df['participant_id'] = participant_id
        df['date'] = df['timestamp'].apply(unix_date)
        df['hour'] = df['timestamp'].apply(hour_of_day)
        df['minute'] = df['timestamp'].apply(minute_of_day)
        df['day_of_week'] =  df['timestamp'].apply(day_of_week)
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'puff-probability.csv'
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
            if json_data['question_answers'][i]['response'] is None:
                data.extend(['None'])
            else:
                data.extend([json_data['question_answers'][i]['response'][0].encode('utf-8')])
        for i in ratings_questions:
            if json_data['question_answers'][i]['response'] is None:
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
        if response is not None:
            for block in hourblocks:
                if block in response:
                    binary_outcome.extend(['1'])
                else:
                    binary_outcome.extend(['0'])
        else:
            binary_outcome.extend(['NA'] * len(hourblocks))
    else:
        binary_outcome.extend(['NA'] * len(hourblocks))
    return binary_outcome


def end_of_day_ema(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    bz2_marker = 'EMA+END_OF_DAY_EMA+PHONE.csv.bz2'
    zip_matching = [s for s in zip_namelist if bz2_marker in s]
    if not zip_matching:
        print("No end of day ema")
        return None
    else:
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
            stripped_json = strip_end_of_day_ema_json(json_data)
            json_list.append(stripped_json)

        json_df = pd.DataFrame(json_list,
                               columns=['status', '8to9', '9to10', '10to11',
                                        '11to12', '12to13', '13to14',
                                        '14to15', '15to16', '16to17', '17to18',
                                        '18to19', '19to20'])
        json_df['participant_id'] = participant_id
        json_df['timestamp'] = ts_list
        json_df['offset'] = offset_list
        json_df['date'] = json_df['timestamp'].apply(unix_date)
        json_df['hour'] = json_df['timestamp'].apply(hour_of_day)
        json_df['minute'] = json_df['timestamp'].apply(minute_of_day)
        json_df['day_of_week'] = json_df['timestamp'].apply(day_of_week)
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'eod-ema.csv'
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
    bz2_marker = 'EMA+SMOKING_EMA+PHONE.csv.bz2'
    zip_matching = [s for s in zip_namelist if bz2_marker in s]
    if not zip_matching:
        print("No event contingent ema")
        return None
    else:
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
        json_df['offset'] = offset_list
        json_df['date'] = json_df['timestamp'].apply(unix_date)
        json_df['hour'] = json_df['timestamp'].apply(hour_of_day)
        json_df['minute'] = json_df['timestamp'].apply(minute_of_day)
        json_df['day_of_week'] = json_df['timestamp'].apply(day_of_week)
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'eventcontingent-ema.csv'
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
                data.extend(['None'])
            else:
                data.extend([(to_likert(json_data['question_answers'][i]['response'][0])).encode('utf-8')])
    else:
        data.extend(['NA'] * (len(htm_questions) + len(ratings_questions)) )
    return data


def cstress(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    bz2_marker = 'STRESS_LABEL+PHONE.csv.bz2'
    zip_matching = [s for s in zip_namelist if bz2_marker in s]
    bz2_file = participant_zip.open(zip_matching[0])
    temp = bz2_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        return None
    else:
        bz2_file = participant_zip.open(zip_matching[0])
        newfile = pd.read_csv(bz2_file, compression='bz2', header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 3),
                          columns=['timestamp', 'offset', 'event'])
        df['participant_id'] = participant_id
        df['date'] = df['timestamp'].apply(unix_date)
        df['hour'] = df['timestamp'].apply(hour_of_day)
        df['minute'] = df['timestamp'].apply(minute_of_day)
        df['day_of_week'] =  df['timestamp'].apply(day_of_week)
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'stress-label.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a'  # append if already exists
            header_binary = False
        else:
            append_write = 'w'  # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        df.to_csv(temp_csv_file, header=header_binary, index=False)
        temp_csv_file.close()
        print('Added to stress label file!')
        return None

def stress_episodes(participant_zip, participant_id):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
    zip_namelist = participant_zip.namelist()
    bz2_marker = 'CSTRESS_STRESS_EPISODE_ARRAY_CLASSIFICATION_FULL_EPISODE+PHONE.csv.bz2'
    zip_matching = [s for s in zip_namelist if bz2_marker in s]
    bz2_file = participant_zip.open(zip_matching[0])
    temp = bz2_file.read()
    if not temp or temp == 'BZh9\x17rE8P\x90\x00\x00\x00\x00':
        return None
    else:
        bz2_file = participant_zip.open(zip_matching[0])
        newfile = pd.read_csv(bz2_file, compression='bz2', header=None)
        df = pd.DataFrame(np.array(newfile).reshape(-1, 6),
                          columns=['timestamp', 'offset', 'start_ts', 'peak_ts', 'end_ts', 'episode_label'])
        df['participant_id'] = participant_id
        df['start_date'] = df['start_ts'].apply(unix_date)
        df['peak_date'] = df['peak_ts'].apply(unix_date)
        df['end_date'] = df['end_ts'].apply(unix_date)
        save_dir = '/Users/walterdempsey/Box/MD2K Processed Data/smoking-lvm-cleaned-data/'
        save_filename = 'stress-episodes.csv'
        if os.path.isfile(save_dir + save_filename):
            append_write = 'a'  # append if already exists
            header_binary = False
        else:
            append_write = 'w'  # make a new file if not
            header_binary = True
        temp_csv_file = open(save_dir+save_filename, append_write)
        df.to_csv(temp_csv_file, header=header_binary, index=False)
        temp_csv_file.close()
        print('Added to stress label file!')
        return None


def study_days(participant_zip, participant_id, participant_dates):
    # Inputs: zipfile, participant_id
    # Output: add to csv (prints when done)
zip_namelist = participant_zip.namelist()
bz2_marker = 'WAKEUP+PHONE.csv.bz2'
zip_matching = [s for s in zip_namelist if bz2_marker in s]
if not zip_matching:
    print("No WAKE UP file")
    #return None
else:
    local_tz = pytz.timezone('US/Central')
    global_tz = pytz.timezone('GMT')
    bz2_file = participant_zip.open(zip_matching[0])
    tempfile = bz2.decompress(bz2_file.read())
    tempfile = tempfile.rstrip('\n').split('\r')
    tempfile.pop()
    wakeup_ts_list = []
    wakeup_date_list = []
    for line in tempfile:
        line = line.replace("\n", "")
        ts, offset, values = line.rstrip().split(',', 2)
        ts = datetime.fromtimestamp(int(ts)/1000, local_tz)
        wakeup_ts_list.append(ts)
        date = datetime.fromtimestamp(int(values)/1000, global_tz)
        wakeup_date_list.append(date)
bz2_marker = 'SLEEP+PHONE.csv.bz2'
zip_matching = [s for s in zip_namelist if bz2_marker in s]
if not zip_matching:
    print("No SLEEP file")
    #return None
else:
    local_tz = pytz.timezone('US/Central')
    global_tz = pytz.timezone('GMT')
    bz2_file = participant_zip.open(zip_matching[0])
    tempfile = bz2.decompress(bz2_file.read())
    tempfile = tempfile.rstrip('\n').split('\r')
    tempfile.pop()
    sleep_ts_list = []
    sleep_date_list = []
    for line in tempfile:
        line = line.replace("\n", "")
        ts, offset, values = line.rstrip().split(',', 2)
        ts = datetime.fromtimestamp(int(ts)/1000, local_tz)
        sleep_ts_list.append(ts)
        date = datetime.fromtimestamp(int(values)/1000, global_tz)
        sleep_date_list.append(date)
bz2_marker = 'DAY_START+PHONE.csv.bz2'
zip_matching = [s for s in zip_namelist if bz2_marker in s]
if not zip_matching:
    print("No DAY START file")
    #return None
else:
    global_tz = pytz.timezone('GMT')
    local_tz = pytz.timezone('US/Central')
    bz2_file = participant_zip.open(zip_matching[0])
    tempfile = bz2.decompress(bz2_file.read())
    tempfile = tempfile.rstrip('\n').split('\r')
    tempfile.pop()
    daystart_ts_list = []
    daystart_date_list = []
    for line in tempfile:
        line = line.replace("\n", "")
        ts, offset, values = line.rstrip().split(',', 2)
        ts = datetime.fromtimestamp(int(ts)/1000, local_tz)
        daystart_ts_list.append(ts)
        date = datetime.fromtimestamp(int(values)/1000, local_tz)
        daystart_date_list.append(date)
bz2_marker = 'DAY_END+PHONE.csv.bz2'
zip_matching = [s for s in zip_namelist if bz2_marker in s]
if not zip_matching:
    print("No DAY END file")
    #return None
else:
    local_tz = pytz.timezone('US/Central')
    bz2_file = participant_zip.open(zip_matching[0])
    tempfile = bz2.decompress(bz2_file.read())
    tempfile = tempfile.rstrip('\n').split('\r')
    tempfile.pop()
    dayend_ts_list = []
    dayend_date_list = []
    for line in tempfile:
        line = line.replace("\n", "")
        ts, offset, values = line.rstrip().split(',', 2)
        ts = datetime.fromtimestamp(int(ts)/1000, local_tz)
        dayend_ts_list.append(ts)
        date = datetime.fromtimestamp(int(values)/1000, local_tz)
        dayend_date_list.append(date)
bz2_marker = '+DATA_QUALITY+ACCELEROMETER+MOTION_SENSE+LEFT_WRIST.csv.bz2'
zip_matching = [s for s in zip_namelist if bz2_marker in s]
if not zip_matching:
    print("No LEFT WRIST file")
    #return None
else:
    local_tz = pytz.timezone('US/Central')
    bz2_file = participant_zip.open(zip_matching[0])
    tempfile = bz2.decompress(bz2_file.read())
    tempfile = tempfile.rstrip('\n').split('\r')
    tempfile.pop()
    leftwrist_ts_list = []
    for line in tempfile:
        line = line.replace("\n", "")
        ts, offset, values = line.rstrip().split(',', 2)
        if values == '3':
            ts = datetime.fromtimestamp(int(ts)/1000, local_tz)
            leftwrist_ts_list.append(ts)
bz2_marker = '+DATA_QUALITY+ACCELEROMETER+MOTION_SENSE+RIGHT_WRIST.csv.bz2'
zip_matching = [s for s in zip_namelist if bz2_marker in s]
if not zip_matching:
    print("No RIGHT WRIST file")
    #return None
else:
    local_tz = pytz.timezone('US/Central')
    bz2_file = participant_zip.open(zip_matching[0])
    tempfile = bz2.decompress(bz2_file.read())
    tempfile = tempfile.rstrip('\n').split('\r')
    tempfile.pop()
    rightwrist_ts_list = []
    for line in tempfile:
        line = line.replace("\n", "")
        ts, offset, values = line.rstrip().split(',', 2)
        if values == '3':
            ts = datetime.fromtimestamp(int(ts)/1000, local_tz)
            rightwrist_ts_list.append(ts)
bz2_marker = '+DATA_QUALITY+RESPIRATION+AUTOSENSE_CHEST+CHEST.csv.bz2'
zip_matching = [s for s in zip_namelist if bz2_marker in s]
if not zip_matching:
    print("No RESPIRATION file")
    #return None
else:
    local_tz = pytz.timezone('US/Central')
    bz2_file = participant_zip.open(zip_matching[0])
    tempfile = bz2.decompress(bz2_file.read())
    tempfile = tempfile.rstrip('\n').split('\r')
    tempfile.pop()
    respiration_ts_list = []
    for line in tempfile:
        line = line.replace("\n", "")
        ts, offset, values = line.rstrip().split(',', 2)
        if values == '3':
            ts = datetime.fromtimestamp(int(ts)/1000, local_tz)
            respiration_ts_list.append(ts)

entry_date = participant_dates['start_date'][np.where(participant_dates['participant'] == participant_id)[0][0]]
quit_date = participant_dates['quit_date'][np.where(participant_dates['participant'] == participant_id)[0][0]]
end_date = participant_dates['actual_end_date'][np.where(participant_dates['participant'] == participant_id)[0][0]]

entry_date = datetime.strptime(entry_date, '%m/%d/%y')
quit_date = datetime.strptime(quit_date, '%m/%d/%y')
end_date = datetime.strptime(end_date, '%m/%d/%y')

current_date = entry_date
date_iter = 1
# while (current_date <= end_date and iter < 1000):
# CHECK IF DAYSTART/DAYEND EXIST FOR THAT DAY
start_time = -1
for daystart in daystart_ts_list:
    if daystart.date() == current_date.date():
        start_time = daystart
if start_time == -1:
    start_time = current_date
    for iter in range(0,len(wakeup_ts_list)):
        wakeup_ts = wakeup_ts_list[iter]
        if current_date.date() <= wakeup_ts.date():
            wakeup_hour = int(wakeup_date_list[iter].strftime('%H'))
            wakeup_minute = int(wakeup_date_list[iter].strftime('%M'))
            start_time = start_time.replace(hour=wakeup_hour,minute=wakeup_minute)

end_time = - 1
for dayend in dayend_ts_list:
    if dayend.date() == current_date.date():
        end_time = dayend
if end_time == -1:
    end_time = current_date
    for iter in range(0,len(sleep_ts_list)):
        sleep_ts = sleep_ts_list[iter]
        if current_date.date() <= sleep_ts.date():
            sleep_hour = int(sleep_date_list[iter].strftime('%H'))
            sleep_minute = int(sleep_date_list[iter].strftime('%M'))
    end_time = end_time.replace(hour=sleep_hour,minute=sleep_minute)

## WITH START AND END TIMES DEFINED
## NEXT DEFINE HQ WINDOWS FOR L/R wrist
## AND RESPIRATION

## LEFT WRIST
lw_jumpstart_list = [start_time]
lw_jumpend_list = []
first = 0
for iter in range(0,len(leftwrist_ts_list)-1):
    lw_ts = leftwrist_ts_list[iter]
    next_lw_ts = leftwrist_ts_list[iter+1]
    if start_time.date() == lw_ts.date() and start_time <= lw_ts and end_time >= lw_ts:
        if first == 0:
            print "We hit the first of times"
            diff = lw_ts - start_time
            if diff.seconds > 30.:
                lw_jumpstart_list[0] = lw_ts
                print "And it's a gap"
                print start_time, lw_ts
            first = 1
        diff = next_lw_ts - lw_ts
        if diff.seconds > 30.:
            lw_jumpend_list.append(lw_ts)
            if next_lw_ts <= end_time:
                lw_jumpstart_list.append(next_lw_ts)
            print lw_ts, next_lw_ts
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


# RIGHT WRIST
rw_jumpstart_list = [start_time]
rw_jumpend_list = []
first = 0
for iter in range(0,len(rightwrist_ts_list)-1):
    rw_ts = rightwrist_ts_list[iter]
    next_rw_ts = rightwrist_ts_list[iter+1]
    if start_time.date() == rw_ts.date() and start_time <= rw_ts and end_time >= rw_ts:
        if first == 0:
            print "Hit that first time"
            diff = rw_ts - start_time
            if diff.seconds > 30.:
                rw_jumpstart_list[0] = rw_ts
                print "And there's a gap"
                print start_time, rw_ts
            first = 1
        diff = next_rw_ts - rw_ts
        if diff.seconds > 30.:
            rw_jumpend_list.append(rw_ts)
            if next_rw_ts <= end_time:
                rw_jumpstart_list.append(next_rw_ts)
            print rw_ts, next_rw_ts
if len(rw_jumpend_list) < len(rw_jumpstart_list):
    rw_jumpend_list.append(end_time)
else:
    rw_jumpstart_list.append(rw_jumpend_list[len(rw_jumpend_list)-1])
    rw_jumpend_list.append(end_time)
# CHECK IF START == END Of Interval and toss if true
single_point_list = []
for i in range(len(rw_jumpstart_list)):
    if rw_jumpstart_list[i] == rw_jumpend_list[i]:
        single_point_list.append(i)

rw_jumpstart_list = np.array(rw_jumpstart_list)
rw_jumpend_list = np.array(rw_jumpend_list)

rw_start = np.delete(rw_jumpstart_list, single_point_list)
rw_end = np.delete(rw_jumpend_list, single_point_list)

# RESPIRATION CHEST
respiration_jumpstart_list = [start_time]
respiration_jumpend_list = []
first = 0
for iter in range(0, len(respiration_ts_list)-1):
    resp_ts = respiration_ts_list[iter]
    next_resp_ts = respiration_ts_list[iter+1]
    if start_time.date() == resp_ts.date() and start_time <= resp_ts and end_time >= resp_ts:
        if first == 0:
            print "At first time"
            diff = resp_ts - start_time
            if diff.seconds > 30.:
                print "And there's a gap"
                respiration_jumpstart_list[0] = resp_ts
                print start_time, resp_ts
            first = 1
        diff = next_resp_ts - resp_ts
        if diff.seconds > 30.:
            respiration_jumpend_list.append(resp_ts)
            if next_resp_ts <= end_time:
                respiration_jumpstart_list.append(next_resp_ts)
            print resp_ts, next_resp_ts
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

# COMBINE TO GENERATE THE INTERVALS.
# FIRST, TAKE UNION OF LW/RW
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
    print iter
    #print "Currently we have %s intervals left" % len(jointwrist_start_list)
    while not move_on:
        print ("Entered move on sub loop")
        which_in_interval = [i for i in range(0,len(jointwrist_start_list)) if jointwrist_start_list[i] > min_start and jointwrist_start_list[i] <= max_end]
        if len(which_in_interval) == 0:
            print("None in interval, move on")
            move_on = True
        else:
            if max(jointwrist_end_list[which_in_interval]) > max_end:
                max_end = max(jointwrist_end_list[which_in_interval])
            else:
                print ("Already got max_end correct, move on")
                move_on = True
    lrw_start_list.append(min_start)
    lrw_end_list.append(max_end)
    keep_obs = jointwrist_start_list > max(lrw_end_list)
    jointwrist_end_list = jointwrist_end_list[keep_obs]
    jointwrist_start_list = jointwrist_start_list[keep_obs]
    if len(jointwrist_start_list) == 0 or iter > max_iter:
        union_complete = True

# NEXT, TAKE INTERSECTION OF UNION WITH RESP LISTS
# wpc stands for wrist plus chest
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
    move_on = False
    print iter
    while not move_on:
        print "Entered move on sub loop"
        which_in_interval = [i for i in range(0,len(jointwrist_plus_resp_start_list)) if jointwrist_plus_resp_start_list[i] <= max_end and jointwrist_plus_resp_end_list[i] >= min_start]
        if len(which_in_interval) == 0:
            print "None in interval, move on"
            whichmin = [i for i in range(0,len(jointwrist_plus_resp_start_list)) if jointwrist_plus_resp_start_list[i] >= max_end]
            min_start = min(jointwrist_plus_resp_start_list[whichmin])
            move_on = True
        else:
            print "Something in interval"
            temp_start = jointwrist_plus_resp_start_list[which_in_interval]
            temp_end = jointwrist_plus_resp_end_list[which_in_interval]
            # Check if min is start or end time
            if min(temp_start) > min(temp_end):
                # If min is end, then define
                # interval to that min and move on
                max_end = min(temp_end)
            else:
                # If min is start, then define
                # Interval from that point
                min_start = min(temp_start[temp_start >= min_start])
                whichcur_in_interval = [i for i in which_in_interval if jointwrist_plus_resp_start_list[i] == min_start]
                end_cur = max(jointwrist_plus_resp_end_list[whichcur_in_interval])
            if end_cur > max_end:
                print "Already got max_end of current interval "
            else:
                print ("Need new max_end")
                max_end = end_cur
            wpc_start_list.append(min_start)
            wpc_end_list.append(max_end)
            print "Finished appending"
            whichmin = [i for i in range(0,len(jointwrist_plus_resp_start_list)) if jointwrist_plus_resp_start_list[i] >= max_end]
            min_start = min(jointwrist_plus_resp_start_list[whichmin])
            move_on = True
            print "Let's move on"
    if min_start == max(jointwrist_plus_resp_start_list) or iter > max_iter:
        intersection_complete = True

for i in range(len(wpc_start_list)):
    print wpc_start_list[i], wpc_end_list[i]

for i in range(len(respiration_start)):
    print respiration_start[i], respiration_end[i]

for i in range(len(lrw_start_list)):
    print lrw_start_list[i], lrw_end_list[i]

## UPDATE TO NEXT DAY
current_date = current_date + timedelta(days=1)
iter +=1
print current_date
