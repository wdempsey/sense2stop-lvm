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
    htm_questions = range(0,6)
    ratings_questions = range(8,16)
    data.extend([json_data['status'].encode('utf-8')])
    if (json_data['status'] == 'COMPLETED' or json_data['status'] == 'ABANDONED_BY_TIMEOUT'):
        for i in htm_questions:
            if json_data['question_answers'][i]['response'] == None:
                data.append('None')
            else:
                data.append([json_data['question_answers'][i]['response'][0].encode('utf-8')])                    
        for i in ratings_questions:
            if json_data['question_answers'][i]['response'] == None:
                data.append('None')
            else:                    
                data.append([(to_likert(json_data['question_answers'][i]['response'][0])).encode('utf-8')])                    
    else:
        data.append(['NA'] * (len(htm_questions) + len(ratings_questions)) )
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
    for line in tempfile:
        line = line.replace("\n", "")
        ts, offset, values = line.rstrip().split(',', 2)
        values = values.replace("\'","")
        json_data = json.loads(values)
        stripped_json = strip_random_ema_json(json_data)

        # 0: smoked?, 1: when smoke,
        # 2: eaten, 3: when eaten,
        # 4: drink, 5: when drink
        # 7: urge, 8: cheerful, 9:happy, 10:angry,
        # 11: stressed, 12: sad, 13:see/smell,
        # 14: access, 15: smoking location,
    newfile = newfile.split(",")

    
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



s = []
file = participant_zip.open(zip_matching[0],'r')
    for line in file.readlines():
        [ts, offset, jsondata] = line.decode().split(',', 2)
        a = re.sub('"{', "{", re.sub('}"', "}", jsondata))
        b = re.sub("u'(.*?)'", "\"\g<1>\"", a)
        c = re.sub("None", "\"None\"", b)
        d = re.sub("\"\"", "\\\"", c)

        j = json.loads(d[1:-3])
        s.append(j)
