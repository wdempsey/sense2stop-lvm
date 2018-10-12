
# coding: utf-8

# In[ ]:

#Author: Soujanya Chatterjee

import pandas as pd
import json, ast, csv
from datetime import datetime
import os
import os.path
from bz2 import BZ2File as bzopen
import glob


def unix_date(intime):
    return (datetime.fromtimestamp(int(intime)/1000).strftime('%Y-%m-%d %H:%M:%S'))
def time_of_day(intime):
    return (datetime.fromtimestamp(int(intime)/1000).strftime("%H"))
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
    elif instring=="yes":
        return '4'
    elif instring=="YES":
        return '5'
    elif instring=="YES!!!":
        return '6'
    else:
        return instring
    
def day_week(inday):
    if inday=='Monday':
        return 1
    elif inday=='Tuesday':
        return 2
    elif inday=='Wednesday':
        return 3
    elif inday=='Thursday':
        return 4
    elif inday=='Friday':
        return 5
    elif inday=='Saturday':
        return 6
    else:
        return 7
for i in range(201,223,1):
    print (i)
    flag = 0
    header = []
    id = str(i)
	#Directory where the raw EMA files are kept and also where the processed EMA files are stored
    dir = 'C:\Soujanya\Estimation_of_First_Lapse\\NU_Data\%s\EMA' %id+'\\'
    if os.path.isfile(dir + 'EMA_Random_%s'%id+'.csv') == True:
        os.remove(dir + 'EMA_Random_%s'%id+'.csv')        
    data = []
    ema_input = []    
    time_start = []
    d_time_start = []
	#regex - *+RANDOM_EMA+PHONE for random EMA
	#regex - *+END_OF_DAY_EMA+PHONE for end of day EMA
	#regex - *+SMOKING_EMA+PHONE for smoking EMA
    for fname in glob.glob(dir + '*+RANDOM_EMA+PHONE.csv'):
        with open(fname, 'r') as csvfile:
            for l in csvfile:                
                ts, offset, values = l.rstrip().split(',', 2)        
                ema_input.append(values)
                ts = int(ts)*1000000 
    
	#Directory where the day_start file is kept
    dir_time = 'C:\Soujanya\Estimation_of_First_Lapse\\NU_Data\%s\Day' %id+'\\'
    for fname in glob.glob(dir_time + '*DAY_START+*.csv'):
        with open(fname, 'r') as csvfile:
            for l in csvfile:
                time, offset, time_ = l.rstrip().split(',')        
                time_start.append(time_)                    
    d_time_start.append(date_of_month(time_start[0]))   
    
    header1 = ['1','2','3','4','5','6','7','8']
    header2 = ['Participant', 'Id', 'Day', 'Status', 'Starttime', 'Date/Time', 'Time of Day', 'Day of Week']
    json_data = ema_input[2].strip("'\"")
    json_data = json.loads(json_data)    
    print(json_data)
    if (json_data['status'] == 'COMPLETED' or json_data['status'] == 'ABANDONED_BY_TIMEOUT'):
        for i in range(len(json_data['question_answers'])):
            header2.append(json_data['question_answers'][i]['question_text'].encode('ascii','ignore'))
            header2.append('Time')
            header1.append('q%d'%(i))
            header1.append(' ')            
    for i in range(len(ema_input)):
        json_data = ema_input[i].strip("'\"")
        json_data = json.loads(json_data)        
        u = date_of_month(json_data['start_timestamp'])         
        day = days_between(d_time_start[0], u)+1
        if (json_data['status'] == 'COMPLETED' or json_data['status'] == 'ABANDONED_BY_TIMEOUT'):  
            data.append(id)
            data.append(str(i+1))     
            data.append(day)
            data.extend([json_data['status'],json_data['start_timestamp'], unix_date(json_data['start_timestamp']),\
                        time_of_day(json_data['start_timestamp']), day_week(day_of_week(json_data['start_timestamp']))])
            for i in range(len(json_data['question_answers'])):
                if json_data['question_answers'][i]['response'] == None:
                    data.append('None')
                    data.append(json_data['question_answers'][i]['finish_time'])                                
                else:                    
                    data.append([(to_likert(json_data['question_answers'][i]['response'][0])).encode('utf-8')])                    
                    data.append(json_data['question_answers'][i]['finish_time'])
        else:
            data.append(id)
            data.append(str(i+1))
            data.append(day)
            data.extend([json_data['status'],json_data['start_timestamp'], unix_date(json_data['start_timestamp']), \
                        time_of_day(json_data['start_timestamp']), day_week(day_of_week(json_data['start_timestamp']))])
        
		#replace with the directory to store the respective EMA files
        with open(dir + 'EMA_Random_%s'%id+'.csv', 'a') as csvfile:
            ema_data = csv.writer(csvfile, delimiter=',')        
            if flag==0:       
                ema_data.writerow(header2)
            ema_data.writerow(data)
            flag+=1            
        data=[]        