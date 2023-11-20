import pandas as pd
import numpy as np
import torch

class Pandemic_Data:
    def __init__(self, look_back, pred_len):
        self.x = np.empty((0,look_back), int)
        self.y = np.empty((0,pred_len), int)
        self.meta_data = np.array([])
    
    def get_sample(self,i):
        return self.x[i], self.y[i], self.meta_data[i]


def read_pandemic_ts (file_name, root_dir = None, look_back = 100, pred_len = 200, with_meta_data = False, 
                      moving_average = 1, data_smoothing = 'False', data_frequency = 'Daily'):

    pandemic_ts_file = pd.read_csv(root_dir + file_name, dtype={'Domain':str})

    data_type = pandemic_ts_file['type'][0]

    country_names = list(set(pandemic_ts_file['Country']))[0:2]

    data = Pandemic_Data(look_back=look_back, pred_len=pred_len)

    for country in country_names:

        processing_country_data = pandemic_ts_file[pandemic_ts_file['Country'] == country]
        
        processing_country_data = processing_country_data.reset_index(drop=True)

        ## Set the first day that case number exceed 100 as the start date
        start_idx = np.argmax(processing_country_data['number']>100)

        processing_country_data = processing_country_data[start_idx:]

        min_date = min(pd.to_datetime(processing_country_data['date']).dt.date)
        max_date = max(pd.to_datetime(processing_country_data['date']).dt.date)

        # print(processing_country_data)

        if (max_date - min_date).days < (look_back + pred_len):
            print(processing_country_data['Country'][0], processing_country_data['Domain'][0], processing_country_data['Sub-Domain'][0], "doesn't contain enough time span for desired look back length and prediction length.")
            continue

        look_back_data = processing_country_data['number'][:look_back]
        pred_data = processing_country_data['number'][look_back:look_back + pred_len]

        data.x = np.append(data.x, [look_back_data.to_list()], axis=0)
        data.y = np.append(data.y, [pred_data.to_list()], axis = 0)

    # print(data.x)

    if data_frequency == 'Daily':
        data.x = process_daily_data(data.x, moving_average)
    elif data_frequency == 'Weekly':
        data.x = process_weekly_data(data.x, data_smoothing)
    else:
        print("Data Frequency Type not supported, currently only accepting Daily and Weekly Data.")
        exit(1)

    print(data.x.shape)
    print(data.y.shape)

    return 0

def process_daily_data(ts, avg_len):
    
    processed_data = moving_average(ts, avg_len)

    return processed_data

def process_weekly_data(data, smoothing):

    if smoothing == 'False':
        processed_data = data_padding(data, replace_value='prev')
    else:
        processed_data= data_smoothing(data, method = 'linear')
    
    return processed_data
    
def moving_average(ts, avg_len = 7):

    moving_list = np.empty((0,len(ts[0])), int)

    for data in ts:

        moving_number = np.empty(len(data))

        for i in range(len(moving_number)):
            if i == 0:
                moving_number[i] = data[i]
            elif i < (avg_len - 1):
                moving_number[i] = round(sum(data[:i+1])/(i+1))
            else:
                moving_number[i] = round(sum(data[i-avg_len+1:i+1]/ avg_len))

        moving_list = np.append(moving_list,[moving_number],axis=0)

    return moving_list

### TODO
def data_smoothing(data, method = 'linear'):

    smooth_data = data

    return smooth_data

### TODO
def data_padding(data, replace_value = 'prev'):

    pad_data = data

    return pad_data



if __name__ == '__main__':
    read_pandemic_ts(root_dir='F:/Pandemic-Database/Processed_Data/Covid_19/',
                     file_name='Covid_World_Domain_Daily_CumCases.csv',
                     moving_average=7)