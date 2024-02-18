import pandas as pd
import numpy as np
import math

def process_daily_data(data, smoothing, look_back, pred_len, avg_len):
    
    if smoothing == 'False':
        look_back_data, pred_data, look_back_timestamp, pred_data_timestamp = data_padding(data,
                                                                                           look_back,
                                                                                           pred_len, 
                                                                                           replace_value='prev')
    else:
        look_back_data, pred_data, look_back_timestamp, pred_data_timestamp = data_smoothing(data, 
                                                                                             look_back,
                                                                                             pred_len,
                                                                                             method = 'linear')
    
    look_back_data = moving_average(look_back_data, avg_len)

    pred_data = moving_average(pred_data, avg_len)

    return look_back_data, pred_data, look_back_timestamp, pred_data_timestamp

def process_weekly_data(data, smoothing, look_back, pred_len):

    if smoothing == 'False':
        look_back_data, pred_data, look_back_timestamp, pred_data_timestamp = data_padding(data,
                                                                                           look_back,
                                                                                           pred_len, 
                                                                                           replace_value='prev')
    else:
        look_back_data, pred_data, look_back_timestamp, pred_data_timestamp = data_smoothing(data, 
                                                                                             look_back,
                                                                                             pred_len,
                                                                                             method = 'linear')
    
    return look_back_data, pred_data, look_back_timestamp, pred_data_timestamp
    
def moving_average(ts, avg_len = 7):

    moving_number = np.empty(len(ts))

    for i in range(len(moving_number)):
        if i == 0:
            moving_number[i] = ts[i]
        elif i < (avg_len - 1):
            moving_number[i] = round(sum(ts[:i+1])/(i+1))
        else:
            moving_number[i] = round(sum(ts[i-avg_len+1:i+1]/ avg_len))

    return moving_number

def data_smoothing(data, look_back, pred_len, method = 'linear'):

    data = data.drop_duplicates(subset=['date'], keep = 'first')

    min_date = min(data['date'])
    max_date = max(data['date'])

    ts_data = data['number']
    ts_data.index = data['date']
    ts_data.index.name = None

    idx = pd.date_range(min_date,max_date)

    ts_data.index = pd.DatetimeIndex(ts_data.index)

    ts_data = ts_data.reindex(idx)

    ts_data = ts_data.interpolate(method = method,
                                  limit_direction = "forward")
    
    ts_data = ts_data.round()

    look_back_data = ts_data.to_numpy()[:look_back]
    pred_data = ts_data.to_numpy()[look_back:look_back + pred_len]
    look_back_timestamp = ts_data.index.to_numpy()[:look_back]
    pred_data_timestamp = ts_data.index.to_numpy()[look_back:look_back + pred_len]

    return look_back_data, pred_data, look_back_timestamp, pred_data_timestamp

def data_padding(data, look_back, pred_len, replace_value = 'prev'):

    min_date = min(data['date'])
    max_date = max(data['date'])

    ts_data = data['number']
    ts_data.index = data['date']
    ts_data.index.name = None

    idx = pd.date_range(min_date,max_date)

    ts_data.index = pd.DatetimeIndex(ts_data.index)

    ts_data = ts_data.reindex(idx).ffill()

    look_back_data = ts_data.to_numpy()[:look_back]
    pred_data = ts_data.to_numpy()[look_back:look_back + pred_len]
    look_back_timestamp = ts_data.index.to_numpy()[:look_back]
    pred_data_timestamp = ts_data.index.to_numpy()[look_back:look_back + pred_len]

    return look_back_data, pred_data, look_back_timestamp, pred_data_timestamp

def get_meta_data(pandemic_name):
    feature_names = ['R0_Low','R0_Up','Object_Transmit','Blood_Transmit','Airborne',
                     'Vector-Borne','Droplets','Motality_Rate','Latent_Period_Low','Latent_Period_Up',
                     'Incubation_Period_Low', 'Incubation_Period_Up','Hosp_Low','Hosp_Up']

    pandemic_meta_data = {'Covid': [1.4,6.49,1,0,1,0,1,0.011,5.1,5.9,6.3, 7.5, 13.47,17.23],
                          'Ebola': [2.47,6.3, 1,1,0,1,1,0.5066,9.42,15.12, 4.65, 7.79, 12.8,27.96],
                          'SARS': [2.2,3.6,1,0,1,0,1, 0.11,2,10,2,10,17.4,29.7],
                          'Dengue': [14.8,49.3,0,1,0,1,0,0.2,5.7,3.46,5,7,0,6],
                          'MPox':[1.1,2.4,1,0,0,0,1,0.075,3.2,15.6,0.2,12.4,6.93,19.67],
                          'Zika': [2.4,5.6,0,1,0,1,0,0.0526,3,14,3,14,0,0],
                          'Influenza': [1,2,1,0,0,0,1,0.000002,2,2,2,2,6,9]}
    
    return feature_names, pandemic_meta_data[pandemic_name]

def get_date_from_date_time_list(datetime):
    return pd.to_datetime(datetime).date()

def pandemic_meta_data_imputation(model_input:list, method = "from_same_pandemic", impute_value = 0):

    return [impute_value if math.isnan(x) else x for x in model_input]