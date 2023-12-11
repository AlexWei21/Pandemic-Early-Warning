import pandas as pd
import numpy as np
from data.data_utils import get_meta_data, process_daily_data, process_weekly_data
import torch
from data.data import Pandemic_Data
from tqdm import tqdm


def read_pandemic_ts (data_dir, pandemic_name, look_back_len = 100, pred_len = 200, with_meta_data = False, multiple_files = False,
                      moving_average = 1, data_smoothing = 'False', target_data_frequency = 'Daily', past_data_frequency = [], meta_data_len = 14):

    if multiple_files == False:
        data_list = read_one_pandemic_file(data_dir=data_dir,
                               pandemic_name=pandemic_name,
                               look_back_len=look_back_len,
                               pred_len=pred_len,
                               with_meta_data=with_meta_data,
                               moving_average=moving_average,
                               data_smoothing=data_smoothing,
                               data_frequency=target_data_frequency,
                               meta_data_len=meta_data_len)
        
        print("Length of Testing Data:", len(data_list))

    else:
        for i in range(len(pandemic_name)):
            if i == 0:
                data_list = read_one_pandemic_file(data_dir=data_dir[0],
                               pandemic_name=pandemic_name[0],
                               look_back_len=look_back_len,
                               pred_len=pred_len,
                               with_meta_data=with_meta_data,
                               moving_average=moving_average,
                               data_smoothing=data_smoothing,
                               data_frequency=past_data_frequency[0],
                               meta_data_len=meta_data_len)
            else:
                data_list.extend(read_one_pandemic_file(data_dir=data_dir[i],
                               pandemic_name=pandemic_name[i],
                               look_back_len=look_back_len,
                               pred_len=pred_len,
                               with_meta_data=with_meta_data,
                               moving_average=moving_average,
                               data_smoothing=data_smoothing,
                               data_frequency=past_data_frequency[i],
                               meta_data_len=meta_data_len))

        print("Length of Training Data:", len(data_list))

    return data_list


def read_one_pandemic_file(data_dir, pandemic_name, look_back_len = 100, pred_len = 200, with_meta_data = False,
                      moving_average = 1, data_smoothing = 'False', data_frequency = 'Daily', meta_data_len = 14):
        
    pandemic_ts_file = pd.read_csv(data_dir, dtype={'Domain':str})

    data_type = pandemic_ts_file['type'][0]

    country_names = pandemic_ts_file['Country'].unique()

    data_list = []

    meta_data_features, meta_data = get_meta_data(pandemic_name)

    for country in tqdm(country_names):

        processing_country_data = pandemic_ts_file[pandemic_ts_file['Country'] == country]
        
        processing_country_data = processing_country_data.reset_index(drop=True)

        # print(processing_country_data)

        for domain in processing_country_data['Domain'].unique():

            if pd.isna(domain):
                processing_domain_data = processing_country_data[processing_country_data['Domain'].isna()]
            else:
                processing_domain_data = processing_country_data[processing_country_data['Domain'] == domain]

            for subdomain in processing_domain_data['Sub-Domain'].unique():

                data_point = Pandemic_Data(look_back_len=look_back_len, pred_len=pred_len, meta_data_len=meta_data_len)

                data_point.pandemic_name = pandemic_name
                data_point.country_name = country
                data_point.domain_name = domain
                data_point.subdomain_name = subdomain

                if pd.isna(subdomain):
                    if pd.isna(domain):
                        print(f"Processing {country} Overall Data")
                    else:
                        print(f"Processing {country} {domain} Data")
                    processing_subdomain_data = processing_domain_data[processing_domain_data['Sub-Domain'].isna()]
                else:
                    print(f"Processing {country} {domain} {subdomain} Data")
                    processing_subdomain_data = processing_domain_data[processing_domain_data['Sub-Domain'] == subdomain]

                ## Set the first day that case number exceed 100 as the start date
                if max(processing_subdomain_data['number']) < 100:
                    print(country,domain,subdomain, "data never exceeded 100 cases")
                    continue
                else:
                    start_idx = np.argmax(processing_subdomain_data['number']>100)

                processing_subdomain_data = processing_subdomain_data[start_idx:]

                min_date = min(pd.to_datetime(processing_subdomain_data['date']).dt.date)
                max_date = max(pd.to_datetime(processing_subdomain_data['date']).dt.date)

                if (max_date - min_date).days < (look_back_len + pred_len):
                    print(country,domain,subdomain, "doesn't contain enough time span for desired look back length and prediction length.")
                    continue

                if data_frequency == 'Daily':

                    look_back_data, pred_data, look_back_timestamp, pred_data_timestamp = process_daily_data(processing_subdomain_data, 
                                                                                                              data_smoothing,
                                                                                                              look_back = look_back_len,
                                                                                                              pred_len = pred_len,
                                                                                                              avg_len=moving_average)
                    # look_back_timestamp = processing_subdomain_data['date'].to_numpy()[:look_back_len]
                    # pred_data_timestamp = processing_subdomain_data['date'].to_numpy()[look_back_len:look_back_len + pred_len]

                    # look_back_data = process_daily_data(processing_subdomain_data['number'].to_numpy()[:look_back_len], 
                    #                                     moving_average)
                    # pred_data = process_daily_data(processing_subdomain_data['number'].to_numpy()[look_back_len:look_back_len + pred_len], 
                    #                                moving_average)
                elif data_frequency == 'Weekly':
                    look_back_data, pred_data, look_back_timestamp, pred_data_timestamp = process_weekly_data(processing_subdomain_data, 
                                                                                                              data_smoothing,
                                                                                                              look_back = look_back_len,
                                                                                                              pred_len = pred_len)

                else:
                    print("Data Frequency Type not supported, currently only accepting Daily and Weekly Data.")
                    exit(1)

                data_point.x = look_back_data
                data_point.y = pred_data
                data_point.meta_data = np.array(meta_data)

                data_point.time_stamp_x = look_back_timestamp
                data_point.time_stamp_y = pred_data_timestamp

                data_point.decoder_input = np.concatenate([[data_point.x[-1]],data_point.y[:-1]])

                data_point.x = torch.from_numpy(data_point.x).to(torch.float32)
                data_point.y = torch.from_numpy(data_point.y).to(torch.float32)
                data_point.decoder_input = torch.from_numpy(data_point.decoder_input).to(torch.float32)
                

                data_point.meta_data = torch.from_numpy(data_point.meta_data).to(torch.float)

                data_list.append(data_point)

    return data_list