import pandas as pd
import numpy as np
import torch
from processing import read_pandemic_ts


def save_processed_data(look_back_len, pred_len, with_meta_data, moving_average, data_smoothing, meta_data_len):

    # covid_data_list = read_pandemic_ts(data_dir= "F:/Pandemic-Database/Processed_Data/Covid_19/Covid_World_Domain_Daily_CumCases.csv", 
    #                                    pandemic_name="Covid", look_back_len=look_back_len, pred_len=pred_len, 
    #                                    with_meta_data=with_meta_data, moving_average=moving_average,
    #                                    data_smoothing=data_smoothing, data_frequency="Daily", meta_data_len=meta_data_len)
    
    # print(len(covid_data_list)) # 374

    # ebola_data_list = read_pandemic_ts(data_dir= "F:/Pandemic-Database/Processed_Data/Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv", 
    #                                     pandemic_name="Ebola", look_back_len=look_back_len, pred_len=pred_len, 
    #                                     with_meta_data=with_meta_data, moving_average=moving_average,
    #                                     data_smoothing=data_smoothing, data_frequency="Weekly", meta_data_len=meta_data_len)
    
    # print(len(ebola_data_list)) # 3
    
    # mpox_data_list = read_pandemic_ts(data_dir= "F:/Pandemic-Database/Processed_Data/Monkeypox/Mpox_World_Country_Daily_CumCases.csv", 
    #                                     pandemic_name="MPox", look_back_len=look_back_len, pred_len=pred_len, 
    #                                     with_meta_data=with_meta_data, moving_average=moving_average,
    #                                    data_smoothing=data_smoothing, data_frequency="Daily", meta_data_len=meta_data_len)
    
    # print(len(mpox_data_list)) # 31
    
    # sars_data_list = read_pandemic_ts(data_dir= "F:/Pandemic-Database/Processed_Data/SARS/SARS_World_Country_Daily_CumCases.csv", 
    #                                     pandemic_name="SARS", look_back_len=look_back_len, pred_len=pred_len, 
    #                                     with_meta_data=with_meta_data, moving_average=moving_average,
    #                                     data_smoothing=data_smoothing, data_frequency="Daily", meta_data_len=meta_data_len)
    
    # print(sars_data_list) # 5

    # dengue_data_list = read_pandemic_ts(data_dir= "F:/Pandemic-Database/Processed_Data/Dengue_Fever/Dengue_AMRO_Country_Weekly_CumCases.csv", 
    #                                     pandemic_name="Dengue", look_back_len=look_back_len, pred_len=pred_len, 
    #                                     with_meta_data=with_meta_data, moving_average=moving_average,
    #                                     data_smoothing=data_smoothing, data_frequency="Weekly", meta_data_len=meta_data_len)
    
    # print(dengue_data_list) # 2
    


if __name__ == '__main__':
    save_processed_data(look_back_len = 30,
                        pred_len = 60, 
                        with_meta_data = True,
                        moving_average = 7, 
                        data_smoothing = True, 
                        meta_data_len = 14)