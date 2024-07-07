import pandas as pd 
import numpy as np
from utils.data_processing_compartment_model import process_data
from matplotlib import pyplot as plt
from tqdm import tqdm

data_file_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/'
output_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/data_inspection/case_figures/'
train_len = 46
test_len = 71

### Switch
plot_covid = False
plot_dengue = False
plot_ebola = False
plot_mpox = False
plot_sars = False
plot_influenza = True

### Covid Plots
if plot_covid:
    # Load Data
    covid_data = process_data(processed_data_path = data_file_dir + 'compartment_model_covid_data_objects_no_smoothing.pickle',
                            raw_data = False)

    # Plot Data
    for data in tqdm(covid_data):
        train_time_stamp = list(np.arange(0,train_len,1))
        test_time_stamp = list(np.arange(0,test_len,1))

        plt.figure()

        plt.plot(test_time_stamp,
                data.cumulative_case_number[:test_len],
                label='Test',
                color='grey',)
        
        plt.plot(train_time_stamp,
                data.cumulative_case_number[:train_len],
                label='Train',
                color='blue')
        
        plt.xlabel("Time from start date")
        plt.ylabel("Cumulative Case Number")
        plt.title(f"Covid {data.country_name} {data.domain_name} Cumulative Case Plot")
        plt.legend()

        plt.savefig(output_dir + f"covid/{data.country_name}_{data.domain_name}.png")

        plt.close()


### Dengue Plots
if plot_dengue:
    # Load Data
    dengue_data = process_data(processed_data_path = data_file_dir+'compartment_model_dengue_data_objects.pickle',
                            raw_data=False)

    # Plot Data
    for data in tqdm(dengue_data):
        train_time_stamp = list(np.arange(0,train_len,1))
        test_time_stamp = list(np.arange(0,test_len,1))

        plt.figure()

        plt.plot(test_time_stamp,
                data.cumulative_case_number[:test_len],
                label='Test',
                color='grey',)
        
        plt.plot(train_time_stamp,
                data.cumulative_case_number[:train_len],
                label='Train',
                color='blue')
        
        plt.xlabel("Time from start date")
        plt.ylabel("Cumulative Case Number")
        plt.title(f"Dengue {data.country_name} {data.domain_name} Cumulative Case Plot")
        plt.legend()

        plt.savefig(output_dir + f"dengue/{data.country_name}_{data.domain_name}.png")

        plt.close()


### Ebola Plots
if plot_ebola:
    # Load Data
    ebola_data = process_data(processed_data_path = data_file_dir+'compartment_model_ebola_data_objects.pickle',
                              raw_data=False)

    # Plot Data
    for data in tqdm(ebola_data):
        train_time_stamp = list(np.arange(0,train_len,1))
        test_time_stamp = list(np.arange(0,test_len,1))

        plt.figure()

        plt.plot(test_time_stamp,
                data.cumulative_case_number[:test_len],
                label='Test',
                color='grey',)
        
        plt.plot(train_time_stamp,
                data.cumulative_case_number[:train_len],
                label='Train',
                color='blue')
        
        plt.xlabel("Time from start date")
        plt.ylabel("Cumulative Case Number")
        plt.title(f"Ebola {data.country_name} {data.domain_name} Cumulative Case Plot")
        plt.legend()

        plt.savefig(output_dir + f"ebola/{data.country_name}_{data.domain_name}.png")

        plt.close()


### Mpox Plots
if plot_mpox:
    # Load Data
    mpox_data = process_data(processed_data_path = data_file_dir+'compartment_model_mpox_data_objects.pickle',
                              raw_data=False)

    # Plot Data
    for data in tqdm(mpox_data):
        train_time_stamp = list(np.arange(0,train_len,1))
        test_time_stamp = list(np.arange(0,test_len,1))

        plt.figure()

        plt.plot(test_time_stamp,
                data.cumulative_case_number[:test_len],
                label='Test',
                color='grey',)
        
        plt.plot(train_time_stamp,
                data.cumulative_case_number[:train_len],
                label='Train',
                color='blue')
        
        plt.xlabel("Time from start date")
        plt.ylabel("Cumulative Case Number")
        plt.title(f"Mpox {data.country_name} {data.domain_name} Cumulative Case Plot")
        plt.legend()

        plt.savefig(output_dir + f"mpox/{data.country_name}_{data.domain_name}.png")

        plt.close()


### SARS Plots
if plot_sars:
    # Load Data
    sars_data = process_data(processed_data_path = data_file_dir+'compartment_model_sars_data_objects.pickle',
                              raw_data=False)

    # Plot Data
    for data in tqdm(sars_data):
        train_time_stamp = list(np.arange(0,train_len,1))
        test_time_stamp = list(np.arange(0,test_len,1))

        if len(data.cumulative_case_number) < test_len:
            continue

        plt.figure()

        plt.plot(test_time_stamp,
                data.cumulative_case_number[:test_len],
                label='Test',
                color='grey',)
        
        plt.plot(train_time_stamp,
                data.cumulative_case_number[:train_len],
                label='Train',
                color='blue')
        
        plt.xlabel("Time from start date")
        plt.ylabel("Cumulative Case Number")
        plt.title(f"SARS {data.country_name} {data.domain_name} Cumulative Case Plot")
        plt.legend()

        plt.savefig(output_dir + f"sars/{data.country_name}_{data.domain_name}.png")

        plt.close()


### SARS Plots
if plot_influenza:

    for year in [2010,2011,2012,2013,2014,2015,2016,2017]:

        # Load Data
        influenza_data = process_data(processed_data_path = data_file_dir+f'compartment_model_{year}_influenza_data_objects.pickle',
                                      raw_data=False)

        # Plot Data
        for data in tqdm(influenza_data):
            train_time_stamp = list(np.arange(0,train_len,1))
            test_time_stamp = list(np.arange(0,test_len,1))

            if len(data.cumulative_case_number) < test_len:
                continue

            plt.figure()

            plt.plot(test_time_stamp,
                    data.cumulative_case_number[:test_len],
                    label='Test',
                    color='grey',)
            
            plt.plot(train_time_stamp,
                    data.cumulative_case_number[:train_len],
                    label='Train',
                    color='blue')
            
            plt.xlabel("Time from start date")
            plt.ylabel("Cumulative Case Number")
            plt.title(f"Influenza {year} {data.country_name} {data.domain_name} Cumulative Case Plot")
            plt.legend()

            plt.savefig(output_dir + f"influenza/{data.country_name}_{data.domain_name}_{year}.png")

            plt.close()