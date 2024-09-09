import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from utils.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
from tqdm import tqdm
from pathlib import Path

## 56 Days - 84 Days
Path(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/plots/DELPHI_HGDCM_56_84/').mkdir(parents=False, exist_ok=True)

delphi_pred_case = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_56_84_case_only_pred_case.csv')
hgdcm_pred_case = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/08-28-1100/case_prediction.csv')

target_pandemic_data = process_data(processed_data_path = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/compartment_model_covid_data_objects.pickle',
                                        raw_data=False)
    
target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                              target_training_len=56,
                                              pred_len = 84,
                                              batch_size=64,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=True,)

time_stamp = np.arange(0,84,1)

for index, row in tqdm(hgdcm_pred_case.iterrows(), total=len(hgdcm_pred_case)):

    country = row['Country']
    domain = row['Domain']

    if pd.isna(domain):
        delphi_case = delphi_pred_case[(delphi_pred_case['country']==country) & (delphi_pred_case['domain'].isna())].values[0][4:]
        hgdcm_case = hgdcm_pred_case[(hgdcm_pred_case['Country']==country) & (hgdcm_pred_case['Domain'].isna())].values[0][2:]
        true_case = [item.cumulative_case_number for item in target_pandemic_data if ((item.country_name == country)&(pd.isna(item.domain_name)))][0][:84]
    else:
        delphi_case = delphi_pred_case[(delphi_pred_case['country']==country) & (delphi_pred_case['domain'] == domain)].values[0][4:]
        hgdcm_case = hgdcm_pred_case[(hgdcm_pred_case['Country']==country) & (hgdcm_pred_case['Domain'] == domain)].values[0][2:]
        true_case = [item.cumulative_case_number for item in target_pandemic_data if ((item.country_name == country)&(item.domain_name == domain))][0][:84]
        

    plt.figure(figsize=(6.4,3.2))
    ax = plt.subplot(111)
    plt.plot(time_stamp,
             true_case,
             'k--',
             label='True Case')
    plt.plot(time_stamp,
             delphi_case,
             label='DELPHI')
    plt.plot(time_stamp,
             hgdcm_case,
             label='History Guided Deep Compartmental Model')
    ax.spines[['right', 'top']].set_visible(False)
    plt.xlabel("Days")
    plt.ylabel("Cumulative Case Number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/plots/DELPHI_HGDCM_56_84/{country}_{domain}.png')
    plt.close()
    
## 42 Days - 84 Days
Path(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/plots/DELPHI_HGDCM_42_84/').mkdir(parents=False, exist_ok=True)

delphi_pred_case = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_42_84_case_only_pred_case.csv')
hgdcm_pred_case = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/08-28-1500/case_prediction.csv')

target_pandemic_data = process_data(processed_data_path = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/compartment_model_covid_data_objects.pickle',
                                        raw_data=False)
    
target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                              target_training_len=42,
                                              pred_len = 84,
                                              batch_size=64,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=True,)

time_stamp = np.arange(0,84,1)

for index, row in tqdm(hgdcm_pred_case.iterrows(), total=len(hgdcm_pred_case)):

    country = row['Country']
    domain = row['Domain']

    if pd.isna(domain):
        delphi_case = delphi_pred_case[(delphi_pred_case['country']==country) & (delphi_pred_case['domain'].isna())].values[0][4:]
        hgdcm_case = hgdcm_pred_case[(hgdcm_pred_case['Country']==country) & (hgdcm_pred_case['Domain'].isna())].values[0][2:]
        true_case = [item.cumulative_case_number for item in target_pandemic_data if ((item.country_name == country)&(pd.isna(item.domain_name)))][0][:84]
    else:
        delphi_case = delphi_pred_case[(delphi_pred_case['country']==country) & (delphi_pred_case['domain'] == domain)].values[0][4:]
        hgdcm_case = hgdcm_pred_case[(hgdcm_pred_case['Country']==country) & (hgdcm_pred_case['Domain'] == domain)].values[0][2:]
        true_case = [item.cumulative_case_number for item in target_pandemic_data if ((item.country_name == country)&(item.domain_name == domain))][0][:84]
        

    plt.figure(figsize=(6.4,3.2))
    ax = plt.subplot(111)
    plt.plot(time_stamp,
             true_case,
             'k--',
             label='True Case')
    plt.plot(time_stamp,
             delphi_case,
             label='DELPHI')
    plt.plot(time_stamp,
             hgdcm_case,
             label='History Guided Deep Compartmental Model')
    ax.spines[['right', 'top']].set_visible(False)
    plt.xlabel("Days")
    plt.ylabel("Cumulative Case Number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/plots/DELPHI_HGDCM_42_84/{country}_{domain}.png')
    plt.close()

## 28 Days - 84 Days
Path(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/plots/DELPHI_HGDCM_28_84/').mkdir(parents=False, exist_ok=True)

delphi_pred_case = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_28_84_case_only_pred_case.csv')
hgdcm_pred_case = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/08-29-0800_28-84/case_prediction.csv')

target_pandemic_data = process_data(processed_data_path = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/compartment_model_covid_data_objects.pickle',
                                        raw_data=False)
    
target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                              target_training_len=28,
                                              pred_len = 84,
                                              batch_size=64,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=True,)

time_stamp = np.arange(0,84,1)

for index, row in tqdm(hgdcm_pred_case.iterrows(), total=len(hgdcm_pred_case)):

    country = row['Country']
    domain = row['Domain']

    if pd.isna(domain):
        delphi_case = delphi_pred_case[(delphi_pred_case['country']==country) & (delphi_pred_case['domain'].isna())].values[0][4:]
        hgdcm_case = hgdcm_pred_case[(hgdcm_pred_case['Country']==country) & (hgdcm_pred_case['Domain'].isna())].values[0][2:]
        true_case = [item.cumulative_case_number for item in target_pandemic_data if ((item.country_name == country)&(pd.isna(item.domain_name)))][0][:84]
    else:
        delphi_case = delphi_pred_case[(delphi_pred_case['country']==country) & (delphi_pred_case['domain'] == domain)].values[0][4:]
        hgdcm_case = hgdcm_pred_case[(hgdcm_pred_case['Country']==country) & (hgdcm_pred_case['Domain'] == domain)].values[0][2:]
        true_case = [item.cumulative_case_number for item in target_pandemic_data if ((item.country_name == country)&(item.domain_name == domain))][0][:84]
        

    plt.figure(figsize=(6.4,3.2))
    ax = plt.subplot(111)
    plt.plot(time_stamp,
             true_case,
             'k--',
             label='True Case')
    plt.plot(time_stamp,
             delphi_case,
             label='DELPHI')
    plt.plot(time_stamp,
             hgdcm_case,
             label='History Guided Deep Compartmental Model')
    ax.spines[['right', 'top']].set_visible(False)
    plt.xlabel("Days")
    plt.ylabel("Cumulative Case Number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/plots/DELPHI_HGDCM_28_84/{country}_{domain}.png')
    plt.close()