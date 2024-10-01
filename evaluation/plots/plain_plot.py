import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from utils.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
from tqdm import tqdm
from pathlib import Path

compare_trial_name = "ResNet50_Combined_Loss_[0.5,100]"
Path(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/plots/{compare_trial_name}/').mkdir(parents=False, exist_ok=True)

delphi_pred_case = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_46_71_case_only_pred_case.csv')
guided_pred_case = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/07-29-0900/case_prediction.csv')

target_pandemic_data = process_data(processed_data_path = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/compartment_model_covid_data_objects.pickle',
                                        raw_data=False)
    
target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                              target_training_len=28,
                                              pred_len = 84,
                                              batch_size=64,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=True,)

time_stamp = np.arange(0,71,1)

for index, row in tqdm(guided_pred_case.iterrows(), total=len(guided_pred_case)):

    country = row['Country']
    domain = row['Domain']

    if pd.isna(domain):
        delphi_case = delphi_pred_case[(delphi_pred_case['country']==country) & (delphi_pred_case['domain'].isna())].values[0][4:]
        # selftune_case = selftune_pred_case[(selftune_pred_case['Country']==country) & (selftune_pred_case['Domain'].isna())].values[0][2:]
        guided_case = guided_pred_case[(guided_pred_case['Country']==country) & (guided_pred_case['Domain'].isna())].values[0][2:]
        true_case = [item.cumulative_case_number for item in target_pandemic_data if ((item.country_name == country)&(pd.isna(item.domain_name)))][0][:71]
        
    else:
        delphi_case = delphi_pred_case[(delphi_pred_case['country']==country) & (delphi_pred_case['domain'] == domain)].values[0][4:]
        # selftune_case = selftune_pred_case[(selftune_pred_case['Country']==country) & (selftune_pred_case['Domain'] == domain)].values[0][2:]
        guided_case = guided_pred_case[(guided_pred_case['Country']==country) & (guided_pred_case['Domain'] == domain)].values[0][2:]
        true_case = [item.cumulative_case_number for item in target_pandemic_data if ((item.country_name == country)&(item.domain_name == domain))][0][:71]
        

    plt.figure()
    plt.plot(time_stamp,
             true_case,
             label='True Case')
    plt.plot(time_stamp,
             delphi_case,
             label='DELPHI')
    plt.plot(time_stamp,
             guided_case,
             label='Past-Guided')
    plt.legend()

    plt.savefig(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/plots/{compare_trial_name}/{country}_{domain}.png')
    
    plt.close()
    
