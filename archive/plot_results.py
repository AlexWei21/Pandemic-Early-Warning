import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from tqdm import tqdm

def save_plot_curves(save_dir:str,
                     location:str,
                     true_case:list,
                     delphi_pred:list,
                     selftune_pred:list,
                     past_guided_pred:list):

    time_stamp = list(np.arange(0,71,1))

    plt.figure()
    plt.plot(time_stamp,
             true_case[:71],
             '-b',
             label='True Case')
    plt.plot(time_stamp,
             selftune_pred,
             '-r',
             label='Self Tune Predicted Case')
    plt.plot(time_stamp,
             past_guided_pred,
             '-g',
             label='Past Pandemic Guided Predicted Case')
    plt.legend()
    plt.savefig(save_dir + f"{location[0]}_{location[1]}_plot.png")
    plt.close()

def visualiza_model_outputs(delphi_pred_dir:str,
                            selftune_pred_dir:str,
                            guided_pred_dir:str,
                            delphi_better_save_dir:str,
                            selftune_better_save_dir:str,
                            guided_better_save_dir:str,
                            comparison_dir:str,):
    
    ## Load Dataframe
    comparison_df = pd.read_csv(comparison_dir)
    selftune_pred_df = pd.read_csv(selftune_pred_dir)
    guided_pred_df = pd.read_csv(guided_pred_dir)

    with open('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_covid_data_objects_no_smoothing.pickle', 'rb') as f:
        data_object_list = pickle.load(f)

    ## Compare Results
    comparison_df['selftune_better'] = np.where((comparison_df['selftune_outsample_mae'] < comparison_df['delphi_outsample_mae']) & (comparison_df['selftune_outsample_mae'] < comparison_df['guided_outsample_mae']), 1, 0)
    comparison_df['guided_better'] = np.where((comparison_df['guided_outsample_mae'] < comparison_df['delphi_outsample_mae']) & (comparison_df['guided_outsample_mae'] < comparison_df['selftune_outsample_mae']), 1, 0)
    comparison_df['location'] = list(zip(comparison_df['country'],comparison_df['domain']))

    ## Get Location List
    selftune_best_df = comparison_df[comparison_df['selftune_better']==1]
    guided_best_df = comparison_df[comparison_df['guided_better']==1]    
    selftune_best_location_list = list(selftune_best_df['location'])
    guided_best_location_list = list(guided_best_df['location'])

    ## 
    selftune_pred_df['location'] = list(zip(selftune_pred_df['Country'],selftune_pred_df['Domain']))
    guided_pred_df['location'] = list(zip(guided_pred_df['Country'],guided_pred_df['Domain']))
    case_row_name = np.arange(0,71,1)
    case_row_name = [str(item) for item in case_row_name]

    ## selftune_better
    selftune_better_pred_guided = guided_pred_df[guided_pred_df['location'].isin(selftune_best_location_list)]
    selftune_better_pred_selftune = selftune_pred_df[selftune_pred_df['location'].isin(selftune_best_location_list)]

    for location in tqdm(selftune_best_location_list):
        selftune_row = selftune_better_pred_guided[selftune_better_pred_guided['location'] == location]
        guided_row = selftune_better_pred_selftune[selftune_better_pred_selftune['location'] == location]

        if pd.isna(location[1]):
            true_case = list([item.cumulative_case_number for item in data_object_list if ((item.country_name == location[0]) & (pd.isna(item.domain_name)))][0])
        else:
            true_case = list([item.cumulative_case_number for item in data_object_list if ((item.country_name == location[0]) & (item.domain_name == location[1]))][0])

        save_plot_curves(save_dir=selftune_better_save_dir,
                         location=location,
                         true_case=true_case,
                         delphi_pred=None,
                         selftune_pred=selftune_row[case_row_name].values[0],
                         past_guided_pred=guided_row[case_row_name].values[0],)
    
    ## guided_better
    guided_better_pred_guided = guided_pred_df[guided_pred_df['location'].isin(guided_best_location_list)]
    guided_better_pred_selftune = selftune_pred_df[selftune_pred_df['location'].isin(guided_best_location_list)]

    for location in tqdm(guided_best_location_list):
        selftune_row = guided_better_pred_guided[guided_better_pred_guided['location'] == location]
        guided_row = guided_better_pred_selftune[guided_better_pred_selftune['location'] == location]

        if pd.isna(location[1]):
            true_case = list([item.cumulative_case_number for item in data_object_list if ((item.country_name == location[0]) & (pd.isna(item.domain_name)))][0])
        else:
            true_case = list([item.cumulative_case_number for item in data_object_list if ((item.country_name == location[0]) & (item.domain_name == location[1]))][0])

        save_plot_curves(save_dir=guided_better_save_dir,
                         location=location,
                         true_case=true_case,
                         delphi_pred=None,
                         selftune_pred=selftune_row[case_row_name].values[0],
                         past_guided_pred=guided_row[case_row_name].values[0],)    


if __name__ == '__main__':

    visualiza_model_outputs(delphi_pred_dir=None,
                            selftune_pred_dir='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/DeepCompartmentModel/self-tune_only/case_prediction.csv',
                            guided_pred_dir='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/DeepCompartmentModel/past_pandemic_guided/case_prediction.csv',
                            delphi_better_save_dir='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/model_comparison/plots/delphi_better/',
                            selftune_better_save_dir='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/model_comparison/plots/selftune_better/',
                            guided_better_save_dir='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/model_comparison/plots/guided_better/',
                            comparison_dir='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/DeepCompartmentModel/self-tune_only/compare_results.csv',)