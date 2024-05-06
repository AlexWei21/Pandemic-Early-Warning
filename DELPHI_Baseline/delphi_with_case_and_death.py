import pandas as pd
import numpy as np
from data.data import Compartment_Model_Pandemic_Data
from scripts.get_delphi_parameters import get_perfect_parameters, visualize_result
import pickle
import multiprocessing
from tqdm import tqdm

# covid_case = pd.read_csv("/export/home/dor/zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumCases.csv")
# covid_death = pd.read_csv("/export/home/dor/zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumDeaths.csv")

with open('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_covid_data_objects_no_smoothing.pickle', 'rb') as f:
    data_object_list = pickle.load(f)

# data_object_list = data_object_list[:2]

def generate_and_save_delphi_performance(data_object):

    train_length = 46
    test_length = 71

    try: 
        parameters, x_sol, y_true, data = get_perfect_parameters(data_object,
                            train_length = train_length,
                            test_length = test_length,)

        ## Case Plot
        last_15_day_mape_loss, all_mape_loss, train_loss,last_15_day_mae_loss, all_mae_loss, train_mae_loss= visualize_result(x_sol[15],
                                    y_true,
                                    output_dir = '/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/covid_figures/case/',
                                    data_object=data,
                                    type = 'case',
                                    train_len=train_length)

        ## Death Plot
        last_15_day_death_mape_loss, all_death_mape_loss, death_train_loss, last_15_day_death_mae_loss, all_death_mae_loss, death_train_mae_loss = visualize_result(x_sol[14],
                                    data_object.cumulative_death_number[:test_length],
                                    output_dir = '/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/covid_figures/death/',
                                    data_object=data,
                                    type = 'death',
                                    train_len = train_length)
        
        performance_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        parameter_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        parameter_row = parameter_row + list(parameters)
        performance_row = performance_row + [last_15_day_mape_loss, all_mape_loss, train_loss]
        performance_row = performance_row + [last_15_day_mae_loss, all_mae_loss, train_mae_loss]
        performance_row = performance_row + [last_15_day_death_mape_loss, all_death_mape_loss, death_train_loss]
        performance_row = performance_row + [last_15_day_death_mae_loss, all_death_mae_loss, death_train_mae_loss]
    
    except:
        
        parameters = [-999] * 12

        performance_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        parameter_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        parameter_row = parameter_row + parameters

        performance_row = performance_row + [999,999,999]
        performance_row = performance_row + [999,999,999]
        performance_row = performance_row + [999,999,999]
        performance_row = performance_row + [999,999,999]

        print(f"{data_object.country_name} {data_object.domain_name} Fail to Generate DELPHI Prediction")


    return parameter_row, performance_row


core_num = multiprocessing.cpu_count()

print(f'Using {core_num} cores in running!')

performance_df = []
parameter_df = []
failure_list = []

with multiprocessing.Pool(core_num) as pool:
    with tqdm(total=len(data_object_list)) as pbar:
        for parameter_row, performance_row in pool.imap_unordered(generate_and_save_delphi_performance, data_object_list):
            performance_df.append(performance_row)
            parameter_df.append(parameter_row)
            pbar.update()

parameter_df = pd.DataFrame(parameter_df,
                            columns=['country', 'domain', 'first_day_above_hundred','alpha','days','r_s','r_dth','p_dth','r_dthdecay','k1','k2','jump','t_jump','std_normal','k3'])

performance_df = pd.DataFrame(performance_df,
                              columns=['country', 'domain', 'first_day_above_hundred',
                                       'Last_15_Days_Case_MAPE','Case_MAPE','Train_Case_MAPE',
                                       'Last_15_Days_Case_MAE','Case_MAE','Train_Case_MAE',
                                       'Last_15_Days_Death_MAPE','Death_MAPE','Train_Death_MAPE',
                                       'Last_15_Days_Death_MAE','Death_MAE','Train_Death_MAE'])

parameter_df.to_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/covid_46_71_case_death_parameters.csv')
performance_df.to_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/covid_46_71_case_death_performance.csv')