import pandas as pd
import numpy as np
from data.data import Compartment_Model_Pandemic_Data
from scripts.get_delphi_parameters import get_perfect_parameters, visualize_result
import pickle
import multiprocessing
from tqdm import tqdm


def generate_and_save_delphi_performance(data_object):

    train_length = 46
    test_length = 71

    try: 

        parameters, x_sol, y_true, data = get_perfect_parameters(data_object,
                                train_length = train_length,
                                test_length = test_length,)

        ## Case Plot
        outsample_mae, overall_mae, insample_mae, outsample_mape, overall_mape, insample_mape = visualize_result(x_sol[15],
                                    y_true,
                                    output_dir = '/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/covid_figures/case_only_figures/case/',
                                    data_object=data,
                                    type = 'case',
                                    train_len=train_length)
            
        performance_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        parameter_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        predicted_case_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        parameter_row = parameter_row + list(parameters)
        performance_row = performance_row + [outsample_mape, overall_mape, insample_mape]
        performance_row = performance_row + [outsample_mae, overall_mae, insample_mae]
        predicted_case_row = predicted_case_row + list(x_sol[15][:test_length])
    
    except:
        
        parameters = [-999] * 12

        performance_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        parameter_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        predicted_case_row = [data_object.country_name, data_object.domain_name, data_object.first_day_above_hundred]
        parameter_row = parameter_row + parameters
        performance_row = performance_row + [999,999,999]
        performance_row = performance_row + [999,999,999]
        predicted_case_row = predicted_case_row + [0] * test_length

        print(f"{data_object.country_name} {data_object.domain_name} Fail to Generate DELPHI Prediction")


    return parameter_row, performance_row, predicted_case_row


if __name__ == '__main__':

    core_num = multiprocessing.cpu_count()

    print(f'Using {core_num} cores in running!')

    performance_df = []
    parameter_df = []
    predicted_case_df = []

    failure_list = []

    with open('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_covid_data_objects_no_smoothing.pickle', 'rb') as f:
        data_object_list = pickle.load(f)

    for item in data_object_list:
        item.cumulative_death_number = None

    with multiprocessing.Pool(core_num) as pool:
        with tqdm(total=len(data_object_list)) as pbar:
            for parameter_row, performance_row, predicted_case_row in pool.imap_unordered(generate_and_save_delphi_performance, data_object_list):
                performance_df.append(performance_row)
                parameter_df.append(parameter_row)
                predicted_case_df.append(predicted_case_row)
                pbar.update()

    parameter_df = pd.DataFrame(parameter_df,
                                columns=['country', 'domain', 'first_day_above_hundred','alpha','days','r_s','r_dth','p_dth','r_dthdecay','k1','k2','jump','t_jump','std_normal','k3'])

    performance_df = pd.DataFrame(performance_df,
                                columns=['country', 'domain', 'first_day_above_hundred',
                                        'outsample_mape','overall_mape','train_mape',
                                        'outsample_mae','overall_mae','train_mae',])
    
    case_column_name = [str(item) for item in np.arange(0,71,1)]

    predicted_case_df = pd.DataFrame(predicted_case_df,
                                     columns=['country', 'domain', 'first_day_above_hundred'] + case_column_name)

    parameter_df.to_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/covid_46_71_case_only_parameters.csv')
    performance_df.to_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/covid_46_71_case_only_performance.csv')
    predicted_case_df.to_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/covid_46_71_case_only_pred_case.csv')