import pandas as pd
import pickle
from tqdm import tqdm
import multiprocessing
from scripts.get_delphi_parameters import get_perfect_parameters, visualize_result

def generate_and_save_baseline_parameters(data_object):

    try:
        parameters, x_sol, y_true, data = get_perfect_parameters(data_object=data_object,
                                                                    train_length=60,
                                                                    test_length=90,)

        mape_loss = visualize_result(x_sol[15],
                                     y_true,
                                     output_dir = save_fig_dir,
                                     data_object=data)

    except:

        parameters = [-999] * 12
        mape_loss = 999

    return data_object.country_name, data_object.domain_name, parameters, mape_loss

if __name__ == '__main__':

    pandemic_name = 'covid'

    data_path = f"/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_{pandemic_name}_data_objects.pickle"

    with open(data_path, 'rb') as f:
        data_object_list = pickle.load(f)

    data = [item for item in data_object_list if item.country_name == 'Russia']
    
    russia_data = data[0]
    for i in range(58):
        print(f"{russia_data.timestamps[i]}: {russia_data.cumulative_case_number[i]}")

    exit()

    data_object_list = data

    parameter_list = []
    mape_list = []
    country_name_list = []
    domain_name_list = []
    save_fig_dir = f"/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/output/{pandemic_name}_figures/"

    core_num = multiprocessing.cpu_count()

    print(f"Using {core_num} cores in running!")

    with multiprocessing.Pool(core_num) as pool:
        with tqdm(total=len(data_object_list)) as pbar:
            for country_name, domain_name, parameters, mape_loss in pool.imap_unordered(generate_and_save_baseline_parameters, data_object_list):
                parameter_list.append(parameters)   
                country_name_list.append(country_name)
                domain_name_list.append(domain_name)
                mape_list.append(mape_loss)
                pbar.update()

    parameter_df = pd.DataFrame(parameter_list, columns=['alpha','days','r_s','r_dth','p_dth','r_dthdecay','k1','k2','jump','t_jump','std_normal','k3'])
    parameter_df['country'] = country_name_list
    parameter_df['domain'] = domain_name_list
    parameter_df['last_15_days_mape'] = mape_list

    print(parameter_df)

    # parameter_df.to_csv(f"/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_Baseline/output/DELPHI_params_{pandemic_name}.csv",
    #                     index = False)
