import pandas as pd 
import numpy as np
from data.data import Compartment_Model_Pandemic_Dataset
from data.data_processing_compartment_model import process_data
from data.data_utils import parameter_max
from model.Compartment_Model.DELPHI import DELPHI_model
from scipy.integrate import solve_ivp
from utils.training_utils import create_fitting_data_from_validcases, get_initial_conditions
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
from model.delphi_default_parameters import (
    perfect_washington_parameter_list,
    p_v,
    p_d,
    p_h,
    max_iter,
    dict_default_reinit_parameters,
    dict_default_reinit_lower_bounds,
    IncubeD,
    DetectD,
    RecoverID,
    RecoverHD,
    VentilatedD)

def sampling_from_distribution (data: list,
                                sample_size: int = 1,
                                train_len: int = 30,
                                pred_len: int = 60,
                                sigma: float = 0.001,
                                save_plot: bool = False,):
    
    synthetic_data_list = []

    for i in tqdm(range(len(data))):

        true_params = data[i].true_delphi_params

        ## Get Noisiness of Each Time Stamp of True Data
        simulated_case_num = get_cumulative_case_num(data[i],
                                                     true_params,
                                                     train_len,
                                                     pred_len,)
        simulated_daily_case = [simulated_case_num[n] - simulated_case_num[n-1] for n in range(1,len(simulated_case_num))]

        observed_case_num = data[i].cumulative_case_number[:train_len + pred_len]
        observed_daily_case = [observed_case_num[n] - observed_case_num[n-1] for n in range(1,len(simulated_case_num))]

        percentage_daily_diff = [(observed_daily_case[k] - simulated_daily_case[k])/simulated_daily_case[k] if simulated_daily_case[k] != 0 else 0 for k in range(train_len + pred_len - 1)]
        
        # Generate Noise Factor
        noise_factor = [np.random.uniform((1 - noise), (1 + noise), 1)[0] for noise in percentage_daily_diff]

        ## Generate Samples from Sampling Parameters
        synthetic_params_list = []
        synthetic_case_num_list = []
        noisy_synthetic_case_num_list = []

        for k in range(sample_size):

            synthetic_params = np.zeros(len(true_params))

            for j in range(len(true_params)):
                found = False
                count = 0
                while not found:
                    synthetic_params[j] = np.random.normal(loc = true_params[j],
                                                        scale = sigma * parameter_max[j],)
                    count += 1

                    if count == 100:
                        print(j)
                        print("Real Value:", true_params[j])
                        print("Max_Value", parameter_max[j])
                        print("Synthesized Value:", synthetic_params[j])
                        exit()

                    if (synthetic_params[j] < parameter_max[j]) & (synthetic_params[j] > 0):
                        found = True

            synthetic_params_list.append(synthetic_params)

            synthesized_case_num = get_cumulative_case_num(data[i],
                                                          synthetic_params,
                                                          train_len,
                                                          pred_len)

            ## Add Noise to Synthesized Daily Data
            synthesized_daily_case = [synthesized_case_num[n] - synthesized_case_num[n-1] for n in range(1,len(synthesized_case_num))]
            noisy_synthesized_daily_case = [synthesized_daily_case[n] * noise_factor[n] for n in range(len(synthesized_daily_case))]

            ## Daily Case couldn't be smaller than 0
            noisy_synthesized_daily_case = [0 if item < 0 else item for item in noisy_synthesized_daily_case]

            ## Calculate Cumulative Sum of Noisy Data
            noisy_synthesized_daily_case.insert(0,synthesized_case_num[0])
            noisy_synthesized_case_num = list(np.cumsum(noisy_synthesized_daily_case))

            ## Create New Data Point from Synthesized Data
            synthetic_data_point = copy.deepcopy(data[i])
            synthetic_data_point.true_delphi_parameters = synthetic_params
            # synthetic_data_point.cumulative_case_number = noisy_synthesized_case_num
            synthetic_data_point.ts_input = noisy_synthesized_case_num[:train_len]
            synthetic_case_num_list.append(synthesized_case_num)
            noisy_synthetic_case_num_list.append(noisy_synthesized_case_num)
            synthetic_data_list.append(synthetic_data_point)
        
        true_params_case_num = get_cumulative_case_num(data[i],
                                                       true_params,
                                                       train_len,
                                                       pred_len,)        

        if save_plot:
        
            plt.figure()

            plt.plot(data[i].cumulative_case_number[:90], label = 'Observed Cumulative Case')
            plt.plot(true_params_case_num, label = 'Fitted Perfect Parameter')

            for k in range(sample_size):
                plt.plot(synthetic_case_num_list[k], label = 'Synthetic Parameter' + str(k))
                plt.plot(noisy_synthetic_case_num_list[k], label = 'Synthetic Parameter' + str(k)+ " with Noise")

            plt.legend()

            plt.show()
            exit()

            plt.savefig(f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/Data_Sampling_Output/{data[i].country_name}_synthetic_samples.png')

            plt.close()

    return synthetic_data_list


def get_cumulative_case_num(data,
                            delphi_params: list,
                            train_len,
                            pred_len,
                            ):

    t_predictions = [i for i in range(train_len + pred_len)]

    ## Setting Range
    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = delphi_params
    delphi_params = [
                max(alpha, dict_default_reinit_parameters["alpha"]),
                days,
                max(r_s, dict_default_reinit_parameters["r_s"]),
                max(min(r_dth, 1), dict_default_reinit_parameters["r_dth"]),
                max(min(p_dth, 1), dict_default_reinit_parameters["p_dth"]),
                max(r_dthdecay, dict_default_reinit_parameters["r_dthdecay"]),
                max(k1, dict_default_reinit_parameters["k1"]),
                max(k2, dict_default_reinit_parameters["k2"]),
                max(jump, dict_default_reinit_parameters["jump"]),
                max(t_jump, dict_default_reinit_parameters["t_jump"]),
                max(std_normal, dict_default_reinit_parameters["std_normal"]),
                max(k3, dict_default_reinit_lower_bounds["k3"]),
                ]

    if data.cumulative_death_number is not None:
        validcases = pd.DataFrame(list(zip(np.arange(0,len(data.timestamps)), data.cumulative_case_number, data.cumulative_death_number)),
                                columns=['day_since100','case_cnt','death_cnt'])
    else: 
            validcases = pd.DataFrame(list(zip(np.arange(0,len(data.timestamps)), data.cumulative_case_number)),
                                    columns=['day_since100','case_cnt'])
    
    validcases = validcases[:train_len + pred_len]

    N = int(float(data.population.replace(',','')))

    PopulationI = validcases.loc[0, "case_cnt"]
    PopulationD = validcases.loc[0, "death_cnt"] if data.cumulative_death_number is not None else int(data.pandemic_meta_data['mortality_rate'] * data.cumulative_case_number[0])

    R_0 = PopulationD * 5 if PopulationI - PopulationD > PopulationD * 5 else 0

    R_upperbound = validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"] if data.cumulative_death_number is not None else PopulationI - PopulationD
    R_heuristic = 10

    GLOBAL_PARAMS_FIXED = (N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v)

    x_0_cases = get_initial_conditions(
            params_fitted = delphi_params,
            global_params_fixed = GLOBAL_PARAMS_FIXED,
        )

    x_sol = solve_ivp(
        fun = DELPHI_model,
        y0 = x_0_cases,
        t_span = [t_predictions[0], t_predictions[-1]],
        t_eval = t_predictions,
        args = tuple(delphi_params + [N])
    )

    return x_sol.y[15]

if __name__ == '__main__':
    target_pandemic_data = process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_covid_data_objects.pickle',
                                        raw_data=False)

    sampling_from_distribution(target_pandemic_data,
                            sample_size=5,
                            sigma = 0.01,
                            save_plot=True)