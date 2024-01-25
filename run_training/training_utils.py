import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import numpy as np
from model.deep_learning_model_for_parameter_estimation import naive_nn
from datetime import datetime
import torch, torch.nn as nn
from model.Compartment_Pytorch_Models import DELPHI_pytorch
from torchdiffeq import odeint as odeint
from torchmetrics.regression import MeanAbsolutePercentageError as mape

from model.delphi_default_parameters import (
    default_parameter_list, 
    perfect_washington_parameter_list,
    perfect_washington_bounds_params,
    default_bounds_params, 
    default_maxT,
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

###########################################
## Training Utils for Compartment Models ##
###########################################
def create_fitting_data_from_validcases(validcases: pd.DataFrame) -> (float, float, list, list, list):
    """
    Creates the balancing coefficient (regularization coefficient between cases & deaths in cost function) as well as
    the cases and deaths data on which to be fitted
    :param validcases: Dataframe containing cases and deaths data on the relevant time period for our optimization
    :return: the balancing coefficient and two lists containing cases and deaths over the right time span for fitting
    """
    validcases_nondeath = validcases["case_cnt"].tolist()
    validcases_death = validcases["death_cnt"].tolist()
    # balance = validcases_nondeath[-1] / max(validcases_death[-1], 10)
    cases_data_fit = validcases_nondeath
    deaths_data_fit = validcases_death
    weights = list(range(1, len(cases_data_fit) + 1))
    # weights = [(x/len(cases_data_fit))**2 for x in weights]
    balance = np.average(cases_data_fit, weights = weights) / max(np.average(deaths_data_fit, weights = weights), 10)
    balance_total_difference = np.average(np.abs(np.array(cases_data_fit[7:])-np.array(cases_data_fit[:-7])), weights = weights[7:]) / np.average(np.abs(cases_data_fit), weights = weights)
    return balance, balance_total_difference, cases_data_fit, deaths_data_fit, weights

def get_initial_conditions(params_fitted: tuple, global_params_fixed: tuple) -> list:
    """
    Generates the initial conditions for the DELPHI model based on global fixed parameters (mostly populations and some
    constant rates) and fitted parameters (the internal parameters k1 and k2)
    :param params_fitted: tuple of parameters being fitted, mostly interested in k1 and k2 here (parameters 7 and 8)
    :param global_params_fixed: tuple of fixed and constant parameters for the model defined a while ago
    :return: a list of initial conditions for all 16 states of the DELPHI model
    """
    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = params_fitted 
    N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v = global_params_fixed

    PopulationR = min(R_upperbound - 1, min(int(R_0*p_d), R_heuristic))
    PopulationCI = (PopulationI - PopulationD - PopulationR)*k3

    S_0 = (
        (N - PopulationCI / p_d)
        - (PopulationCI / p_d * (k1 + k2))
        - (PopulationR / p_d)
        - (PopulationD / p_d)
    )
    E_0 = PopulationCI / p_d * k1
    I_0 = PopulationCI / p_d * k2
    UR_0 = (PopulationCI / p_d - PopulationCI) * (1 - p_dth)
    DHR_0 = (PopulationCI * p_h) * (1 - p_dth)
    DQR_0 = PopulationCI * (1 - p_h) * (1 - p_dth)
    UD_0 = (PopulationCI / p_d - PopulationCI) * p_dth
    DHD_0 = PopulationCI * p_h * p_dth
    DQD_0 = PopulationCI * (1 - p_h) * p_dth
    R_0 = PopulationR / p_d
    D_0 = PopulationD / p_d
    TH_0 = PopulationCI * p_h
    DVR_0 = (PopulationCI * p_h * p_v) * (1 - p_dth)
    DVD_0 = (PopulationCI * p_h * p_v) * p_dth
    DD_0 = PopulationD
    DT_0 = PopulationI
    x_0_cases = [
        S_0, E_0, I_0, UR_0, DHR_0, DQR_0, UD_0, DHD_0, DQD_0, R_0,
        D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0,
    ]
    return x_0_cases

def get_residuals_value(
        optimizer: str, balance: float, x_sol: list, cases_data_fit: list, deaths_data_fit: list, weights: list, balance_total_difference: float
) -> float:
    """
    Obtain the value of the loss function depending on the optimizer (as it is different for global optimization using
    simulated annealing)
    :param optimizer: String, for now either tnc, trust-constr or annealing
    :param balance: Regularization coefficient between cases and deaths
    :param x_sol: Solution previously fitted by the optimizer containing fitted values for all 16 states
    :param fitcasend: cases data to be fitted on
    :param deaths_data_fit: deaths data to be fitted on
    :param weights: time-related weights to give more importance to recent data points in the fit (in the loss function)
    :return: float, corresponding to the value of the loss function
    """
    if optimizer in ["trust-constr"]:
        residuals_value = sum(
            np.multiply((x_sol[15, :] - cases_data_fit) ** 2, weights)
            + balance
            * balance
            * np.multiply((x_sol[14, :] - deaths_data_fit) ** 2, weights)
        )
    elif optimizer in ["tnc", "annealing"]:
        residuals_value =  sum(      
            np.multiply(
                (x_sol[15, 7:] - x_sol[15, :-7] - cases_data_fit[7:] + cases_data_fit[:-7]) ** 2,
                weights[7:],
            )
            + balance * balance * np.multiply(
                (x_sol[14, 7:] - x_sol[14, :-7] - deaths_data_fit[7:] + deaths_data_fit[:-7]) ** 2,
                weights[7:],
            )
        ) + sum(
            np.multiply((x_sol[15, :] - cases_data_fit) ** 2, weights)
            + balance
            * balance
            * np.multiply((x_sol[14, :] - deaths_data_fit) ** 2, weights)
        ) * balance_total_difference * balance_total_difference
    else:
        raise ValueError("Optimizer not in 'tnc', 'trust-constr' or 'annealing' so not supported")

    return residuals_value

#################################################
## Training Utils for weight estimation Models ##
#################################################

def get_weight_estimation_model(model_name,
                                input_dim,
                                output_dim,
                                num_hidden = 2,
                                hidden_dim = 512):

    assert (model_name in ['Naive_nn']), "Provided weight estimation model is not supported"

    if model_name == 'Naive_nn':
        return naive_nn(input_dim=input_dim,
                        output_dim=output_dim,
                        num_hidden=num_hidden,
                        hidden_dim=hidden_dim)
    else:
        return -1

## TODO: Update Run_Compartment Model
def run_compartment_model(data,
                          predicted_edge_weights,
                          predicted_parameters, 
                          batch_size,
                          target_training_len = 30,
                          pred_len = 60,):

    loss_list = []
    loss_function = mape()

    performance_df = []
    terrible_samples = []

    for i in range(batch_size):
         
        # How to define parameter bounds?
        # bounds_params = 

        edge_weights = predicted_edge_weights[i]

        parameter_list = predicted_parameters[i]
        start_date = data['start_date'][i]
        timestamps = data['timestamps'][i]

        cumulative_case_number = data['cumulative_case_number'][i]

        if data['cumulative_death_number'] is None:
            cumulative_death_number = None
        else:
            cumulative_death_number = data['cumulative_death_number'][i]

        N = data['population'][i]

        if cumulative_death_number is not None:
            validcases = pd.DataFrame(list(zip(np.arange(0,len(timestamps)), cumulative_case_number, cumulative_death_number)),
                                      columns=['day_since100','case_cnt','death_cnt'])
            
        else:
            validcases = pd.DataFrame(list(zip(np.arange(0,len(timestamps)), cumulative_case_number)),
                                      columns=['day_since100','case_cnt'])

        PopulationI = validcases.loc[0, "case_cnt"]

        if cumulative_death_number is not None:
            PopulationD = validcases.loc[0, "death_cnt"]
            R_0 = validcases.loc[0, "death_cnt"] * 5 if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]> validcases.loc[0, "death_cnt"] * 5 else 0

            R_upperbound = validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]
            # R_heuristic = cases_t_14days - deaths_t_9days
            R_heuristic = 10

        # TODO: What should we put here if no death data?
        else:
            PopulationD = 0
            R_0 = 0
            R_upperbound = validcases.loc[0, "case_cnt"]
            R_heuristic = 10

        # maxT = (default_maxT.date() - start_date).days + 1
        maxT = len(cumulative_case_number)
        t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
        # balance, balance_total_difference, cases_data_fit, deaths_data_fit, weights = create_fitting_data_from_validcases(validcases)

        GLOBAL_PARAMS_FIXED = (N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v)

        t_predictions = [i for i in range(pred_len + target_training_len)]

        def solve_best_params_and_predict(optimal_params, t_predictions):
            # Variables Initialization for the ODE system
            alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = optimal_params
            optimal_params = [
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

            x_0_cases = get_initial_conditions(
                params_fitted=optimal_params,
                global_params_fixed=GLOBAL_PARAMS_FIXED,
            )

            delphi_model = DELPHI_pytorch(optimal_params,
                                        N = N)

            x_0_cases = torch.tensor(x_0_cases)
            t_predictions = torch.tensor(t_predictions).float()

            sol = odeint(delphi_model,
                        x_0_cases,
                        t_predictions,
                        method='dopri5')

            return sol
       
        x_sol = solve_best_params_and_predict(parameter_list, t_predictions=t_predictions)

        pred_case = x_sol[target_training_len:,15]
        # pred_death = x_sol[target_training_len:,14]

        loss = loss_function(pred_case, torch.tensor(cumulative_case_number[target_training_len:target_training_len+pred_len]))

        performance_df.append([data['pandemic_name'][i], loss.item()])

        if loss.item() > 100:
            terrible_samples.append(data)

        loss_list.append(loss)

    avg_loss = torch.mean(torch.stack(loss_list), dim=0)

    performance_df = pd.DataFrame(performance_df, columns=['Pandemic_Name','Loss'])

    performance_df = (performance_df.groupby(['Pandemic_Name'])
                      .agg([('Average_Loss','mean'),('Count', 'count')])
                      .reset_index())

    print(performance_df)

    return avg_loss, terrible_samples