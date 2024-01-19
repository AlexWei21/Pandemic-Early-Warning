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
def run_compartment_model(predicted_edge_weights,
                          predicted_parameters, 
                          start_date_list,
                          compartment_model,
                          cumulative_case_number_list,
                          population_list,
                          batch_size,
                          time_stamps_list,
                          cumulative_death_number_list = None,):

    loss_list = []
    loss_function = mape()

    for i in range(batch_size):
         
        # How to define parameter bounds?
        # bounds_params = 

        edge_weights = predicted_edge_weights[i]

        parameter_list = predicted_parameters[i]
        start_date = start_date_list[i]
        timestamps = time_stamps_list[i]

        cumulative_case_number = cumulative_case_number_list[i]

        if cumulative_death_number_list is None:
            cumulative_death_number = None
        else:
            cumulative_death_number = cumulative_death_number_list[i]

        N = population_list[i]

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
                
        def DELPHI_model(
                    t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3
                ) -> list:
            """
            SEIR based model with 16 distinct states, taking into account undetected, deaths, hospitalized and
            recovered, and using an ArcTan government response curve, corrected with a Gaussian jump in case of
            a resurgence in cases
            :param t: time step
            :param x: set of all the states in the model (here, 16 of them)
            :param alpha: Infection rate
            :param days: Median day of action (used in the arctan governmental response)
            :param r_s: Median rate of action (used in the arctan governmental response)
            :param r_dth: Rate of death
            :param p_dth: Initial mortality percentage
            :param r_dthdecay: Rate of decay of mortality percentage
            :param k1: Internal parameter 1 (used for initial conditions)
            :param k2: Internal parameter 2 (used for initial conditions)
            :param jump: Amplitude of the Gaussian jump modeling the resurgence in cases
            :param t_jump: Time where the Gaussian jump will reach its maximum value
            :param std_normal: Standard Deviation of the Gaussian jump (~ time span of the resurgence in cases)
            :param k3: Internal parameter 2 (used for initial conditions)
            :return: predictions for all 16 states, which are the following
            [0 S, 1 E, 2 I, 3 UR, 4 DHR, 5 DQR, 6 UD, 7 DHD, 8 DQD, 9 R, 10 D, 11 TH, 12 DVR,13 DVD, 14 DD, 15 DT]
            """
            r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
            r_d = np.log(2) / DetectD  # Rate of detection
            r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
            r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
            r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
            gamma_t = (
                (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1
                + jump * np.exp(-(t - t_jump) ** 2 / (2 * std_normal ** 2))
                )
            p_dth_mod = (2 / np.pi) * (p_dth - 0.001) * (np.arctan(-t / 20 * r_dthdecay) + np.pi / 2) + 0.001
            assert (
                len(x) == 16
            ), f"Too many input variables, got {len(x)}, expected 16"
            S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT = x
            # Equations on main variables
            dSdt = -alpha * gamma_t * S * I / N
            dEdt = alpha * gamma_t * S * I / N - r_i * E
            dIdt = r_i * E - r_d * I
            dARdt = r_d * (1 - p_dth_mod) * (1 - p_d) * I - r_ri * AR
            dDHRdt = r_d * (1 - p_dth_mod) * p_d * p_h * I - r_rh * DHR
            dDQRdt = r_d * (1 - p_dth_mod) * p_d * (1 - p_h) * I - r_ri * DQR
            dADdt = r_d * p_dth_mod * (1 - p_d) * I - r_dth * AD
            dDHDdt = r_d * p_dth_mod * p_d * p_h * I - r_dth * DHD
            dDQDdt = r_d * p_dth_mod * p_d * (1 - p_h) * I - r_dth * DQD
            dRdt = r_ri * (AR + DQR) + r_rh * DHR
            dDdt = r_dth * (AD + DQD + DHD)
            # Helper states (usually important for some kind of output)
            dTHdt = r_d * p_d * p_h * I
            dDVRdt = r_d * (1 - p_dth_mod) * p_d * p_h * p_v * I - r_rv * DVR
            dDVDdt = r_d * p_dth_mod * p_d * p_h * p_v * I - r_dth * DVD
            dDDdt = r_dth * (DHD + DQD)
            dDTdt = r_d * p_d * I
            return [
                dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt,
                dDQDdt, dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt,
            ]

        t_predictions = [i for i in range(maxT)]

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

            # x_sol_best = solve_ivp(
            #     fun=DELPHI_model,
            #     y0=x_0_cases,
            #     t_span=[t_predictions[0], t_predictions[-1]],
            #     t_eval=t_predictions,
            #     args=tuple(optimal_params),
            # ).y

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

        pred_case = x_sol[:,15]
        pred_death = x_sol[:,14]

        loss = loss_function(pred_case, torch.tensor(cumulative_case_number))

        loss_list.append(loss)

    avg_loss = torch.mean(torch.stack(loss_list), dim=0)

    return avg_loss