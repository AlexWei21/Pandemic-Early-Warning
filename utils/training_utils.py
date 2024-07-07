import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import numpy as np
from datetime import datetime
import torch, torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError as mape


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
        optimizer: str, 
        x_sol: list, 
        cases_data_fit: list, 
        weights: list, 
        balance_total_difference: float,
        balance: float = None,
        deaths_data_fit: list = None, 
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
    if deaths_data_fit is not None:
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

    else: 
        if optimizer in ["trust-constr"]:
            residuals_value = sum(
                np.multiply((x_sol[15, :] - cases_data_fit) ** 2, weights)
            )
        elif optimizer in ["tnc", "annealing"]:

            residuals_value =  sum(      
                np.multiply(
                    (x_sol[15, 7:] - x_sol[15, :-7] - cases_data_fit[7:] + cases_data_fit[:-7]) ** 2,
                    weights[7:],
                )
            )
            + sum(
                np.multiply((x_sol[15, :] - cases_data_fit) ** 2, weights)
            ) * balance_total_difference * balance_total_difference 
        else:
            raise ValueError("Optimizer not in 'tnc', 'trust-constr' or 'annealing' so not supported")
        
    return residuals_value

