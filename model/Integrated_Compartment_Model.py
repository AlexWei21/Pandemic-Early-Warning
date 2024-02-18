from data.data_processing_compartment_model import get_data, get_population
from model.Compartment_Models import SEIRD, SEIRD_solver
from scipy.optimize import minimize
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle
import logging
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing

from utils.compartment_utils import plot_compartment
from utils.training_utils import create_fitting_data_from_validcases, get_initial_conditions, get_residuals_value

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

edge_weights =[1, 1, 1, 1, 1, 1, 1, 1, 1] ## Should be the same length as trainable parameters

def run_integrated(processed_data_path,country,domain = None,subdomain = None, past_parameters_ = None, train_length = 500):
    
    with open(processed_data_path, 'rb') as f:
        data_object_list = pickle.load(f)

    data_object_list = [x for x in data_object_list if x.country_name == country]

    if domain == None:
        data_object_list = [x for x in data_object_list if str(x.domain_name) == 'nan']
    else:
        data_object_list = [x for x in data_object_list if x.domain_name == domain]

    if subdomain == None:
        data_object_list = [x for x in data_object_list if pd.isnull(x.subdomain_name)]
    else:
        data_object_list = [x for x in data_object_list if x.subdomain_name == subdomain]

    data_object = data_object_list[0]

    parameter_list = perfect_washington_parameter_list
    bounds_params = perfect_washington_bounds_params
    start_date = pd.to_datetime(data_object.first_day_above_hundred)

    # print(parameter_list)
    # print(bounds_params)
    # print(start_date)

    validcases = pd.DataFrame(list(zip(np.arange(0,len(data_object.timestamps)), data_object.cumulative_case_number, data_object.cumulative_death_number)),
                              columns=['day_since100','case_cnt','death_cnt'])

    validcases = validcases[:train_length]

    ## TODO: get_population
    # PopulationT = data_object.country_meta_data['population']
    # PopulationT = 329500000
    PopulationT = 7694000
    OPTIMIZER = "trust-constr"
    ## OPTIMIZER = "annealing"

    N = PopulationT
    PopulationI = validcases.loc[0, "case_cnt"]
    PopulationD = validcases.loc[0, "death_cnt"]
    R_0 = validcases.loc[0, "death_cnt"] * 5 if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]> validcases.loc[0, "death_cnt"] * 5 else 0
    bounds_params_list = list(bounds_params)
    # bounds_params_list[-1] = (0.999,1)
    bounds_params = tuple(bounds_params_list)

    # cases_t_14days = totalcases[totalcases.date >= str(start_date- pd.Timedelta(14, 'D'))]['case_cnt'].values[0]
    # deaths_t_9days = totalcases[totalcases.date >= str(start_date - pd.Timedelta(9, 'D'))]['death_cnt'].values[0]
    R_upperbound = validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]
    # R_heuristic = cases_t_14days - deaths_t_9days
    R_heuristic = 10
    
    if int(R_0*p_d) >= R_upperbound and R_heuristic >= R_upperbound:
        logging.error(f"Initial conditions for PopulationR too high for {country}-{domain}")

    maxT = (default_maxT - start_date).days + 1
    t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
    balance, balance_total_difference, cases_data_fit, deaths_data_fit, weights = create_fitting_data_from_validcases(validcases)
    GLOBAL_PARAMS_FIXED = (N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v)

    def integrated_model(
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

        ### New States 
        # dIadt = r_i * E * r_asym - r_d * Ia
        # dImdt = r_i * E * (1 - r_asym) - r_d * Im

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

    def residuals_totalcases(params) -> float:
        """
        Function that makes sure the parameters are in the right range during the fitting process and computes
        the loss function depending on the optimizer that has been chosen for this run as a global variable
        :param params: currently fitted values of the parameters during the fitting process
        :return: the value of the loss function as a float that is optimized against (in our case, minimized)
        """
        # Variables Initialization for the ODE system
        alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = params
        # Force params values to stay in a certain range during the optimization process with re-initializations
        params = (
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
                    )

        x_0_cases = get_initial_conditions(
            params_fitted=params, global_params_fixed=GLOBAL_PARAMS_FIXED
            )
        x_sol_total = solve_ivp(
             fun=integrated_model,
             y0=x_0_cases,
             t_span=[t_cases[0], t_cases[-1]],
             t_eval=t_cases,
             args=tuple(params),
             )
        x_sol = x_sol_total.y
        # weights = list(range(1, len(cases_data_fit) + 1))
        # weights = [(x/len(cases_data_fit))**2 for x in weights]

        if x_sol_total.status == 0:
            residuals_value = get_residuals_value(
                optimizer=OPTIMIZER,
                balance=balance,
                x_sol=x_sol,
                cases_data_fit=cases_data_fit,
                deaths_data_fit=deaths_data_fit,
                weights=weights,
                balance_total_difference=balance_total_difference 
            )
        else:
            residuals_value = 1e16
        
        return residuals_value

    if OPTIMIZER in ["tnc", "trust-constr"]:
        output = minimize(
                residuals_totalcases,
                parameter_list,
                method=OPTIMIZER,
                bounds=bounds_params,
                options={"maxiter": max_iter},
            )
    elif OPTIMIZER == "annealing":
        output = dual_annealing(
            residuals_totalcases, x0=parameter_list, bounds=bounds_params
            )
        print(f"Parameter bounds are {bounds_params}")
        print(f"Parameter list is {parameter_list}")
    else:
        raise ValueError("Optimizer not in 'tnc', 'trust-constr' or 'annealing' so not supported")

    if (OPTIMIZER in ["tnc", "trust-constr"]) or (OPTIMIZER == "annealing" and output.success):
        best_params = output.x
        t_predictions = [i for i in range(maxT)]
    
        def solve_best_params_and_predict(optimal_params):
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
            x_sol_best = solve_ivp(
                fun=integrated_model,
                y0=x_0_cases,
                t_span=[t_predictions[0], t_predictions[-1]],
                t_eval=t_predictions,
                args=tuple(optimal_params),
            ).y
            
            return x_sol_best
        
        x_sol_final = solve_best_params_and_predict(best_params)

        np.save('x_sol_final.npy',np.array(x_sol_final, dtype=object), allow_pickle=True)
        np.save('true_case.npy', np.array(data_object.cumulative_case_number[:maxT]), allow_pickle=True)
        np.save('true_death.npy', np.array(data_object.cumulative_death_number[:maxT]), allow_pickle=True)


run_integrated(processed_data_path='/Users/alex/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_covid_data_objects.pickle',
           country = "United States",
           domain='Washington')