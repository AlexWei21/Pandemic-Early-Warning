import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle
import logging
from tqdm import tqdm
import multiprocessing
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing
from utils.training_utils import create_fitting_data_from_validcases, get_initial_conditions, get_residuals_value
from model.delphi_default_parameters import (
    perfect_washington_parameter_list,
    default_parameter_list,
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
    VentilatedD,
    default_bounds_params)

def get_perfect_parameters(data_object,
                           train_length,
                           test_length, ## Includes train_length
                           ):
    
    # bounds_params_name = [
    # 'alpha',
    # 'days',
    # 'r_s',
    # 'r_dth',
    # 'p_dth',
    # 'r_dthdecay',
    # 'k1',
    # 'k2',
    # 'jump',
    # 't_jump',
    # 'std_normal',
    # 'k3']

    # Perfect Bounds for UK (Early Model)
    # bounds_params = ((1,2),
    #                  (10,14),
    #                  (1,2),
    #                  (8,9),
    #                  (0.2,0.3),
    #                  (0,0.0001), # r_dthdecay
    #                  (6,7),
    #                  (0.06,0.07),
    #                  (0,0.0001), # jump
    #                  (0,300), # t_jump
    #                  (0.1,100), # std_normal
    #                  (0.999,1) #k3
    #                  )

    # bounds_params = ((0,3.0),
    #                    (0,2.0), 
    #                    (0,3.0), 
    #                    (0,1.0), 
    #                     (0, 1.0), 
    #                    (0,4.5), 
    #                    (0,10), 
    #                    (0,10.0), # 454 
    #                    (0,8.22), 
    #                    (0,250.0), 
    #                    (0,100.0), 
    #                    (0.2,2.0))

    ## Perfect Bounds for UK
    # bounds_params = ((2.5,2.6),
    #                  (1.1,1.2),
    #                  (2,3),
    #                  (0.02,0.06),
    #                  (0.5,1.0),
    #                  (1.5,2.0),
    #                  (8.5,9.5),
    #                  (0.75,1.25),
    #                  (0.2,0.3),
    #                  (100,200),
    #                  (50,60),
    #                  (0.999,1))
    
    bounds_params = default_bounds_params
    # bounds_params_list = list(bounds_params)
    # bounds_params_list[3] = (0.02,10) # r_dth
    # bounds_params_list[5] = (0,1e-16) # r_dthdecay
    # bounds_params_list[8] = (0,1e-16) # jump
    # bounds_params_list[11] = (0.999, 1) # k3
    # bounds_params = tuple(bounds_params_list)
    
    start_date = pd.to_datetime(data_object.first_day_above_hundred)

    if data_object.cumulative_death_number is not None:
        validcases = pd.DataFrame(list(zip(np.arange(0,len(data_object.timestamps)), data_object.cumulative_case_number, data_object.cumulative_death_number)),
                                columns=['day_since100','case_cnt','death_cnt'])
    else: 
        validcases = pd.DataFrame(list(zip(np.arange(0,len(data_object.timestamps)), data_object.cumulative_case_number)),
                                columns=['day_since100','case_cnt'])
    
    validcases = validcases[:train_length]

    PopulationT = int(float(data_object.population.replace(',','')))

    OPTIMIZER = "annealing"

    N = PopulationT
    
    PopulationI = validcases.loc[0, "case_cnt"]
    PopulationD = validcases.loc[0, "death_cnt"] if data_object.cumulative_death_number is not None else int(data_object.pandemic_meta_data['mortality_rate'] * data_object.cumulative_case_number[0])

    R_0 = PopulationD * 5 if ((PopulationI - PopulationD) > (PopulationD * 5)) else 0
    
    bounds_params_list = list(bounds_params)
    # bounds_params_list[-1] = (0.999,1)
    bounds_params = tuple(bounds_params_list)

    R_upperbound = (validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]) if data_object.cumulative_death_number is not None else (PopulationI - PopulationD)
    R_heuristic = 10

    if int(R_0*p_d) >= R_upperbound and R_heuristic >= R_upperbound:
        logging.error(f"Initial conditions for PopulationR too high")

    maxT = test_length

    t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]

    if data_object.cumulative_death_number is not None:
        balance, balance_total_difference, cases_data_fit, deaths_data_fit, weights = create_fitting_data_from_validcases(validcases)
    else: 
        cases_data_fit = validcases['case_cnt'].to_list()
        weights = list(range(1, len(cases_data_fit) + 1))
        balance_total_difference = np.average(np.abs(np.array(cases_data_fit[7:])-np.array(cases_data_fit[:-7])), weights = weights[7:]) / np.average(np.abs(cases_data_fit), weights = weights)

    GLOBAL_PARAMS_FIXED = (N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v)
    print('Global_params_fixed:', GLOBAL_PARAMS_FIXED)

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
        
        # print(gamma_t)
        
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
             fun=DELPHI_model,
             y0=x_0_cases,
             t_span=[t_cases[0], t_cases[-1]],
             t_eval=t_cases,
             args=tuple(params),
             )
        x_sol = x_sol_total.y
        # weights = list(range(1, len(cases_data_fit) + 1))
        # weights = [(x/len(cases_data_fit))**2 for x in weights]

        ## Case Death Ratio
        # balance = 1

        if x_sol_total.status == 0:
            if data_object.cumulative_death_number is not None:
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
                residuals_value = get_residuals_value(
                    optimizer=OPTIMIZER,
                    x_sol=x_sol,
                    cases_data_fit=cases_data_fit,
                    weights=weights,
                    balance_total_difference=balance_total_difference,
                )

        else:
            residuals_value = 1e16
        
        return residuals_value

    # parameter_list = perfect_washington_parameter_list
    parameter_list = default_parameter_list

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

            param_dict = {'alpha': optimal_params[0],
                      'days': optimal_params[1],
                      'r_s': optimal_params[2],
                      'r_dth': optimal_params[3],
                      'p_dth': optimal_params[4],
                      'r_dthdecay': optimal_params[5],
                      'k1': optimal_params[6],
                      'k2': optimal_params[7],
                      'jump': optimal_params[8],
                      't_jump': optimal_params[9],
                      'std_normal': optimal_params[10],
                      'k3': optimal_params[11],}

            for key in param_dict:
                print(key, ": ", param_dict[key])

            x_0_cases = get_initial_conditions(
                params_fitted=optimal_params,
                global_params_fixed=GLOBAL_PARAMS_FIXED,
            )

            print("x_0_cases",x_0_cases)

            x_sol_best = solve_ivp(
                fun=DELPHI_model,
                y0=x_0_cases,
                t_span=[t_predictions[0], t_predictions[-1]],
                t_eval=t_predictions,
                args=tuple(optimal_params),
            ).y 
            
            return x_sol_best
        
        x_sol_final = solve_best_params_and_predict(best_params)


    return best_params, np.array(x_sol_final, dtype = object), np.array(data_object.cumulative_case_number[:maxT]), data_object

def visualize_result(pred_case, true_case, output_dir, data_object, type = 'case', train_len = 40,):

    case_loss = compute_mape(true_case[-15:],
                             pred_case[len(true_case)-15:len(true_case)])  

    full_case_loss = compute_mape(true_case,
                                  pred_case) 
    
    train_loss = compute_mape(true_case[:train_len],
                              pred_case[:train_len])
    
    case_mae = mean_absolute_error(true_case[-15:],
                             pred_case[len(true_case)-15:len(true_case)])  
    
    full_case_mae = mean_absolute_error(true_case,
                                  pred_case)
    
    train_mae = mean_absolute_error(true_case[:train_len],
                              pred_case[:train_len])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = np.arange(1,len(pred_case)),
                             y = true_case,
                             mode = 'markers',
                             name='Observed Infections',
                             line = dict(dash='dot')))
    
    fig.add_trace(go.Scatter(x = np.arange(1,len(pred_case)),
                             y = pred_case,
                             mode = 'markers',
                             name='Predicted Infections',
                             line = dict(dash='dot')))    
    
    fig.update_layout(title='DELPHI: Observed vs Fitted',
                       xaxis_title='Day',
                       yaxis_title='Counts',
                       title_x=0.5,
                       width=1000, height=600
                     )
    
    fig.write_image(output_dir + f'{data_object.country_name}_{data_object.domain_name}_{data_object.pandemic_name}_{len(true_case)}_{type}_prediction.png')   

    return case_loss, full_case_loss, train_loss, case_mae, full_case_mae, train_mae

def compute_mape(y_true: list, y_pred: list) -> float:
    """
    Compute the Mean Absolute Percentage Error (MAPE) between two lists of values
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a float corresponding to the MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred)[y_true > 0] / y_true[y_true > 0])) * 100
    
    return mape 

def generate_and_save_perfect_parameters(data_object):

    try:
        parameters, x_sol, y_true, data = get_perfect_parameters(data_object=data_object,
                                                                train_length=90,
                                                                test_length=90,)

        mape_loss = visualize_result(x_sol[15],
                                        y_true,
                                        output_dir = save_fig_dir,
                                        data_object=data)

    except:
        parameters = [-999] * 12
        mape_loss = 999

    return data_object.country_name, data_object.domain_name, parameters, mape_loss, data_object.first_day_above_hundred

if __name__ == '__main__':

    pandemic_name = 'covid'

    data_path = f"/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_{pandemic_name}_data_objects_no_smoothing.pickle"

    with open(data_path, 'rb') as f:
        data_object_list = pickle.load(f)

    parameter_list = []
    mape_list = []
    country_name_list = []
    domain_name_list = []
    year_list = []
    save_fig_dir = f"/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/DELPHI_{pandemic_name}_figures_46_71/"

    core_num = multiprocessing.cpu_count()

    print(f"Using {core_num} cores in running!")

    with multiprocessing.Pool(core_num) as pool:
        with tqdm(total=len(data_object_list)) as pbar:
            for country_name, domain_name, parameters, mape_loss, first_date in pool.imap_unordered(generate_and_save_perfect_parameters, data_object_list):
                parameter_list.append(parameters)   
                country_name_list.append(country_name)
                domain_name_list.append(domain_name)
                mape_list.append(mape_loss)
                year_list.append(pd.to_datetime(first_date).year)
                pbar.update()

    parameter_df = pd.DataFrame(parameter_list, columns=['alpha','days','r_s','r_dth','p_dth','r_dthdecay','k1','k2','jump','t_jump','std_normal','k3'])
    parameter_df['country'] = country_name_list
    parameter_df['domain'] = domain_name_list
    parameter_df['last_15_days_mape'] = mape_list
    parameter_df['year'] = year_list

    print(parameter_df)

    parameter_df.to_csv(f"DELPHI_params_{pandemic_name}.csv",
                        index = False)
    
        


    
    
