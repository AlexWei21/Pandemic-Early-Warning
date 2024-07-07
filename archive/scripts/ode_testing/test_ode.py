import numpy as np
from utils.training_utils import create_fitting_data_from_validcases, get_initial_conditions
from scipy.integrate import solve_ivp

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

def DELPHI_model(
                t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3        ) -> list:
       
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

N = 7029949

params = [0.9996851463475758,1.0714172563245972,0.8103914466106216,0.08836124097346854,0.13893279055875515,0.3554766503979124,0.1829384712428458,325.7774999023526,4.388315686084252,19.71896316591024,0.11532655753837417,0.005448978479350776]

alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = params
params = [
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

GLOBAL_PARAMS_FIXED = 7029949, 108.0, 10, 0.0, 0.0, 108.0, 0.2, 0.03, 0.25

x_0_cases = get_initial_conditions(
    params_fitted=params, global_params_fixed=GLOBAL_PARAMS_FIXED
    )
x_sol_total = solve_ivp(
      fun=DELPHI_model,
      y0=x_0_cases,
        t_span=[0,90],
        t_eval=[i for i in range(90)],
        args=tuple(params),
)

x_sol = x_sol_total.y

print(x_sol[15])