import torch
import torchode as to
import numpy as np
import matplotlib.pyplot as plt
from utils.training_utils import get_initial_conditions
from scipy.integrate import solve_ivp

from model.delphi_default_parameters import (
    p_v,
    p_d,
    p_h,
    IncubeD,
    DetectD,
    RecoverID,
    RecoverHD,
    VentilatedD,
    dict_default_reinit_parameters,
    dict_default_reinit_lower_bounds)

def DELPHI_model(t, x, args):
    
    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = args[0]
    N = args[1]

    r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
    r_d = np.log(2) / DetectD  # Rate of detection
    r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
    r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
    r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
    gamma_t = (
        (2 / torch.pi) * torch.arctan(-(t - days) / 20 * r_s) + 1
        + jump * torch.exp(-(t - t_jump) ** 2 / (2 * std_normal ** 2))
        )
        
    p_dth_mod = (2 / torch.pi) * (p_dth - 0.001) * (torch.arctan(-t / 20 * r_dthdecay) + torch.pi / 2) + 0.001

    x = x.t()

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

    return torch.stack((dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt,
        dDQDdt, dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt), dim = 1)

term = to.ODETerm(DELPHI_model, with_args=True)
step_method = to.Tsit5(term=term)
step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
solver = to.AutoDiffAdjoint(step_method, step_size_controller)

gp = [[7705247, 94.0, 10, 80.0, 16.0, 110.0, 0.2, 0.03, 0.25],
      [7029949, 108.0, 10, 0.0, 0.0, 108.0, 0.2, 0.03, 0.25]]

y0 = [[],
      []]

args = [[0.649687487743167,0.4624400134967142,0.47758990199391455,0.10494688074832446,0.10485288376582451,0.5131378137360779,0.00016189674924008504,209.6828346901557,2.0835045763508333,76.09967159838479,1.0351199245150966,0.039545911591631046],
        [0.9996851463475758,1.0714172563245972,0.8103914466106216,0.08836124097346854,0.13893279055875515,0.3554766503979124,0.1829384712428458,325.7774999023526,4.388315686084252,19.71896316591024,0.11532655753837417,0.005448978479350776]]

args = torch.tensor(args).t()

alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = args

args = torch.stack((
    torch.max(torch.stack((alpha,torch.tensor([dict_default_reinit_parameters["alpha"]] * len(gp)))), dim = 0).values,
    days,
    torch.max(torch.stack((r_s,torch.tensor([dict_default_reinit_parameters["r_s"]] * len(gp)))), dim = 0).values,
    torch.max(torch.stack((torch.min(torch.stack((r_dth,torch.tensor([1]*len(gp)))), dim = 0).values,torch.tensor([dict_default_reinit_parameters["r_dth"]] * len(gp)))), dim = 0).values,
    torch.max(torch.stack((torch.min(torch.stack((p_dth,torch.tensor([1]*len(gp)))), dim = 0).values,torch.tensor([dict_default_reinit_parameters["p_dth"]] * len(gp)))), dim = 0).values,
    torch.max(torch.stack((r_dthdecay,torch.tensor([dict_default_reinit_parameters["r_dthdecay"]] * len(gp)))), dim = 0).values,
    torch.max(torch.stack((k1,torch.tensor([dict_default_reinit_parameters["k1"]] * len(gp)))), dim = 0).values,
    torch.max(torch.stack((k2,torch.tensor([dict_default_reinit_parameters["k2"]] * len(gp)))), dim = 0).values,
    torch.max(torch.stack((jump,torch.tensor([dict_default_reinit_parameters["jump"]] * len(gp)))), dim = 0).values,
    torch.max(torch.stack((t_jump,torch.tensor([dict_default_reinit_parameters["t_jump"]] * len(gp)))), dim = 0).values,
    torch.max(torch.stack((std_normal,torch.tensor([dict_default_reinit_parameters["std_normal"]] * len(gp)))), dim = 0).values,
    torch.max(torch.stack((k3,torch.tensor([dict_default_reinit_lower_bounds["k3"]] * len(gp)))), dim = 0).values,
))

print(args.shape)

for i in range(len(gp)):
    y0[i] = get_initial_conditions(params_fitted=args[:,i], 
                                   global_params_fixed=gp[i])

# for i in range(len(gp)):
#     args[i] = args[i] + [gp[i][0]]

y0 = torch.tensor(y0)
N = torch.tensor([7705247,7029949])

t_eval = torch.stack((torch.linspace(0,90,90), torch.linspace(0,90,90)))

problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
sol = solver.solve(problem, args=[args, N])

print(sol.ys[1,:,15])

plt.plot(sol.ts[0],sol.ys[0,:,15])
plt.plot(sol.ts[1],sol.ys[1,:,15])
plt.show()

