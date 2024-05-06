import torch
import torch.nn as nn
import torch.optim as optim
import torchode as to
import numpy as np
from model.Compartment_Pytorch_Models import DELPHI_pytorch
from torchdiffeq import odeint as odeint
from utils.training_utils import get_initial_conditions
import random
from model.resnet_conv1d import res_net
import pytorch_lightning as lightning
from model.resnet18_1d import ResNet18
from model.delphi_default_parameters import default_bounds_params

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


class pandemic_early_warning_model_with_DELPHI(nn.Module):
    def __init__(self,
                 # ts_dim: int = 46,
                 pred_len: int = 71,
                 dropout: float = 0.5,
                 include_death: bool = True,):
        
        super().__init__()       

        self.parameter_prediction_layer = parameter_prediction_layer(dropout=dropout,
                                                                     include_death = include_death) 
        
        self.output_range = default_bounds_params
        self.output_min = torch.tensor([item[0] for item in default_bounds_params])
        self.output_max = torch.tensor([item[1] for item in default_bounds_params])

        self.range_restriction_function = nn.Sigmoid()

        self.delphi_layer = delphi_layer(pred_len=pred_len)

    def forward(self, ts_input, global_params_fixed):

        population = global_params_fixed[:,0]

        delphi_parameters = self.parameter_prediction_layer(ts_input)

        delphi_parameters = self.range_restriction_function(delphi_parameters) * self.output_max.to(delphi_parameters) + self.output_min.to(delphi_parameters)

        # print(delphi_parameters)

        output = self.delphi_layer(delphi_parameters,
                                   global_params_fixed,
                                   population,)

        return output

class parameter_prediction_layer(nn.Module):
    def __init__(self,
                 dropout: float = 0.5,
                 include_death: bool = True):
        
        super().__init__()

        input_dim = 2 if include_death else 1

        self.encoding_layer = ResNet18(input_dim= input_dim,
                                       output_dim=12,
                                       dropout_percentage=dropout,)

    def forward(self,
                time_series_x,):
        
        x = self.encoding_layer(time_series_x)

        return x
    
class delphi_layer(nn.Module):
    def __init__(self,
                 pred_len,
                 ):
        
        super().__init__()

        self.pred_len = pred_len

    def forward(self, 
               x,
               gp, 
               population):

        term = to.ODETerm(DELPHI_model, with_args=True)
        step_method = to.Tsit5(term=term)
        step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)

        y0 = [None] * x.shape[0]
    
        x = x.t()

        assert len(y0) == x.shape[1]

        for i in range(x.shape[1]):
            y0[i] = get_initial_conditions(params_fitted=x[:,i], 
                                           global_params_fixed=gp[i])
        
        y0 = torch.tensor(y0).to(x)

        N = population

        t_eval = torch.linspace(0,self.pred_len,self.pred_len).repeat(y0.shape[0],1).to(y0)

        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
        sol = solver.solve(problem, args=[x, N])

        return sol.ys


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

if __name__ == '__main__':
    model = pandemic_early_warning_model_with_DELPHI(pred_len=71,
                                                    dropout=0.5)

    print(model)

    ts_input = torch.randn((2,46))
    global_params_fixed = torch.tensor([[7705247, 94.0, 10, 80.0, 16.0, 110.0, 0.2, 0.03, 0.25],
                                        [7029949, 108.0, 10, 0.0, 0.0, 108.0, 0.2, 0.03, 0.25]])

    print(model(ts_input, global_params_fixed).shape)
    print(model(ts_input, global_params_fixed)[0,:,15])
    print(model(ts_input, global_params_fixed)[0,:,14])
