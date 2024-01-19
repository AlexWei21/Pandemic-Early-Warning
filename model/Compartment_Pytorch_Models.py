import torch, torch.nn as nn
import math

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

class DELPHI_pytorch(nn.Module):

    def __init__(self, model_params, N):
        super().__init__()

        self.model_params = nn.Parameter(torch.tensor(model_params))
        self.N = N

    def forward(self,
                t: float,
                states: torch.TensorType,):
        
        alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3 = self.model_params

        r_i = torch.log(torch.tensor(2)) / IncubeD  # Rate of infection leaving incubation phase
        r_d = torch.log(torch.tensor(2)) / DetectD  # Rate of detection
        r_ri = torch.log(torch.tensor(2)) / RecoverID  # Rate of recovery not under infection
        r_rh = torch.log(torch.tensor(2)) / RecoverHD  # Rate of recovery under hospitalization
        r_rv = torch.log(torch.tensor(2)) / VentilatedD  # Rate of recovery under ventilation
        gamma_t = (
            (2 / torch.pi) * torch.arctan(-(t - days) / 20 * r_s) + 1
            + jump * torch.exp(-(t - t_jump) ** 2 / (2 * std_normal ** 2))
            )
        p_dth_mod = (2 / torch.pi) * (p_dth - 0.001) * (torch.arctan(-t / 20 * r_dthdecay) + torch.pi / 2) + 0.001
        
        assert (
            len(states) == 16
        ), f"Too many input variables, got {len(states)}, expected 16"

        S = states[..., 0]
        E = states[..., 1]
        I = states[..., 2]
        AR = states[..., 3]
        DHR = states[..., 4]
        DQR = states[..., 5]
        AD = states[..., 6]
        DHD = states[..., 7]
        DQD = states[..., 8]
        R = states[..., 9]
        D = states[..., 10]
        TH = states[..., 11]
        DVR = states[..., 12]
        DVD = states[..., 13]
        DD = states[..., 14]
        DT = states[..., 15]

        # Equations on main variables
        dSdt = -alpha * gamma_t * S * I / self.N
        dEdt = alpha * gamma_t * S * I / self.N - r_i * E
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

        dfunc = torch.zeros_like(states)
        dfunc[..., 0] = dSdt
        dfunc[..., 1] = dEdt
        dfunc[..., 2] = dIdt
        dfunc[..., 3] = dARdt
        dfunc[..., 4] = dDHRdt
        dfunc[..., 5] = dDQRdt
        dfunc[..., 6] = dADdt
        dfunc[..., 7] = dDHDdt
        dfunc[..., 8] = dDQDdt
        dfunc[..., 9] = dRdt
        dfunc[..., 10] = dDdt

        dfunc[..., 11] = dTHdt
        dfunc[..., 12] = dDVRdt
        dfunc[..., 13] = dDVDdt
        dfunc[..., 14] = dDDdt
        dfunc[..., 15] = dDTdt

        return dfunc
    
    def __repr__(self):
        return f"alpha: {self.model_params[0].item()}, \
            days: {self.model_params[1].item()}, \
                r_s : {self.model_params[2].item()}, \
                    r_dth: {self.model_params[3].item()}, \
                        p_dth: {self.model_params[4].item()}, \
                            r_dthdecay: {self.model_params[5].item()}, \
                                k1: {self.model_params[6].item()}, \
                                    k2: {self.model_params[7].item()}, \
                                        jump: {self.model_params[8].item()}, \
                                            t_jump: {self.model_params[9].item()}, \
                                                std_normal: {self.model_params[10].item()}, \
                                                    k3: {self.model_params[11].item()}"