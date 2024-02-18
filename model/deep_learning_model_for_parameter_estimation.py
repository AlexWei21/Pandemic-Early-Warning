import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.Compartment_Pytorch_Models import DELPHI_pytorch
from torchdiffeq import odeint as odeint
# from scipy.integrate import solve_ivp
from utils.training_utils import get_initial_conditions
import random

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

class naive_nn(nn.Module):
    def __init__(self, input_dim, output_dim, device = 'cpu',
                 num_hidden = 2, hidden_dim = 512, 
                 pred_len = 60, target_training_len = 30,
                 dnn_output_range = None, output_dir = None,
                 batch_size = 64, population_normalization = True,):

        super().__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.rnn_layers = 5

        ## Create time series embedding layer
        self.lstm = nn.LSTM(input_size = target_training_len,
                            hidden_size = int(hidden_dim/2),
                            num_layers = self.rnn_layers,
                            batch_first = True)

        ## Create Meta-data embedding layer
        modules = []
        modules.append(nn.Linear(input_dim - target_training_len, int(hidden_dim/2)))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(int(hidden_dim/2), int(hidden_dim/2)))
        modules.append(nn.ReLU())

        self.meta_input_layer = nn.Sequential(*modules)

        modules = []

        for i in range(num_hidden):
            if i == (num_hidden - 1):
                modules.append(nn.Linear(hidden_dim,output_dim))
            else:
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.Tanh())
        
        self.readout = nn.Sequential(*modules)

        self.range_restriction = nn.Sigmoid()

        self.dnn_output_range = dnn_output_range

        self.t_predictions = [i for i in range(pred_len + target_training_len)]
        self.pred_len = pred_len
        self.target_training_len = target_training_len

        self.output_dir = output_dir
        self.device = device
        self.population_normalization = population_normalization

    def forward(self, x, sigmoid_scale = 1,):

        print(x['model_input'][:,:self.target_training_len])
        print(x['population'])

        if self.population_normalization:
            ts_input = x['model_input'][:,:self.target_training_len]
            ts_input = torch.div(ts_input, x['population'].unsqueeze(1)).to(self.device)
            print(ts_input)
        else:
            ts_input = x['model_input'][:,:self.target_training_len].to(self.device)
            
        meta_input = x['model_input'][:,self.target_training_len:].to(self.device)

        h0 = torch.randn(self.rnn_layers, len(x['model_input']), int(self.hidden_dim / 2)).cuda()
        c0 = torch.randn(self.rnn_layers, len(x['model_input']), int(self.hidden_dim / 2)).cuda()

        ts_input, h = self.lstm(ts_input.unsqueeze(1), (h0,c0))
        meta_input = self.meta_input_layer(meta_input)

        # print(ts_input)
        # print(meta_input)

        dnn_output = torch.cat((torch.squeeze(ts_input,1),meta_input), 1)

        # print(dnn_output)

        dnn_output = self.readout(dnn_output)

        # print(dnn_output)

        if self.dnn_output_range is not None:
            
            ## Scaling Sigmoid Function
            dnn_output = dnn_output * sigmoid_scale

            dnn_output = self.range_restriction(dnn_output)

            dnn_output = dnn_output * self.dnn_output_range.to(self.device)

        perfect_washington_parameter_list = [0.4142615697024255, 
                                             2.1279631272659053e-08,
                                             2.0628398549857805e-14, 
                                             0.959999999997383,
                                             0.001, 
                                             7.512244771221772e-08,
                                             0.0002969779597709444, 
                                             163.15756156175354,
                                             9.081779702773275e-11,
                                             141.4605182630534, 
                                             1,
                                             0.02796605040144473]

        t_predictions = torch.tensor(self.t_predictions).float().to(self.device)

        for i in range(len(dnn_output)):
            N = x['population'][i]

            PopulationD = int(x['pandemic_meta_data'][i]['mortality_rate'] * x['cumulative_case_number'][i][0])
            PopulationI = x['cumulative_case_number'][i][0]
            R_upperbound = PopulationI - PopulationD
            R_heuristic = 10
            R_0 = PopulationD * 5 if PopulationI - PopulationD > PopulationD * 5 else 0


            # print(dnn_output[i,11:])
            
            compartment_model = DELPHI_pytorch(dnn_output[i,11:],N)
            # compartment_model = DELPHI_pytorch(perfect_washington_parameter_list,N)

            # print((N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v))

            x_0_cases = get_initial_conditions(
                params_fitted = dnn_output[i,11:].tolist(),
                global_params_fixed=(N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v),
                # global_params_fixed = (N, 94, 10, 80, 16, 110, p_d, p_h, p_v),
                )

            x_0_cases = torch.tensor(x_0_cases).to(self.device)

            # print(x_0_cases)

            sol = odeint(compartment_model,
                            x_0_cases,
                            t_predictions,
                            method='dopri5')

            # print(sol)

            state_name = ['S','E','I','AR','DHR','DQR','AD','DHD','DQD','R','D','TH','DVR','DVD','DD','DT']

            x_0_cases_list = x_0_cases

            # with open(self.output_dir + f"compartment_model_train{self.target_training_len}_test{self.pred_len}_state_logs.txt", "w") as f:
            #     
            #     for state in range(len(x_0_cases_list)):
            #         f.write(str(state_name[state]))
            #         f.write(str(': '))
            #         f.write(str(x_0_cases_list[state]))
            #         f.write('\n')
                
            #     for stamp in sol.tolist():
            #         for state in range(len(stamp)):
            #             f.write(str(state_name[state]))
            #             f.write(str(': '))
            #             f.write(str(stamp[state]))
            #             f.write('\n')
            #         f.write('\n')


            if i == 0:
                sol_list = torch.unsqueeze(sol,0)
            else:
                sol_list = torch.cat((sol_list,torch.unsqueeze(sol,0)),0)

        return sol_list, dnn_output
