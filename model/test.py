import torchdiffeq as tde
import torchode as to
import torch
import torch.nn as nn

from model.Compartment_Pytorch_Models import DELPHI_pytorch

y0 = torch.randn((32,16))

print(y0.shape)

t_eval = torch.linspace(0,100,100)

params = torch.randn(12)

print(params.shape)

model = DELPHI_pytorch(params,N=10000)

sol_tde = tde.odeint(model,y0,t_eval)

print(sol_tde.shape)