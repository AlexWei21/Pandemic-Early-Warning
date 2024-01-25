import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class naive_nn(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden = 2, hidden_dim = 512):
        super().__init__()

        modules = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]

        for i in range(num_hidden - 1):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.Sigmoid())

        # self.hidden1 = nn.Linear(input_dim, hidden_dim)
        # self.act1 = nn.Sigmoid()
        # self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        # self.act2 = nn.Sigmoid()
        
        modules.append(nn.Linear(hidden_dim,output_dim))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):

        x = self.layers(x)
        
        return x 

# model = naive_nn(input_dim=43,
#                  output_dim=8)

# print(model)