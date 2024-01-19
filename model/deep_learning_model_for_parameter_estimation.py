import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class naive_nn(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden = 2, hidden_dim = 512):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim,output_dim)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.output(x)
        x = self.act_output(x)
        return x 

# model = naive_nn(input_dim=43,
#                  output_dim=8)

# print(model)