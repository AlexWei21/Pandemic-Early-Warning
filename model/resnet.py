import torch.nn as nn
import torch

class res_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.0,
                 res_layer = 4, act_fn = nn.ReLU()):
        
        super().__init__()

        input_modules = []
        input_modules.append(nn.Linear(input_dim, hidden_dim))
        input_modules.append(act_fn)
        self.input_modules = nn.Sequential(*input_modules)

        res_modules = []
        for i in range(res_layer):
            res_modules.append(res_unit(input_dim=hidden_dim,
                                        hidden_dim=hidden_dim,
                                        output_dim=hidden_dim,
                                        dropout=dropout,
                                        act_fn=act_fn))
        self.res_modules = nn.Sequential(*res_modules)

        readout_modules = []
        readout_modules.append(nn.Linear(hidden_dim, hidden_dim))
        readout_modules.append(act_fn)
        readout_modules.append(nn.Linear(hidden_dim, output_dim))
        self.readout_modules = nn.Sequential(*readout_modules)

    def forward(self, x):

        x = self.input_modules(x)

        x = self.res_modules(x)

        x = self.readout_modules(x)

        return x

class res_unit(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.0,
                 act_fn = nn.ReLU()):
        
        super().__init__()
        
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.act_fn = act_fn

        self.fc1 = nn.Linear(input_dim,hidden_dim)

        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.batchnorm2 = nn.BatchNorm1d(output_dim)

        self.dropout1 = nn.Dropout(p=self.dropout)

        if input_dim != output_dim:
            self.adjust = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        if self.input_dim != self.output_dim:
            in_x = self.adjust(x)
        else:
            in_x = x

        x = self.fc1(x)
        if x.shape[0] > 1:
            x = self.batchnorm1(x)
        x = self.act_fn(x)

        x = self.fc2(x)
        if x.shape[0] > 1:
            x = self.batchnorm2(x)

        x = self.dropout1(x)

        x = self.act_fn(x + in_x)

        return x 

# model = res_net(input_dim=76,
#                  hidden_dim=64,
#                  output_dim=25)

# x = torch.randn(128,76)

# output = model(x)

# print(model)

# print(output.shape)
