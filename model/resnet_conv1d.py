import numpy as np
import torch
import torch.nn as nn

class res_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim = None, linear_layer = False, dropout = 0.0):

        super().__init__()

        self.linear_layer = linear_layer

        self.res_unit1 = res_unit(input_dim=input_dim,
                                  hidden_dim=hidden_dim[0],
                                  output_dim=hidden_dim[0],
                                  dropout=dropout)
        
        self.res_unit2 = res_unit(input_dim=hidden_dim[0],
                                  hidden_dim=hidden_dim[1],
                                  output_dim=hidden_dim[1],
                                  dropout=dropout)
        
        self.res_unit3 = res_unit(input_dim=hidden_dim[1],
                                  hidden_dim=hidden_dim[2],
                                  output_dim=hidden_dim[2],
                                  dropout = dropout)

        if linear_layer:
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(hidden_dim[2], output_dim)

    def forward(self, x):

        x = self.res_unit1(x)

        x = self.res_unit2(x)

        x = self.res_unit3(x)

        if self.linear_layer:
            x = self.gap(x)
            x = torch.squeeze(x)
            x = self.fc(x)

        return x

class res_unit(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.0,
                 act_fn = nn.ReLU()):
        
        super().__init__()
        
        # self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(in_channels= input_dim,
                               out_channels = hidden_dim,
                               kernel_size = 3,
                               padding='same')

        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)     

        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels= hidden_dim,
                               out_channels = hidden_dim,
                               kernel_size = 3,
                               padding='same')

        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)

        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels= hidden_dim,
                               out_channels = output_dim,
                               kernel_size = 3,
                               padding='same')
        
        self.batchnorm3 = nn.BatchNorm1d(output_dim)

        self.act_out = nn.ReLU()

        if input_dim != output_dim:
            self.adjust = nn.Sequential(*[nn.Conv1d(in_channels=input_dim,
                                                  out_channels=output_dim,
                                                  kernel_size=1,),
                                         nn.BatchNorm1d(output_dim),
            ])
        else:
            self.shortcut_bn = nn.BatchNorm1d(input_dim)

    def forward(self, x):

        if self.input_dim != self.output_dim:
            in_x = self.adjust(x)
        else:
            in_x = self.shortcut_bn(x)

        x = self.conv1(x)
        if x.shape[0] > 2:
            x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        if x.shape[0] > 2:
            x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        if x.shape[0] > 2:
            x = self.batchnorm3(x)

        x = self.act_out(x + in_x)
        x = self.dropout(x)

        return x 
    
# model = res_net(input_dim=30,
#                 output_dim=12,
#                 hidden_dim=[64,64,64],
#                 linear_layer = False)

# x = torch.randn((64,30,1))

# print(model(x).shape)