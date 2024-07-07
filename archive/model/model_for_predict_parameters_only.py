import pytorch_lightning as lightning
import torch
import torch.nn as nn
from model.resnet_conv1d import res_net

class pandemic_early_warning_model(nn.Module):
    def __init__(self,
                 time_series_encoding: str = 'LSTM',
                 num_ts_encoding_layer: int = 5,
                 ts_dim: int = 30,
                 include_meta_data: bool = False,
                 meta_data_encoding: str = None,
                 num_meta_encoding_layer: int = 5,
                 meta_data_dim: int = 27,
                 readout_type: str = 'ResNet',
                 num_readout_layer: int = 5,
                 output_dim: int = 12,
                 hidden_dim: int = 256,
                 dropout: float = 0.0,
                 ):
        
        super().__init__()       

        self.include_meta_data = include_meta_data

        if include_meta_data:
            # self.meta_data_encoding_layer = meta_data_encoding_layer(meta_data_encoding=meta_data_encoding,
            #                                                          num_meta_encoding_layer=num_meta_encoding_layer,
            #                                                          input_dim=meta_data_dim,
            #                                                          hidden_dim= int(hidden_dim/2),)
            
            ### Directly add meta data into convolutional network
            self.time_series_encoding_layer = ts_encoding_layer(time_series_encoding=time_series_encoding,
                                                                num_ts_encoding_layer=num_ts_encoding_layer,
                                                                input_dim=ts_dim + meta_data_dim,
                                                                hidden_dim= hidden_dim,
                                                                dropout=dropout,
                                                                ) 
        else:
            self.time_series_encoding_layer = ts_encoding_layer(time_series_encoding=time_series_encoding,
                                                                num_ts_encoding_layer=num_ts_encoding_layer,
                                                                input_dim=ts_dim,
                                                                hidden_dim=hidden_dim,
                                                                dropout=dropout,) 

        self.readout_layer = readout_layer(readout_type=readout_type,
                                           num_readout_layer=num_readout_layer,
                                           hidden_dim = hidden_dim,
                                           output_dim = output_dim,
                                           dropout=dropout,)

    def forward(self, ts_input, meta_input,):

        if self.include_meta_data:
            embedding_output = self.time_series_encoding_layer(torch.cat((ts_input,meta_input), dim = 1))
        else:
            embedding_output = self.time_series_encoding_layer(ts_input)
        
        output = self.readout_layer(embedding_output)

        return output

class ts_encoding_layer(nn.Module):
    def __init__(self,
                 time_series_encoding: str = 'LSTM',
                 num_ts_encoding_layer: int = 5,
                 input_dim: int = 30,
                 hidden_dim: int = 256,
                 dropout: float = 0.0,):
        
        super().__init__()

        self.time_series_encoding = time_series_encoding
        self.num_ts_encoding_layer = num_ts_encoding_layer
        self.hidden_dim = hidden_dim

        if time_series_encoding == 'ResNet':
            self.encoding_layer = res_net(input_dim=1,
                                          hidden_dim=[64,64,64],
                                          linear_layer=False,
                                          dropout=dropout)
        elif time_series_encoding == 'LSTM':
            self.encoding_layer = nn.LSTM(input_size=input_dim,
                                          hidden_size=hidden_dim,
                                          num_layers=num_ts_encoding_layer,
                                          batch_first=True)
        elif time_series_encoding == 'Linear':
            module = []
            module.append(nn.Linear(input_dim, hidden_dim))
            module.append(nn.ReLU())

            for i in range(num_ts_encoding_layer - 1):
                module.append(nn.Linear(hidden_dim, hidden_dim))
                module.append(nn.ReLU())
            
            self.encoding_layer = nn.Sequential(*module)

    def forward(self,
                time_series_x,):

        if self.time_series_encoding == 'LSTM':
            h0 = torch.randn(self.num_ts_encoding_layer, len(time_series_x), self.hidden_dim).to(time_series_x)
            c0 = torch.randn(self.num_ts_encoding_layer, len(time_series_x), self.hidden_dim).to(time_series_x)
            x, h = self.encoding_layer(time_series_x.unsqueeze(1), (h0,c0))
        if self.time_series_encoding == 'ResNet':
            time_series_x = torch.unsqueeze(time_series_x,dim=1)
            x = self.encoding_layer(time_series_x)
        else:
            x = self.encoding_layer(time_series_x)

        return x

    
class meta_data_encoding_layer(nn.Module):
    def __init__(self,
                 meta_data_encoding: str = None,
                 num_meta_encoding_layer: int = 5,
                 input_dim: int = 27,
                 hidden_dim: int = 256,):
        
        super().__init__()

        if meta_data_encoding == 'Linear':
            module = []
            module.append(nn.Linear(input_dim, hidden_dim))
            module.append(nn.ReLU())

            for i in range(num_meta_encoding_layer - 1):
                module.append(nn.Linear(hidden_dim, hidden_dim))
                module.append(nn.ReLU())
            
            self.encoding_layer = nn.Sequential(*module)

    def forward(self,
                meta_data_x,):
        
        x = self.encoding_layer(meta_data_x)

        return x 
        
class readout_layer(nn.Module):
    def __init__(self,
                 readout_type: str = 'ResNet',
                 num_readout_layer: int = 5,
                 hidden_dim: int = 256,
                 output_dim: int = 12,
                 dropout: float = 0.0,):
        
        super().__init__()

        if readout_type == 'Linear':
            module = []

            for i in range(num_readout_layer - 1):
                module.append(nn.Linear(hidden_dim, hidden_dim))
                module.append(nn.ReLU())

            module.append(nn.Linear(hidden_dim, output_dim))
            
            self.readout_layer = nn.Sequential(*module)
        
        elif readout_type == 'ResNet':
            self.readout_layer = res_net(input_dim=64,
                                         hidden_dim=[128,128,128],
                                         output_dim=output_dim,
                                         linear_layer=True,
                                         dropout=dropout,)
        
    def forward(self,
                x,):
        
        x = self.readout_layer(x)

        return x
    

# model = pandemic_early_warning_model(time_series_encoding = 'Linear',
#                                      num_ts_encoding_layer = 5,
#                                      ts_dim = 30,
#                                      include_meta_data = True,
#                                      meta_data_encoding = 'Linear',
#                                      num_meta_encoding_layer = 5,
#                                      meta_data_dim = 27,
#                                      readout_type = 'Linear',
#                                      num_readout_layer = 5,
#                                      output_dim = 12,
#                                      hidden_dim = 256,)

# print(model)

# ts_input = torch.randn((64,30))
# meta_input = torch.randn((64,27))

# print(model(ts_input, meta_input).shape)
