from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from model.model_for_predict_parameters_only import pandemic_early_warning_model
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError
from utils.logging import delphi_parameter_wise_loss_logging
import numpy as np
from utils.loss_fn import WMAE
from model.resnet18_1d import ResNet18

class TrainingModule(LightningModule):
    def __init__(self,
                 lr: float = 1e-5,
                 loss_name: str = 'MAE',
                 time_series_encoding: str = 'LSTM',
                 ts_population_normalization: bool = True,
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
                 device: str = 'cpu',
                 parameter_loss_weight: list = None,
                 log_parameter_loss: bool = False,
                 dropout: float = 0.0,
                 batch_size: int = 64,
                 past_pandemics: list = [],
                 normalize_label: bool = False,
                 # labels: dict = {}, ## For Debug Usage
                 ):
        
        super().__init__()

        self.lr = lr
        self.ts_dim = ts_dim
        self.ts_population_normalization = ts_population_normalization
        # self.labels = labels
        self.parameter_loss_weight = torch.tensor(parameter_loss_weight)
        self.loss_name = loss_name
        self.batch_size = batch_size
        self.past_pandemics = past_pandemics
        self.time_series_encoding = time_series_encoding

        self.log_parameter_loss = log_parameter_loss
        self.normalize_label = normalize_label

        if time_series_encoding == 'ResNet18':
            self.model = ResNet18(input_dim=1,
                                  output_dim=12,
                                  dropout_percentage=dropout)
        else:
            self.model = pandemic_early_warning_model(time_series_encoding=time_series_encoding,
                                                    num_ts_encoding_layer=num_ts_encoding_layer,
                                                    ts_dim=ts_dim,
                                                    include_meta_data=include_meta_data,
                                                    meta_data_encoding=meta_data_encoding,
                                                    num_meta_encoding_layer=num_meta_encoding_layer,
                                                    meta_data_dim=meta_data_dim,
                                                    readout_type=readout_type,
                                                    num_readout_layer=num_readout_layer,
                                                    output_dim=output_dim,
                                                    hidden_dim=hidden_dim,
                                                    dropout = dropout,)
            
        if loss_name == 'MAE':
            if parameter_loss_weight is not None:
                self.loss_fn = nn.L1Loss(reduction='none')
            else:
                self.loss_fn = nn.L1Loss()
        elif loss_name == 'MAPE':
            self.loss_fn = MeanAbsolutePercentageError()
        elif loss_name == 'WMAE':
            self.loss_fn = WMAE(output_weights=parameter_loss_weight)
        
        ## Validation Array Initialization
        self.validation_prediction = []
        self.validation_true_value = []

        ## Test Array Initialization
        self.test_prediction = []
        self.test_true_value = []

    def forward(self, batch):

        ts_input = batch['ts_input'].to(self.device)

        meta_input = batch['meta_input'].to(self.device)

        if self.time_series_encoding == 'ResNet18':
            full_input = torch.cat((ts_input,meta_input), dim = 1)
            full_input = full_input.unsqueeze(1)
            predicted_parameters = self.model(full_input)
        else:
            predicted_parameters = self.model(ts_input, meta_input)

        if self.normalize_label:
            predicted_parameters = nn.Sigmoid()(predicted_parameters)

        return predicted_parameters
    
    def loss(self, y_pred, y_true):

        if self.normalize_label:
            y_pred = torch.div(y_pred, self.parameter_loss_weight.to(y_pred))

        if self.loss_name == 'MAPE':
            loss = self.loss_fn(y_pred, y_true + 0.00001)
        elif self.loss_name == 'MAE':
            loss = self.loss_fn(y_pred, y_true)
        elif self.loss_name == 'WMAE':
            loss, unweighted_loss = self.loss_fn(y_pred, y_true)

        return loss
    
    def training_step(self, batch, batch_idx):

        y_true = batch['true_delphi_params']

        predicted_parameters = self.forward(batch).view(len(batch['domain_name']),-1)

        loss = self.loss(predicted_parameters, y_true)

        self.log('train_'+ self.loss_name , 
                 loss, 
                 on_epoch=True, 
                 batch_size = y_true.shape[0])

        print(f"Train_{self.loss_name}:", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        y_true = batch['true_delphi_params']

        predicted_parameters = self.forward(batch).view(len(batch['domain_name']),-1)

        # loss = self.loss(predicted_parameters, y_true)
        # self.log(f'validation_step_{self.loss_name}:', loss)

        self.validation_prediction.append(predicted_parameters)
        self.validation_true_value.append(y_true)

    
    def on_validation_epoch_end(self):

        all_preds = torch.cat(self.validation_prediction)
        all_true = torch.cat(self.validation_true_value)

        loss = self.loss(all_preds, all_true)

        print(f"validation_{self.loss_name}:", loss)

        self.log(f'validation_{self.loss_name}:', loss)

        self.validation_prediction.clear()
        self.validation_true_value.clear()

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)

        return optimizer