from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from model.model_for_predict_casenum_directly import pandemic_early_warning_model_with_DELPHI
from torchmetrics.regression import MeanAbsolutePercentageError
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from model.delphi_default_parameters import (
    perfect_washington_parameter_list,
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

class TrainingModule(LightningModule):
    def __init__(self,
                 lr: float = 1e-3,
                 loss: str = 'MAPE',
                 train_len: int = 46,
                 pred_len: int = 71,
                 dropout: float = 0.5,
                 include_death: bool = True,
                 ):
        
        super().__init__()

        self.lr = lr

        self.train_len = train_len
        self.pred_len = pred_len
        self.include_death = include_death

        self.p_d = p_d
        self.p_h = p_h
        self.p_v = p_v


        self.model = pandemic_early_warning_model_with_DELPHI(pred_len=pred_len,
                                                              dropout = dropout)
        
        if loss == 'MAPE':
            self.loss_fn = MeanAbsolutePercentageError()
        elif loss == 'MAE':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss == 'MSE':
            self.loss_fn = nn.MSELoss(reduction = 'none')

        self.loss_name = loss
        
        self.test_country = []
        self.test_domain = []
        self.test_case_prediction_list = []
        self.test_death_prediction_list = []
        self.test_case_true_list = []
        self.test_death_true_list = []

        self.validation_preds = []
        self.validation_batch = []

        self.train_case_pred = []
        self.train_death_pred = []

    def forward(self, batch):

        ts_case_input = batch['ts_case_input'].to(self.device)
        ts_case_input = ts_case_input.unsqueeze(1)

        if self.include_death:
            ts_death_input = batch['ts_death_input'].to(self.device)
            ts_death_input = ts_death_input.unsqueeze(1)

            ts_input = torch.cat((ts_case_input,ts_death_input), dim = 1)
        else:
            ts_input = ts_case_input

        N = batch['population']

        mortality_rate = torch.tensor([item['mortality_rate'] for item in batch['pandemic_meta_data']]).to(N)

        PopulationI = torch.tensor([item[0] for item in batch['cumulative_case_number']]).to(N)
        PopulationD = mortality_rate * PopulationI
        
        R_upperbound = PopulationI - PopulationD
        R_heuristic = torch.tensor([10] * len(N)).to(N)

        R_0 = torch.zeros(len(N)).to(N)
        for i in range(len(N)):
            R_0[i] = PopulationD[i] * 5 if PopulationI[i] - PopulationD[i] > PopulationD[i] * 5 else 0

        p_d = torch.tensor([self.p_d] * len(N)).to(N)
        p_h = torch.tensor([self.p_h] * len(N)).to(N)
        p_v = torch.tensor([self.p_v] * len(N)).to(N)

        global_params_fixed = torch.stack((N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_d, p_h, p_v)).t()

        predicted_casenum = self.model(ts_input,
                                       global_params_fixed)

        return predicted_casenum
    
    def loss(self, preds, batch, return_detailed = False):

        population = batch['population']
        predicted_case = preds[:,:,15]

        true_case = [item[:self.pred_len] for item in batch['cumulative_case_number']]
        true_case = torch.tensor(true_case).to(predicted_case)

        ## Calculate Weight for Case
        case_weights = list(range(1, self.pred_len + 1))
        case_weights = [case_weights] * true_case.shape[0] 
        case_weights = torch.tensor(case_weights).to(true_case)

        weighted_case = torch.mean(true_case[:,:self.train_len] * case_weights[:,:self.train_len],
                                   dim=1)
        
        # Balance Along Time Stamps
        case_loss = self.loss_fn(predicted_case, true_case) # shape: [10,71]
        detailed_case_loss = case_loss.tolist()
        case_loss = case_loss * case_weights # shape: [10,71]
        case_loss = torch.mean(case_loss,  # shape : [10]
                               dim = 1)

        ## Death Loss
        predicted_death = preds[:,:,14]

        true_death = [item[:self.pred_len] for item in batch['cumulative_death_number']]
        true_death = torch.tensor(true_death).to(predicted_death)

        ## Calculate Weight for Death
        death_weights = list(range(1, self.pred_len + 1))
        death_weights = [death_weights] * true_death.shape[0]
        death_weights = torch.tensor(death_weights).to(true_death)

        weighted_death = torch.mean(true_death[:,:self.train_len] * death_weights[:,:self.train_len],
                                    dim=1)

        weighted_death = torch.maximum(weighted_death, torch.tensor([10]*len(weighted_death)).to(weighted_death))

        # Balance Along Time Stamps
        death_loss = self.loss_fn(predicted_death, true_death) # [10,71]
        detailed_death_loss = death_loss.tolist()
        death_loss = death_loss * death_weights # [10,71]
        death_loss = torch.mean(death_loss,
                                dim = 1) # [10]

        # Balance between case and death
        balance = weighted_case / weighted_death

        if self.loss_name == 'MAE':
            loss = case_loss + balance * death_loss
        elif self.loss_name == 'MSE':
            loss = case_loss + balance * balance * death_loss

        # Balance for population
        loss = torch.div(loss, population) # [10]
        loss = torch.mean(loss)

        # Detailed Loss
        if return_detailed:

            return_columns = ['country_name','domain_name',f'Overall_{self.loss_name}']

            detailed_case_loss_df = pd.DataFrame(detailed_case_loss)
            detailed_death_loss_df = pd.DataFrame(detailed_death_loss)
            detailed_case_loss_df.columns = [f"day_{i}" for i in range(self.pred_len)]
            detailed_death_loss_df.columns = [f"day_{i}" for i in range(self.pred_len)]
            
            detailed_case_loss_df[f'Overall_{self.loss_name}'] = detailed_case_loss_df.mean(axis=1)
            detailed_death_loss_df[f'Overall_{self.loss_name}'] = detailed_case_loss_df.mean(axis=1)

            detailed_case_loss_df.insert(0,'country_name',batch['country_name'])
            detailed_case_loss_df.insert(1,'domain_name', batch['domain_name'])

            detailed_death_loss_df.insert(0,'country_name', batch['country_name'])
            detailed_death_loss_df.insert(1,'domain_name', batch['domain_name'])
        
            return loss, detailed_case_loss_df[return_columns], detailed_death_loss_df[return_columns]
        else:
            return loss
        
    
    def training_step(self, batch, batch_idx):
        
        # print(batch['domain_name'])

        preds = self.forward(batch)

        self.train_preds = preds

        loss = self.loss(preds, batch)

        self.log('train_loss', loss, on_epoch=True)

        print(loss)

        return loss
    
    def validation_step(self, batch):
        
        preds = self.forward(batch)
        
        self.validation_preds.append(preds)

        if self.validation_batch == []:
            self.validation_batch = batch
        else:
            for key in batch:
                if torch.is_tensor(batch[key]):
                    self.validation_batch[key] = torch.cat((self.validation_batch[key], batch[key]))
                else:
                    self.validation_batch[key] = self.validation_batch[key] + batch[key]


    def on_validation_epoch_end(self):
        
        self.validation_preds = torch.cat(self.validation_preds, dim=0) # [samples,pred_len,compartments]
        
        ## Check shape
        # for key in self.validation_batch:
        #     print(key, type(self.validation_batch[key]))
        #     if torch.is_tensor(self.validation_batch[key]):
        #         print(self.validation_batch[key].shape)
        #     else:
        #         print(len(self.validation_batch[key]))

        loss, case_loss_df, death_loss_df = self.loss(self.validation_preds,
                                                      self.validation_batch,
                                                      return_detailed=True)
        
        case_loss_df.to_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/cumulative_case_model_output/predicted_figures/case/day_case_difference.csv',
                            index = False)
        death_loss_df.to_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/cumulative_case_model_output/predicted_figures/death/day_death_difference.csv',
                            index = False)

        self.log('validation_loss', loss, on_epoch=True) 
        print(f"Validation Loss:{loss}")

        self.validation_preds = []
        self.validation_batch = []

    
    def test_step(self, batch, batch_idx):

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_runing_stats=False

        print(batch['domain_name'])

        preds = self.forward(batch)

        loss, case_loss_df, death_loss_df = self.loss(preds,batch,return_detailed=True)

        ## Afterwards Train Loss
        afterward_train_loss = self.loss(self.train_preds, batch)
        print(f"Afterwards Train Loss: {afterward_train_loss}")

        print(f"Test Loss: {loss}")
        self.log('test_loss', loss)

        test_case_prediction = preds[:,:,15]
        test_death_prediction = preds[:,:,14]

        self.train_preds_case = self.train_preds[:,:,15].tolist()
        self.train_preds_death = self.train_preds[:,:,14].tolist()

        self.test_country.append(batch['country_name'])
        self.test_domain.append(batch['domain_name'])
        self.test_case_prediction_list = self.test_case_prediction_list + test_case_prediction.tolist()
        self.test_death_prediction_list = self.test_death_prediction_list + test_death_prediction.tolist()

        self.test_case_true_list = self.test_case_true_list + [item[:self.pred_len] for item in batch['cumulative_case_number']]
        self.test_death_true_list = self.test_death_true_list + [item[:self.pred_len] for item in batch['cumulative_death_number']]
        

    def on_test_epoch_end(self):

        tspan = np.arange(0,len(self.test_case_true_list[0]),1)

        for i in range(len(self.test_country[0])):

            plt.figure()

            plt.plot(tspan, 
                     self.test_case_prediction_list[i],
                     # self.train_preds_case[i]
                     )
            plt.plot(tspan,
                     self.test_case_true_list[i],
                     )
            
            plt.legend(['Predicted Case Values', 'True Case Values'])
            plt.xlabel("days")
            plt.ylabel("Cumulative Cases")

            plt.savefig('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/cumulative_case_model_output/predicted_figures/case/' + self.test_country[0][i] + '_' + self.test_domain[0][i])

            plt.figure()

            plt.plot(tspan, 
                     self.test_death_prediction_list[i],
                     # self.train_preds_death[i]
                     )
            plt.plot(tspan,
                     self.test_death_true_list[i])
            
            plt.legend(['Predicted Death Values', 'True Death Values'])
            plt.xlabel("days")
            plt.ylabel("Cumulative Deaths")

            plt.savefig('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/cumulative_case_model_output/predicted_figures/death/' + self.test_country[0][i] + '_' + self.test_domain[0][i])

            
            
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)

        return optimizer