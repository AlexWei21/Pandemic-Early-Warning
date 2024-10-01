from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from model.PandemicDeepCompartmentModel import pandemic_early_warning_model_with_DELPHI
from utils.loss_fn import MAPE, Combined_Loss
from utils.schedulers import Scheduler
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from utils.delphi_default_parameters import (
    p_v,
    p_d,
    p_h,)

class TrainingModule(LightningModule):
    def __init__(self,
                 output_dir: str,
                 lr: float = 1e-3,
                 loss: str = 'MAPE',
                 train_len: int = 46,
                 pred_len: int = 71,
                 dropout: float = 0.5,
                 include_death: bool = True,
                 plot_validation: bool = False,
                 batch_size: int = 32,
                 population_weighting: bool = False,
                 use_scheduler:bool = False,
                 loss_mape_weight:float = 100,
                 loss_mae_weight:float = 0.5,
                 ):
        
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        self.train_len = train_len
        self.pred_len = pred_len
        self.include_death = include_death
        self.plot_validation = plot_validation
        self.output_dir = output_dir

        self.p_d = p_d
        self.p_h = p_h
        self.p_v = p_v

        self.batch_size = batch_size

        self.model = pandemic_early_warning_model_with_DELPHI(pred_len=pred_len,
                                                              dropout = dropout,
                                                              include_death=include_death)
        
        self.population_weighting = population_weighting
        self.use_scheduler = use_scheduler

        if loss == 'MAPE':
            self.loss_fn = MAPE(reduction='none')
        elif loss == 'MAE':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss == 'MSE':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss == 'Combined_Loss':
            self.loss_fn = Combined_Loss(reduction='none',
                                         mae_weight=loss_mae_weight,
                                         mape_weight=loss_mape_weight)

        self.loss_name = loss
        
        self.epoch_id = 0

        self.test_country = []
        self.test_domain = []
        self.test_case_prediction_list = []
        self.test_death_prediction_list = []
        self.test_case_true_list = []
        self.test_death_true_list = []

        self.validation_country = []
        self.validation_domain = []
        self.validation_preds = []
        self.validation_batch = []
        self.validation_predicted_params = []

        self.train_case_pred = []
        self.train_death_pred = []

        self.best_validation_insample_mape = 9999

    def forward(self, batch):

        ts_case_input = batch['ts_case_input'].to(self.device)
        ts_case_input = ts_case_input.unsqueeze(1)

        if self.include_death:
            ts_death_input = batch['ts_death_input'].to(self.device)
            ts_death_input = ts_death_input.unsqueeze(1)

            ts_input = torch.cat((ts_case_input,ts_death_input), dim = 1)
        else:
            ts_input = ts_case_input

        meta_input = batch['meta_input'].to(self.device)

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

        predicted_casenum, predicted_params = self.model(ts_input,
                                                         global_params_fixed,
                                                         meta_input)

        return predicted_casenum, predicted_params
    
    def loss(self, preds, batch):

        # Get Predicted Values and True Values
        predicted_death = preds[:,:,14]
        predicted_case = preds[:,:,15]

        if torch.is_tensor(batch['cumulative_case_number'][0]):
            true_case = [item[:self.pred_len].cpu() for item in batch['cumulative_case_number']]
        else:
            true_case = [item[:self.pred_len] for item in batch['cumulative_case_number']]
        true_case = torch.tensor(np.array(true_case)).to(predicted_case)

        # Calculate constant for balance death/case
        weighted_case = torch.mean(true_case[:,:self.train_len] * batch['time_dependent_weight'][:,:self.train_len],
                                   dim=1)
        
        # Apply Time Dependent Weight
        case_loss = self.loss_fn(predicted_case, true_case) # shape: [batch_size,71]
        case_loss = case_loss * batch['time_dependent_weight'] # shape: [batch_size,71]
        case_loss = torch.sum(case_loss, dim = 1) / torch.sum(batch['time_dependent_weight'],dim=1)

        # Apply Sample Wise Weight
        case_loss = case_loss * batch['sample_weight']

        # Log Normalization
        # case_loss = torch.log(case_loss)

        ## Death Loss
        if self.include_death:

            true_death = [item[:self.pred_len] for item in batch['cumulative_death_number']]
            true_death = torch.tensor(true_death).to(predicted_death)

            ## Calculate Weight for Death
            death_weights = batch['time_dependent_weight']

            weighted_death = torch.mean(true_death[:,:self.train_len] * death_weights[:,:self.train_len],
                                        dim=1)

            weighted_death = torch.maximum(weighted_death, torch.tensor([10]*len(weighted_death)).to(weighted_death))

            # Balance Along Time Stamps
            death_loss = self.loss_fn(predicted_death, true_death) # [10,71]
            detailed_death_loss = death_loss.tolist()
            death_loss = death_loss * death_weights # [10,71]
            death_loss = torch.mean(death_loss,
                                    dim = 1) # [10]
            
            ## Apply Sample Wise Weight
            death_loss = death_loss * batch['sample_weight']

            # Balance between case and death
            balance = weighted_case / weighted_death
        else:
            balance = 0
            death_loss = 0

        loss = case_loss + balance * death_loss

        # Balance for population
        if self.population_weighting:
            population = batch['population']
            loss = torch.div(loss, population) * 10000 # [10]

        loss = torch.mean(loss)

        return loss
        
    
    def training_step(self, batch, batch_idx):

        preds, predicted_params = self.forward(batch)

        loss = self.loss(preds, batch)   

        self.log('train_loss', 
                 loss, 
                 on_epoch=True,
                 batch_size = self.batch_size)
        
        print(f"Train Loss: {loss}")

        if self.use_scheduler:
            sch = self.lr_schedulers()
            sch.step()

        return loss
    
    def validation_step(self, batch):

        preds, predicted_params = self.forward(batch)
    
        self.validation_country = self.validation_country + batch['country_name']
        self.validation_domain = self.validation_domain + batch['domain_name']
        self.validation_preds.append(preds)
        self.validation_predicted_params.append(predicted_params)

        if self.validation_batch == []:
            self.validation_batch = batch
        else:
            for key in batch:
                if torch.is_tensor(batch[key]):
                    self.validation_batch[key] = torch.cat((self.validation_batch[key], batch[key]))
                else:
                    if self.validation_batch[key] is not None:
                        self.validation_batch[key] = self.validation_batch[key] + batch[key]


    def on_validation_epoch_end(self):
        
        self.validation_preds = torch.cat(self.validation_preds, dim=0) # [samples,pred_len,compartments]

        self.validation_predicted_params = torch.cat(self.validation_predicted_params, dim=0)

        ## Save Readble MAE MAPE Loss
        self.validation_batch['cumulative_case_number'] = torch.tensor(np.array([item[:self.pred_len] for item in self.validation_batch['cumulative_case_number']])).to(self.validation_preds)

        train_time_pred = self.validation_preds[:,:self.train_len,15]
        train_time_true = self.validation_batch['cumulative_case_number'][:,:self.train_len]
        validation_train_mae = self.calculate_mae(train_time_pred, train_time_true)
        validation_train_mape = self.calculate_mape(train_time_pred, train_time_true)

        outsample_pred = self.validation_preds[:,self.train_len:self.pred_len,15]
        outsample_true = self.validation_batch['cumulative_case_number'][:,self.train_len:self.pred_len]
        validation_outsample_mae = self.calculate_mae(outsample_pred, outsample_true)
        validation_outsample_mape = self.calculate_mape(outsample_pred, outsample_true)
        
        # if torch.mean(validation_train_mape).item() < self.best_validation_insample_mape: 
        if self.epoch_id % 100 == 0:
            validation_loss_df = pd.DataFrame()
            validation_loss_df['Country'] = self.validation_country
            validation_loss_df['Domain'] = self.validation_domain
            validation_loss_df['InSample_MAE'] = validation_train_mae.tolist()
            validation_loss_df['OutSample_MAE'] = validation_outsample_mae.tolist()
            validation_loss_df['InSample_MAPE'] = validation_train_mape.tolist()
            validation_loss_df['OutSample_MAPE'] = validation_outsample_mape.tolist()

            validation_loss_df.to_csv(self.output_dir + 'validation_location_loss.csv',
                                    index = False)
            ## Save Predicted Case
            predicted_case_df = pd.DataFrame(self.validation_preds[:,:,15].tolist())
            predicted_case_df.insert(0,'Country',self.validation_country)
            predicted_case_df.insert(1,'Domain',self.validation_domain)
            predicted_case_df.to_csv(self.output_dir + 'case_prediction.csv',
                                    index = False)

            ## Save Predicted Death
            predicted_death_df = pd.DataFrame(self.validation_preds[:,:,14].tolist())
            predicted_death_df.insert(0,'Country',self.validation_country)
            predicted_death_df.insert(1,'Domain',self.validation_domain)
            predicted_death_df.to_csv(self.output_dir + 'death_prediction.csv',
                                    index = False)

            ## Save Parameters
            param_df = pd.DataFrame(self.validation_predicted_params.tolist())
            param_df.insert(0,'Country',self.validation_country)
            param_df.insert(1,'Domain',self.validation_domain)
            param_df.to_csv(self.output_dir + 'predicted_params.csv',
                            index = False)
            
            self.best_validation_insample_mape = torch.mean(validation_train_mape).item()

        
        ## Logging    
        loss = self.loss(self.validation_preds,
                         self.validation_batch)                
        self.log('validation_loss', 
                 loss, 
                 on_epoch=True, 
                 batch_size = self.batch_size) 
        
        print(f"Validation Loss:{loss}")

        self.log('validation_insample_mae',
                 torch.mean(validation_train_mae),
                 on_epoch=True,
                 batch_size = self.batch_size)
        
        self.log('validation_insample_mape',
                 torch.mean(validation_train_mape),
                 on_epoch=True,
                 batch_size = self.batch_size)
        
        self.log('validation_outsample_mae',
                 torch.mean(validation_outsample_mae),
                 on_epoch=True,
                 batch_size = self.batch_size)
        
        self.log('validation_outsample_mape',
                 torch.mean(validation_outsample_mape),
                 on_epoch=True,
                 batch_size = self.batch_size)
        
        self.log('best_validation_insample_mape',
                 self.best_validation_insample_mape,
                 on_epoch=True,
                 batch_size = self.batch_size)

        ## Reset List
        self.validation_country = []
        self.validation_domain = []
        self.validation_preds = []
        self.validation_batch = []
        self.validation_predicted_params = []
        self.epoch_id += 1

    
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

            plt.savefig(self.output_dir+'predicted_figures/case/' + self.test_country[0][i] + '_' + self.test_domain[0][i])

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

            plt.savefig(self.output_dir+'predicted_figures/death/' + self.test_country[0][i] + '_' + self.test_domain[0][i])

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)

        if self.use_scheduler:
            # scheduler = Scheduler(optimizer=optimizer,
            #                     dim_embed=2048 * 10,
            #                     warmup_steps=2000,)

            scheduler = StepLR(optimizer=optimizer,
                               step_size=40000,
                               gamma=0.1)

            return [optimizer], [scheduler]
        else:
            return optimizer
    
    def calculate_mae(self, predicted_value, true_value):

        mae_fn = nn.L1Loss(reduction='none')
        mae = mae_fn(predicted_value, true_value)
        mae = torch.mean(mae, dim = 1)

        return mae
    
    def calculate_mape(self, predicted_value, true_value):

        mape = torch.abs((true_value - predicted_value) / true_value) * 100
        mape = torch.mean(mape, dim = 1)

        return mape