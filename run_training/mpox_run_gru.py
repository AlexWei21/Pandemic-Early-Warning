from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from evaluation.method_comparison.GRU.gru import PF_GRU
from utils.loss_fn import MAPE, Combined_Loss
from utils.schedulers import Scheduler
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from datetime import datetime
from pathlib import Path
from utils.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils.sampler import Location_Fixed_Batch_Sampler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from evaluation.data_inspection.low_quality_data import covid_low_quality_data

'''
Pytorch Lightning Module for Training Gated Recurrent Unit (GRU) on Mpox
Parameters 
    output_dir: The output dir for training logs and outputs
    lr: Initial learning rate
    loss: Loss function name using for training [MAE, MAPE, Combined_Loss]
    train_len: Available time window for targeted pandemic training
    pred_len: The targeted prediction length for target pandemic
    include_death: Whether to include daily death number into training
    plot_validation: Whether to plot validation plots
    batch_size: Batch_size
    use_scheduler: Whether use learning rate scheduler or not
    loss_mape_weight: The weight for MAPE in the Combined Loss
    loss_mae_weight: The weight for MAE in the Combined Loss
'''
class TrainingModule(LightningModule):
    def __init__(self,
                 output_dir: str,
                 lr: float = 1e-3,
                 loss: str = 'MAPE',
                 train_len: int = 46,
                 pred_len: int = 71,
                 include_death: bool = True,
                 plot_validation: bool = False,
                 batch_size: int = 32,
                 use_scheduler:bool = False,
                 loss_mape_weight:float = 100,
                 loss_mae_weight:float = 0.5,
                 ):
        
        super().__init__()

        ## Setting parameters
        self.save_hyperparameters()
        self.lr = lr
        self.train_len = train_len
        self.pred_len = pred_len
        self.include_death = include_death
        self.plot_validation = plot_validation
        self.output_dir = output_dir

        self.batch_size = batch_size

        self.model = PF_GRU(input_size = 1,
                            hidden_size = 512,
                            num_layers = 5,
                            pred_length = pred_len - train_len,
                            sequence_length = train_len)
        
        self.use_scheduler = use_scheduler

        # Define Loss Fcuntion
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
        ts_case_input = ts_case_input.unsqueeze(2)

        predicted_casenum = self.model(ts_case_input)

        return predicted_casenum
    
    def loss(self, preds, batch):

        # Get Predicted Values and True Values
        predicted_case = preds

        if torch.is_tensor(batch['cumulative_case_number'][0]):
            true_case = [item[:self.pred_len].cpu() for item in batch['cumulative_case_number']]
        else:
            true_case = [item[:self.pred_len] for item in batch['cumulative_case_number']]
        true_case = torch.tensor(np.array(true_case)).to(predicted_case)

        true_case = true_case[:,self.train_len:]

        # Apply Time Dependent Weight
        case_loss = self.loss_fn(predicted_case, true_case) # shape: [batch_size,71]

        loss = torch.mean(case_loss)

        return loss
        
    
    def training_step(self, batch, batch_idx):

        preds = self.forward(batch)

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

        preds = self.forward(batch)
    
        self.validation_country = self.validation_country + batch['country_name']
        self.validation_domain = self.validation_domain + batch['domain_name']
        self.validation_preds.append(preds)

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

        ## Save Readble MAE MAPE Loss
        self.validation_batch['cumulative_case_number'] = torch.tensor(np.array([item[:self.pred_len] for item in self.validation_batch['cumulative_case_number']])).to(self.validation_preds)

        outsample_pred = self.validation_preds
        outsample_true = self.validation_batch['cumulative_case_number'][:,self.train_len:self.pred_len]
        
        validation_outsample_mae = self.calculate_mae(outsample_pred, outsample_true)
        validation_outsample_mape = self.calculate_mape(outsample_pred, outsample_true)

        # if torch.mean(validation_train_mape).item() < self.best_validation_insample_mape: 
        if self.epoch_id % 100 == 0:
            validation_loss_df = pd.DataFrame()
            validation_loss_df['Country'] = self.validation_country
            validation_loss_df['Domain'] = self.validation_domain
            validation_loss_df['OutSample_MAE'] = validation_outsample_mae.tolist()
            validation_loss_df['OutSample_MAPE'] = validation_outsample_mape.tolist()

            validation_loss_df.to_csv(self.output_dir + 'validation_location_loss.csv',
                                    index = False)
            ## Save Predicted Case
            predicted_case_df = pd.DataFrame(self.validation_preds.tolist())
            predicted_case_df.insert(0,'Country',self.validation_country)
            predicted_case_df.insert(1,'Domain',self.validation_domain)
            predicted_case_df.to_csv(self.output_dir + 'case_prediction.csv',
                                    index = False)

            ## Save Predicted Death
            predicted_death_df = pd.DataFrame(self.validation_preds.tolist())
            predicted_death_df.insert(0,'Country',self.validation_country)
            predicted_death_df.insert(1,'Domain',self.validation_domain)
            predicted_death_df.to_csv(self.output_dir + 'death_prediction.csv',
                                    index = False)
            
        ## Logging    
        loss = self.loss(self.validation_preds,
                         self.validation_batch)                
        self.log('validation_loss', 
                 loss, 
                 on_epoch=True, 
                 batch_size = self.batch_size) 
        
        print(f"Validation Loss:{loss}")
        
        self.log('validation_outsample_mae',
                 torch.mean(validation_outsample_mae),
                 on_epoch=True,
                 batch_size = self.batch_size)
        
        self.log('validation_outsample_mape',
                 torch.mean(validation_outsample_mape),
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

def run_training(lr: float = 1e-4,
                 batch_size: int = 10,
                 target_training_len: int = 47,
                 pred_len: int = 71,
                 record_run: bool = False,
                 max_epochs: int = 50,
                 log_dir: str = '/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/logs/',
                 loss: str = 'MAE',
                 past_pandemics: list = [],
                 include_death: bool = False,
                 output_dir:str = None,
                 use_lr_scheduler:bool=False,
                 loss_mae_weight: float=0.5,
                 loss_mape_weight: float=100,):
    
    Path(output_dir).mkdir(parents=False, exist_ok=True)

    ########## Load Past Pandemic Data ##########
    data_file_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/'
    past_pandemic_data = []

    for pandemic in past_pandemics:
        if pandemic == 'dengue':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_dengue_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'ebola':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_ebola_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'influenza':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_influenza_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'mpox':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_mpox_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'sars':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_sars_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'covid':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_covid_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == '2010-2017_influenza':
            for year in [2010,2011,2012,2013,2014,2015,2016,2017]:
                data = process_data(processed_data_path = data_file_dir+f'compartment_model_{year}_influenza_data_objects.pickle',
                                    raw_data=False)
                for item in data:
                    item.pandemic_name = item.pandemic_name + str(year)

                past_pandemic_data.extend(data)
        else:
            print(f"{pandemic} not in the processed data list, please process the data prefore running the model, skipping {[pandemic]}")

    past_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=past_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=True,
                                              augmentation=True,
                                              augmentation_method='shifting',
                                              max_shifting_len=10)

    # past_pandemic_dataset.pandemic_data = [item for item in past_pandemic_dataset if sum(item.ts_case_input) != 0]
    past_pandemic_dataset.pandemic_data = [item for item in past_pandemic_dataset if (item.country_name, item.domain_name) not in covid_low_quality_data]

    print(f"Past Pandemic Training Size:{len(past_pandemic_dataset)}")

    ########## Load Target Pandemic Data ##########
    target_pandemic_data = process_data(processed_data_path=data_file_dir+'compartment_model_mpox_data_objects.pickle',
                                        raw_data=False)
    
    target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=True,)

    ## Remove Samples with no change in case num in first 30 days
    # target_pandemic_dataset.pandemic_data = [item for item in target_pandemic_dataset if sum(item.ts_case_input) != 0]
    target_pandemic_dataset.pandemic_data = [item for item in target_pandemic_dataset if (item.country_name, item.domain_name) not in covid_low_quality_data]
    print(f"Validation Length:{len(target_pandemic_dataset)}")

    for item in target_pandemic_dataset:
        item.time_dependent_weight = [1]*pred_len
    
    ## Dataloaders
    for idx, item in enumerate(past_pandemic_dataset):
        item.idx = idx

    train_sampler = Location_Fixed_Batch_Sampler(dataset=past_pandemic_dataset,
                                                 batch_size=batch_size)

    train_data_loader = DataLoader(past_pandemic_dataset,
                                   batch_sampler=train_sampler,
                                   collate_fn=past_pandemic_dataset.collate_fn,
                                   drop_last=False,
                                   num_workers=1,
                                   )

    validation_data_loader = DataLoader(dataset=target_pandemic_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        collate_fn=target_pandemic_dataset.collate_fn,
                                        drop_last=False,
                                        num_workers=1,)
    
    model = TrainingModule(lr = lr,
                           loss = loss,
                           train_len=target_training_len,
                           pred_len = pred_len,
                           include_death = include_death,
                           batch_size = batch_size,
                           output_dir=output_dir,
                           use_scheduler=use_lr_scheduler,
                           loss_mae_weight=loss_mae_weight,
                           loss_mape_weight=loss_mape_weight)

    print(model)
    
    if record_run:
        
        logger = WandbLogger(save_dir=log_dir,
                             project = 'Pandemic_Early_Warning',
                             name = 'Mpox_GRU')
    else:
        logger = None
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=output_dir)

    trainer = Trainer(
        devices = 1,
        max_epochs=max_epochs,
        logger=logger,
        num_sanity_val_steps = 0,
        default_root_dir= log_dir,
        log_every_n_steps=1,
        callbacks=[lr_monitor, checkpoint_callback]
    )

    trainer.fit(model,
                train_data_loader,
                validation_data_loader)
    
if __name__ == '__main__':

    target_training_len = 14
    pred_len = 84

    run_training(### Training Args
                lr = 1e-3,
                batch_size = 1024,
                target_training_len = target_training_len, # 46
                pred_len = pred_len, # 71
                record_run = True,
                max_epochs = 500,
                log_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/logs/',
                ### Model Args
                loss = 'Combined_Loss',
                past_pandemics=['dengue','ebola','sars','2010-2017_influenza','covid'],
                use_lr_scheduler=False,
                loss_mae_weight = 0.5,
                loss_mape_weight = 100,
                output_dir=f"/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/gru/mpox_{datetime.today().strftime('%m-%d-%H00')}_{target_training_len}-{pred_len}/",)