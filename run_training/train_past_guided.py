from run_training.Training_Module import TrainingModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler
from utils.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.sampler import Location_Fixed_Batch_Sampler
from evaluation.data_inspection.low_quality_data import covid_low_quality_data
from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import argparse
import inspect

'''
Training Function for HG-DCM for COVID-19
Parameters 
    lr: Initial learning rate
    batch_size: Batch_size
    target_training_len: Available time window for targeted pandemic training
    pred_len: The targeted prediction length for target pandemic
    record_run: Whether to record run or not on Wandb
    max_epochs: Number of maximum number of training epochs
    log_dir: The directory for saving logs
    loss: Loss function name using for training [MAE, MAPE, Combined_Loss]
    dropout: The dropout value for model training
    past_pandemics: The list of pandemics that are used as past pandemic in training HG-DCM
    include_death: Whether to include daily death number into training
    target_self_runing: Include current pandemic available data in training
    selftune_weight: The weight for selftune data in training
    output_dir: The output dir for training logs and outputs
    population_weighting: Whether to weight loss by population
    use_scheduler: Whether use learning rate scheduler or not
    loss_mape_weight: The weight for MAPE in the Combined Loss
    loss_mae_weight: The weight for MAE in the Combined Loss
'''
def run_training(lr: float = 1e-3,
                 batch_size: int = 10,
                 target_training_len: int = 47,
                 pred_len: int = 71,
                 record_run: bool = False,
                 max_epochs: int = 50,
                 log_dir: str = '/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/logs/',
                 loss: str = 'MAE',
                 dropout: float = 0.5,
                 past_pandemics: list = [],
                 target_pandemic: str = '',
                 include_death: bool = False,
                 target_self_tuning: bool = True,
                 selftune_weight:list = [0.5,0.5],
                 output_dir:str = None,
                 population_weighting:bool = False,
                 use_lr_scheduler:bool=False,
                 loss_mae_weight:float = 0.5,
                 loss_mape_weight:float = 100,
                 compartmental_model:str='delphi',
                 input_log_transform:bool=True,
                 nn_model:str='resnet50',
                 ):

    Path(output_dir).mkdir(parents=False, exist_ok=True)

    torch.manual_seed(15)

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    print("==== Hyperparameters ====")
    for arg in args:
        print(f"{arg}: {values[arg]}")
    print("========================")

    ########## Load Past Pandemic Data ##########
    data_file_dir = 'data_files/processed_data/'
    print(">>>>>>>>>> Processing Past Pandemic Data >>>>>>>>>>")
    past_pandemic_data = []

    for pandemic in past_pandemics:
        if pandemic == 'dengue':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'train/compartment_model_dengue_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'ebola':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'train/compartment_model_ebola_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'influenza':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'train/compartment_model_influenza_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'mpox':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'train/compartment_model_mpox_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'sars':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'train/compartment_model_sars_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'covid':
            past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'train/compartment_model_covid_data_objects.pickle',
                                                   raw_data=False))
        else:
            print(f"{pandemic} not in the processed data list, please process the data prefore running the model, skipping {[pandemic]}")

    past_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=past_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=input_log_transform,
                                              augmentation=True,
                                              augmentation_method='shifting',
                                              max_shifting_len=10,
                                              loss_weight=selftune_weight[0],)

    # past_pandemic_dataset.pandemic_data = [item for item in past_pandemic_dataset if sum(item.ts_case_input) != 0]
    # past_pandemic_dataset.pandemic_data = [item for item in past_pandemic_dataset if (item.country_name, item.domain_name) not in covid_low_quality_data]

    print(f"Past Pandemic Training Size:{len(past_pandemic_dataset)}")
    
    ########## Load Self-Tune Data ##########
    print(">>>>>>>>>> Processing Self-tune Pandemic Data >>>>>>>>>>")
    self_tune_data = process_data(processed_data_path=data_file_dir+f'validation/compartment_model_{target_pandemic}_data_objects.pickle',
                                        raw_data=False)

    self_tune_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=self_tune_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              augmentation=True,
                                              augmentation_method='masking',
                                              normalize_by_population=False,
                                              input_log_transform=input_log_transform,
                                              loss_weight=selftune_weight[1])
    
    # Prevent Leakage in Self Tune Dataset
    for item in self_tune_dataset:
        item.time_dependent_weight = [1]*target_training_len + [0]*(pred_len-target_training_len)
        # item.time_dependent_weight = list(range(1, target_training_len + 1)) + [0]*(pred_len-target_training_len)
    
    # self_tune_dataset.pandemic_data = [item for item in self_tune_dataset if sum(item.ts_case_input) != 0]
    # self_tune_dataset.pandemic_data = [item for item in self_tune_dataset if (item.country_name, item.domain_name) not in covid_low_quality_data]
    print(f"Self-Tune Training Size:{len(self_tune_dataset)}")


    ## Combine Past Pandemic and Self-Tuning Data
    past_pandemic_dataset.pandemic_data = past_pandemic_dataset.pandemic_data + self_tune_dataset.pandemic_data

    print(f"Past Pandemic + Self-Tune Training Size:{len(past_pandemic_dataset)}")

    ########## Load Target Pandemic Data ##########
    print(">>>>>>>>>> Processing Target Pandemic Data >>>>>>>>>>")
    target_pandemic_data = process_data(processed_data_path=data_file_dir+f'validation/compartment_model_{target_pandemic}_data_objects.pickle',
                                        raw_data=False)
    
    target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=input_log_transform,)

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
                           dropout=dropout,
                           include_death = include_death,
                           batch_size = batch_size,
                           output_dir=output_dir,
                           population_weighting=population_weighting,
                           use_scheduler=use_lr_scheduler,
                           loss_mae_weight=loss_mae_weight,
                           loss_mape_weight=loss_mape_weight,
                           compartmental_model=compartmental_model,
                           nn_model=nn_model,
                           )
    
    print(model)
    
    if record_run:
        
        logger = WandbLogger(save_dir=log_dir,
                             project = 'Pandemic_Early_Warning',
                             name = 'Past_Guided_Self-Tune')
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
        callbacks=[lr_monitor, checkpoint_callback],
        gradient_clip_val=1.0,
    )

    trainer.fit(model,
                train_data_loader,
                validation_data_loader)

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','1'): return True
    if v.lower() in ('no','false','f','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    # target_training_len = 56
    # pred_len = 84

    # run_training(### Training Args
    #             lr = 1e-5,
    #             batch_size = 1247,
    #             target_training_len = target_training_len, 
    #             pred_len = pred_len, 
    #             record_run = False,
    #             max_epochs = 10000,
    #             log_dir = 'logs/',
    #             ### Model Args
    #             loss = 'Combined_Loss',
    #             dropout=0.0,
    #             past_pandemics=['dengue','ebola','sars','influenza'],
    #             target_pandemic = 'covid',
    #             target_self_tuning=True,
    #             include_death=False,
    #             population_weighting= False,
    #             selftune_weight=1,
    #             use_lr_scheduler=True,
    #             loss_mae_weight = 0.5,
    #             loss_mape_weight = 100,
    #             output_dir=f"output/past_guided/{datetime.today().strftime('%m-%d-%H00')}_{target_training_len}-{pred_len}/",
    #             compartmental_model='delphi')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1247)
    parser.add_argument('--target_training_len', type=int, default=56)
    parser.add_argument('--pred_len', type=int, default=84)
    parser.add_argument('--record_run', type=str2bool, default=False)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--loss', type=str, default='Combined_Loss')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--past_pandemics', nargs='+', default=['dengue','ebola','sars','influenza'])
    parser.add_argument('--target_pandemic', type=str, default='covid')
    parser.add_argument('--target_self_tuning', type=str2bool, default=True)
    parser.add_argument('--include_death', type=str2bool, default=False)
    parser.add_argument('--population_weighting', type=str2bool, default=False)
    parser.add_argument('--selftune_weight', nargs='+', type=float, default=[0.5,0.5])
    parser.add_argument('--use_lr_scheduler', type=str2bool, default=True)
    parser.add_argument('--loss_mae_weight', type=float, default=0.5)
    parser.add_argument('--loss_mape_weight', type=float, default=100)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--compartmental_model', type=str, default='delphi')
    parser.add_argument('--nn_model', type=str, default='resnet50')

    args = parser.parse_args()

    # If not specified, use the auto-generated output_dir as before
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"output/past_guided/{datetime.today().strftime('%m-%d-%H00')}_{args.target_training_len}-{args.pred_len}/"

    run_training(
        lr = args.lr,
        batch_size = args.batch_size,
        target_training_len = args.target_training_len,
        pred_len = args.pred_len,
        record_run = args.record_run,
        max_epochs = args.max_epochs,
        log_dir = args.log_dir,
        loss = args.loss,
        dropout = args.dropout,
        past_pandemics = args.past_pandemics,
        target_pandemic = args.target_pandemic,
        target_self_tuning = args.target_self_tuning,
        include_death = args.include_death,
        population_weighting = args.population_weighting,
        selftune_weight = args.selftune_weight,
        use_lr_scheduler = args.use_lr_scheduler,
        loss_mae_weight = args.loss_mae_weight,
        loss_mape_weight = args.loss_mape_weight,
        output_dir = output_dir,
        compartmental_model = args.compartmental_model,
        nn_model = args.nn_model,
    )