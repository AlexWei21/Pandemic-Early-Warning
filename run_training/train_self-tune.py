from run_training.Training_Module import TrainingModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path

from evaluation.data_inspection.low_quality_data import covid_low_quality_data

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
                 include_death: bool = False,
                 target_self_tuning: bool = True,
                 selftune_weight:float = 1.0,
                 output_dir:str = None,
                 population_weighting:bool = False,
                 ):
    
    data_file_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/'

    Path(output_dir).mkdir(parents=False, exist_ok=True)

    ## Load Self-Tune Data
    self_tune_data = process_data(processed_data_path = data_file_dir+'compartment_model_covid_data_objects_no_smoothing.pickle',
                                  raw_data=False)

    self_tune_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=self_tune_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              augmentation=False,
                                              normalize_by_population=False,
                                              input_log_transform=True,
                                              loss_weight=selftune_weight)
    
    # Prevent Leakage in Self Tune Dataset
    for item in self_tune_dataset:
        item.time_dependent_weight = [1]*target_training_len + [0]*(pred_len-target_training_len)
    
    # Remove Samples with too few change
    # self_tune_dataset.pandemic_data = [item for item in self_tune_dataset if sum(item.ts_case_input) != 0]
    # self_tune_dataset.pandemic_data = [item for item in self_tune_dataset if sum(item.ts_case_input) >= 100] # Put in Dataset Creation

    # Remove Sample with low quality
    self_tune_dataset.pandemic_data = [item for item in self_tune_dataset if (item.country_name, item.domain_name) not in covid_low_quality_data]

    print(f"Self-tune Dataset Length: {len(self_tune_dataset)}")

    ## Load Target Pandemic Data
    target_pandemic_data = process_data(processed_data_path = data_file_dir+'compartment_model_covid_data_objects_no_smoothing.pickle',
                                        raw_data=False)
    
    target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=True,)

    ## Remove Samples with too few change
    # target_pandemic_dataset.pandemic_data = [item for item in target_pandemic_dataset if sum(item.ts_case_input) >= 100] # Put in Dataset Creation

    # Remove Sample with low quality
    target_pandemic_dataset.pandemic_data = [item for item in target_pandemic_dataset if (item.country_name, item.domain_name) not in covid_low_quality_data]

    print(f"Target Dataset Length: {len(target_pandemic_dataset)}")

    for item in target_pandemic_dataset:
        item.time_dependent_weight = [1]*pred_len

    ## Dataloaders
    train_data_loader = DataLoader(self_tune_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=self_tune_dataset.collate_fn,
                                   drop_last=False,
                                   num_workers=1,
                                   )
    
    validation_data_loader = DataLoader(target_pandemic_dataset,
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
                           population_weighting=population_weighting)
    
    print(model)
    
    if record_run:
        
        logger = WandbLogger(save_dir=log_dir,
                             project = 'Pandemic_Early_Warning',
                             name = 'Self-tune')
    else:
        logger = None
    
    trainer = Trainer(
        devices = 1,
        max_epochs=max_epochs,
        logger=logger,
        num_sanity_val_steps = 0,
        default_root_dir= log_dir,
        log_every_n_steps=1,
    )

    trainer.fit(model,
                train_data_loader,
                validation_data_loader)
    
run_training(### Training Args
             lr = 1e-5,
             batch_size = 245,
             target_training_len = 46, # 46
             pred_len = 71, # 71
             record_run = True,
             max_epochs = 10000,
             log_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/logs/',
             ### Model Args
             loss = 'MAE',
             dropout=0.0,
             past_pandemics=[],
             target_self_tuning=True,
             include_death=False,
             population_weighting=True,
             selftune_weight=1,
             output_dir=f"/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/self_tune/{datetime.today().strftime('%m-%d-%H')}/",)