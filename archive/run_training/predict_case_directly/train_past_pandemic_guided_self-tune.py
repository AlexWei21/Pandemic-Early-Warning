from run_training.predict_case_directly.Training_Module import TrainingModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
                 ):

    ## Load Past Pandemic Data
    past_pandemic_data = []

    for pandemic in past_pandemics:
        if pandemic == 'dengue':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_dengue_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'ebola':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_ebola_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'influenza':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_influenza_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'mpox':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_mpox_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'sars':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_sars_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == 'covid':
            past_pandemic_data.extend(process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_covid_data_objects.pickle',
                                                   raw_data=False))
        elif pandemic == '2010-2017_influenza':
            for year in [2010,2011,2012,2013,2014,2015,2016,2017]:
                past_pandemic_data.extend(process_data(processed_data_path=f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_{year}_influenza_data_objects.pickle',
                                          raw_data=False))
        else:
            print(f"{pandemic} not in the processed data list, please process the data prefore running the model, skipping {[pandemic]}")
    
    past_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=past_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=True,
                                              augmentation=False,
                                              max_shifting_len=10)

    past_pandemic_dataset.pandemic_data = [item for item in past_pandemic_dataset if sum(item.ts_case_input) != 0]

    print(f"Past Pandemic Training Size:{len(past_pandemic_dataset)}")
    
    ## Load Self-Tune Data
    self_tune_data = process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_covid_data_objects_no_smoothing.pickle',
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
        # item.time_dependent_weight = list(range(1, target_training_len + 1)) + [0]*(pred_len-target_training_len)
    
    self_tune_dataset.pandemic_data = [item for item in self_tune_dataset if sum(item.ts_case_input) != 0]

    ## Combine Past Pandemic and Self-Tuning Data
    past_pandemic_dataset.pandemic_data = past_pandemic_dataset.pandemic_data + self_tune_dataset.pandemic_data

    print(f"Past Pandemic + Self-Tune Training Size:{len(past_pandemic_dataset)}")

    ## Load Target Pandemic Data
    target_pandemic_data = process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_covid_data_objects_no_smoothing.pickle',
                                        raw_data=False)
    
    target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=False,
                                              input_log_transform=True,)

    ## Remove Samples with no change in case num in first 30 days
    target_pandemic_dataset.pandemic_data = [item for item in target_pandemic_dataset if sum(item.ts_case_input) != 0]
    print(f"Validation Length:{len(target_pandemic_dataset)}")

    for item in target_pandemic_dataset:
        item.time_dependent_weight = [1]*pred_len

    ## Dataloaders
    train_data_loader = DataLoader(past_pandemic_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=past_pandemic_dataset.collate_fn,
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
                           output_dir=output_dir)
    
    print(model)
    
    if record_run:
        
        logger = WandbLogger(save_dir=log_dir,
                             project = 'Pandemic_Early_Warning',
                             name = 'Past_Guided_Self-Tune')
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
             batch_size = 32,
             target_training_len = 46, # 46
             pred_len = 71, # 71
             record_run = True,
             max_epochs = 5000,
             log_dir = '/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/logs/',
             ### Model Args
             loss = 'MAE',
             dropout=0.0,
             past_pandemics=['dengue','ebola','sars','mpox','2010-2017_influenza'],
             target_self_tuning=True,
             include_death=False,
             selftune_weight=1,
             output_dir='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/DeepCompartmentModel/past_pandemic_guided/',)