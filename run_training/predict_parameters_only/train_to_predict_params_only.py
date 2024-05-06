from run_training.predict_parameters_only.Training_Module import TrainingModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
import pandas as pd
import numpy as np
from utils.data_augmentation import data_augmentation
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import log10
from data.data_delphi_sampling import sampling_from_distribution

def run_training(lr: float = 1e-5,
                 past_pandemics: list = [],
                 target_pandemic: str = 'covid',
                 batch_size: int = 10,
                 target_training_len: int = 30,
                 pred_len: int = 60,
                 record_run: bool = False,
                 max_epochs: int = 50,
                 log_dir: str = '/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/logs/',
                 loss: str = 'MAE',
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
                 mape_threshold: float = 20,
                 log_parameter_loss: bool = False,
                 dropout: float = 0.0,
                 augmentation: bool = False,
                 validation_augmentation: bool = False,
                 log_name: str = None,
                 sample_factor: int = 5,
                 sampling_sigma: float = 0.01,
                 weight_decay: float = 0.0,
                 normalize_label: bool = False,
                 ):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        elif pandemic == '2010-2016_influenza':
            for year in [2010,2011,2012,2013,2014,2015,2016]:
                past_pandemic_data.extend(process_data(processed_data_path=f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_{year}_influenza_data_objects.pickle',
                                          raw_data=False))
        else:
            print(f"{pandemic} not in the processed data list, please process the data prefore running the model, skipping {[pandemic]}")

    for item in past_pandemic_data:
        if pd.isnull(item.domain_name):
            item.domain_name = 'empty'

    print(">>>> Training Data Processing >>>>>")

    past_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=past_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=True,
                                              augmentation=augmentation,
                                              augmentation_method='shifting')

    ## Remove Samples with no change in case num in first 30 days
    past_pandemic_dataset.pandemic_data = [item for item in past_pandemic_dataset if item.ts_input[0] != item.ts_input[-1]]
    
    ## Remove Samples with Failed Delphi Convergence
    past_pandemic_dataset.pandemic_data = [item for item in past_pandemic_dataset if item.true_delphi_params[0] != -999]

    ## Data Sampling Augmentation
    synthetic_data = sampling_from_distribution(past_pandemic_dataset.pandemic_data,
                                                sample_size=sample_factor,
                                                sigma=sampling_sigma,)
    
    past_pandemic_dataset.pandemic_data = past_pandemic_dataset.pandemic_data + synthetic_data

    ## Remove Samples with no change in case num after augmentation
    past_pandemic_dataset.pandemic_data = [item for item in past_pandemic_dataset if item.ts_input[0] != item.ts_input[-1]]

    ## Past Data Normalization
    for item in past_pandemic_dataset:
        ## Log Transformation for Scaling Problem
        log_input = np.log10(item.ts_input)
        ## Min-Max Normalization
        item.ts_input = (log_input - min(log_input)) / (max(log_input) - min(log_input))

    print("Total Train Length:", len(past_pandemic_dataset))

    train_data_loader = DataLoader(past_pandemic_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=past_pandemic_dataset.collate_fn,
                                   drop_last=False,
                                   )

    ## Load Target Pandemic Data

    print(">>>>> Testing Data Processing >>>>>")

    if target_pandemic == 'covid':
        target_pandemic_data = process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_covid_data_objects.pickle',
                                            raw_data=False)
    elif target_pandemic == '2017_influenza':
        target_pandemic_data = process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_2017_influenza_data_objects.pickle',
                                            raw_data=False)
    
    target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                              target_training_len=target_training_len,
                                              pred_len = pred_len,
                                              batch_size=batch_size,
                                              meta_data_impute_value=0,
                                              normalize_by_population=True,
                                              augmentation=validation_augmentation,
                                              augmentation_method='shifting')
    
    ## Remove Samples with no change in case num in first 30 days
    target_pandemic_dataset.pandemic_data = [item for item in target_pandemic_dataset if item.ts_input[0] != item.ts_input[-1]]
    
    ## Target Data Normalization
    for item in target_pandemic_dataset:
        ## Log Transformation for Scaling Problem
        log_input = np.log10(item.ts_input)
        ## Min-Max Normalization
        item.ts_input = (log_input - min(log_input)) / (max(log_input) - min(log_input))
        
    print("Total Testing Length:", len(target_pandemic_dataset))

    validation_data_loader = DataLoader(target_pandemic_dataset,
                                        batch_size = batch_size,
                                        shuffle=False,
                                        collate_fn=target_pandemic_dataset.collate_fn,
                                        drop_last=False)

    parameter_loss_weight = [1.0 / 1.0,
                             1.0 / 2.0,
                             1.0 / 1.0,
                             1.0 / 1.0,
                             1.0 / 0.32,
                             1.0 / 4.5,
                             1.0 / 0.2,
                             1.0 / 454.0,
                             1.0 / 8.22,
                             1.0 / 209.0,
                             1.0 / 2.0,
                             1.0 / 1.0,
                             ]

    model = TrainingModule(lr = lr,
                           loss_name = loss,
                           time_series_encoding=time_series_encoding,
                           ts_population_normalization=ts_population_normalization,
                           num_ts_encoding_layer=num_ts_encoding_layer,
                           ts_dim = ts_dim,
                           include_meta_data=include_meta_data,
                           meta_data_encoding = meta_data_encoding,
                           num_meta_encoding_layer=num_meta_encoding_layer,
                           meta_data_dim=meta_data_dim,
                           readout_type=readout_type,
                           num_readout_layer=num_readout_layer,
                           output_dim=output_dim,
                           hidden_dim=hidden_dim,
                           device = device,
                           parameter_loss_weight=parameter_loss_weight,
                           log_parameter_loss = log_parameter_loss,
                           dropout = dropout,
                           batch_size = batch_size,
                           past_pandemics = past_pandemics,
                           normalize_label=normalize_label,
                           )
    
    print(model)
    
    if log_name is None:
        log_name = f"{time_series_encoding}_Dropout={dropout}_Sampling={sample_factor}_WeightDecay={weight_decay}_Sigma={sampling_sigma}"

    if record_run:
        logger = WandbLogger(save_dir=log_dir,
                             project = 'Pandemic_Early_Warning',
                             name = log_name)
    else:
        logger = None
    
    trainer = Trainer(
        devices = 1,
        accelerator=device,
        max_epochs=max_epochs,
        logger=logger,
        num_sanity_val_steps = 0,
        default_root_dir= log_dir,
        log_every_n_steps=1,
    )

    trainer.fit(model,
                train_dataloaders=train_data_loader,
                val_dataloaders = validation_data_loader,
                )
    
run_training(### Training Args
             lr = 1e-3,
             batch_size = 64,
             target_training_len = 30,
             pred_len = 60,
             record_run = True,
             max_epochs = 1000,
             log_dir = '/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/logs/',
             ### Model Args
             loss = 'WMAE',
             time_series_encoding = 'ResNet18',
             ts_population_normalization = True,
             num_ts_encoding_layer = 5,
             ts_dim = 30,
             include_meta_data = True,
             meta_data_encoding = None,
             num_meta_encoding_layer = 5,
             meta_data_dim = 27,
             readout_type  = 'ResNet',
             num_readout_layer = 5,
             output_dim = 12,
             hidden_dim = 256,
             # past_pandemics=['dengue','ebola','sars','mpox','influenza'],
             past_pandemics=['2010-2016_influenza'],
             target_pandemic='2017_influenza',
             log_parameter_loss = False,
             augmentation=True,
             validation_augmentation=True,
             dropout=0.8,
             sample_factor = 20,
             sampling_sigma=0.1,
             weight_decay=0.0,
             normalize_label=True,)