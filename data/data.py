import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader
from data.data_utils import pandemic_meta_data_imputation

class Pandemic_Data():
    def __init__(self, look_back_len, pred_len, meta_data_len):

        super.__init__

        self.pandemic_name = None
        self.country_name = None
        self.domain_name = None
        self.subdomain_name = None
        self.x = np.empty((0,look_back_len), float)
        self.y = np.empty((0,pred_len), float)
        self.time_stamp_x = np.empty((0,look_back_len), pd.Timestamp)
        self.time_stamp_y = np.empty((0,pred_len), pd.Timestamp)
        self.meta_data = np.empty((0,meta_data_len), float)
        self.decoder_input = np.empty((0,pred_len), float)

class Compartment_Model_Pandemic_Data():
    def __init__(self, pandemic_name:str = None, country_name:str = None, domain_name:str = None, subdomain_name:str = None, 
                 start_date:pd.Timestamp = None, end_date:pd.Timestamp = None, update_frequency:str = None, population:int = None, pandemic_meta_data:dict = None,
                 case_number:list = None, cumulative_case_number:list = None, death_number:list = None, cumulative_death_number:list = None, first_day_above_hundred:pd.Timestamp = None,
                 timestamps:list = None):

        ## Pandemic Information
        self.pandemic_name = pandemic_name

        ## Geological Information
        self.country_name = country_name
        self.domain_name = domain_name
        self.subdomain_name = subdomain_name

        ## Time Information
        self.start_date = start_date
        self.first_day_above_hundred = first_day_above_hundred
        self.end_date = end_date
        self.update_frequency = update_frequency

        ## Meta Data
        self.population = population
        self.pandemic_meta_data = pandemic_meta_data

        ## Time-Series Data
        self.case_number = case_number
        self.cumulative_case_number = cumulative_case_number
        self.death_number = death_number
        self.cumulative_death_number = cumulative_death_number
        self.timestamps = timestamps

    def __str__(self) -> str:
        Pandemic_information = ('pandemic_name: {0} \ncountry_name: {1} \ndomain_name: {2} \nsubdomain_name: {3}').format(self.pandemic_name, self.country_name, self.domain_name, self.subdomain_name)
        Time_information = ('\nstart_date: {0} \nfirst_day_above_hundred: {1} \nend_date: {2} \nupdate_frequency: {3}').format(self.start_date, self.first_day_above_hundred, self.end_date, self.update_frequency)
        Meta_Information = ('\nPopulation: {0} \nPandemic_Meta_data: {1}'.format(self.population, self.pandemic_meta_data))
        Time_Series_Data = ('\ncase_number: {0} \ncumulative_case_number: {1} \ndeath_number: {2} \ncumulative_death_number: {3} \ntime_stamps: {4}' ).format(self.case_number, self.cumulative_case_number, self.death_number, self.cumulative_death_number, self.timestamps)
        return '\n' + Pandemic_information + '\n' + Time_information + '\n' + Meta_Information + '\n' + Time_Series_Data + '\n'

class Compartment_Model_Pandemic_Dataset(LightningDataModule):

    def __init__(self, 
                 pandemic_data,
                 target_training_len = 30,
                 batch_size = 64,
                 ):
        
        self.pandemic_data = pandemic_data
        self.train_len = target_training_len
        self.batch_size = batch_size

        for item in pandemic_data:
            if item.pandemic_meta_data is None:
                pandemic_data.remove(item)
                continue
            else:
                item.model_input = list(item.cumulative_case_number[:target_training_len]) + list(item.pandemic_meta_data.values())
                item.model_input = [float(i) for i in item.model_input]
                item.model_input = pandemic_meta_data_imputation(item.model_input)

        for item in pandemic_data:         
            if len(item.model_input) != 57:
                pandemic_data.remove(item)


    def __len__(self):
        return len(self.pandemic_data)
    
    def __getitem__(self, index):
        return self.pandemic_data[index]
    
    def collate_fn(self, batch):

        pandemic_name = [item.pandemic_name for item in batch]
        population = [float(item.population.replace(',','')) for item in batch]
        cumulative_case_number = [item.cumulative_case_number for item in batch]
        cumulative_death_number = [item.cumulative_death_number for item in batch]
        model_input = [item.model_input for item in batch]

        country_name = [item.country_name for item in batch]
        domain_name = [item.domain_name for item in batch]
        subdomain_name = [item.subdomain_name for item in batch]

        start_date = [item.start_date for item in batch]
        first_day_above_hundred = [item.first_day_above_hundred for item in batch]
        end_date = [item.end_date for item in batch]
        update_frequency = [item.update_frequency for item in batch]

        pandemic_meta_data = [item.pandemic_meta_data for item in batch]

        case_number = [item.case_number for item in batch]
        cumulative_case_number = [item.cumulative_case_number for item in batch]
        death_number = [item.death_number for item in batch]
        cumulative_death_number = [item.cumulative_death_number for item in batch]
        timestamps = [item.timestamps for item in batch]

        return dict(model_input = torch.tensor(model_input),
                    pandemic_name = pandemic_name,
                    population = torch.tensor(population),
                    cumulative_case_number = cumulative_case_number,
                    cumulative_death_number = cumulative_death_number,
                    country_name = country_name,
                    domain_name = domain_name,
                    subdomain_name = subdomain_name,
                    start_date = start_date,
                    first_day_above_hundred = first_day_above_hundred,
                    end_date = end_date,
                    update_frequency = update_frequency,
                    pandemic_meta_data = pandemic_meta_data,
                    case_number = case_number,
                    death_number = death_number,
                    timestamps = timestamps
                    )
    
