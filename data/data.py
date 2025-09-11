import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader
from utils.data_utils import pandemic_meta_data_imputation
from utils.data_augmentation import data_augmentation, find_last_augmentation_date
from matplotlib import pyplot as plt
import pickle

# class Pandemic_Data():
#     def __init__(self, look_back_len, pred_len, meta_data_len):

#         super.__init__

#         self.pandemic_name = None
#         self.country_name = None
#         self.domain_name = None
#         self.subdomain_name = None
#         self.x = np.empty((0,look_back_len), float)
#         self.y = np.empty((0,pred_len), float)
#         self.time_stamp_x = np.empty((0,look_back_len), pd.Timestamp)
#         self.time_stamp_y = np.empty((0,pred_len), pd.Timestamp)
#         self.meta_data = np.empty((0,meta_data_len), float)
#         self.decoder_input = np.empty((0,pred_len), float)

def parse_population(pop):
    return float(str(pop).replace(',', ''))

class Compartment_Model_Pandemic_Data():
    def __init__(self, pandemic_name:str = None, country_name:str = None, domain_name:str = None, 
                 subdomain_name:str = None, start_date:pd.Timestamp = None, end_date:pd.Timestamp = None, 
                 update_frequency:str = None, population:int = None, pandemic_meta_data:dict = None,
                 case_number:list = None, cumulative_case_number:list = None, death_number:list = None, 
                 cumulative_death_number:list = None, first_day_above_hundred:pd.Timestamp = None,
                 timestamps:list = None, series_id:str=None):

        ## Pandemic Information
        self.pandemic_name = pandemic_name

        ## Geological Information
        self.country_name = country_name
        self.domain_name = domain_name
        self.subdomain_name = subdomain_name
        self.series_id = series_id

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
                 pred_len = 60,
                 batch_size = 64,
                 meta_data_impute_value = -999,
                 normalize_by_population = False,
                 input_log_transform = False,
                 augmentation = False,
                 augmentation_method = None,
                 max_shifting_len = 10,
                 loss_weight:int = 1,
                 ):
        
        self.pandemic_data = pandemic_data
        self.train_len = target_training_len
        self.batch_size = batch_size

        pandemic_data = [item for item in pandemic_data if item.pandemic_meta_data is not None]
        pandemic_data = [item for item in pandemic_data if len(item.cumulative_case_number) >= (target_training_len + pred_len)]
        # pandemic_data = [item for item in pandemic_data if (item.cumulative_case_number[target_training_len-1] - item.cumulative_case_number[0]) >= 100]

        for item in pandemic_data:
            if normalize_by_population:
                # item.model_input = list(item.cumulative_case_number[:target_training_len] / float(item.population.replace(',',''))) + list(item.pandemic_meta_data.values())
                item.ts_case_input_full = list(item.cumulative_case_number / parse_population(item.population))
                if item.cumulative_death_number is not None:
                    item.ts_death_input_full = list((item.cumulative_death_number / parse_population(item.population)) * 100)
                else:
                    item.ts_death_input_full = None
            else:
                # item.model_input = list(item.cumulative_case_number[:target_training_len]) + list(item.pandemic_meta_data.values())
                item.ts_case_input_full = list(item.cumulative_case_number)
                if item.cumulative_death_number is not None:
                    item.ts_death_input_full = list(item.cumulative_death_number)
                else:
                    item.ts_death_input_full = None
            
            # Compute Daily case use as input
            item.daily_case_list = [0]
            for n in range(1,len(item.ts_case_input_full)):
                item.daily_case_list.append(item.ts_case_input_full[n] - item.ts_case_input_full[n-1])

            # Transform Input Data
            if input_log_transform:
                item.daily_case_list = [x+1 for x in item.daily_case_list]
                if not all(x>0 for x in item.daily_case_list):
                    print(f"{item.country_name} {item.pandemic_name} data has negative daily cases, please check. Here negative daily cases are set to 0")
                    item.daily_case_list = [x if x >= 1 else 1 for x in item.daily_case_list]
                item.daily_case_list = np.log(item.daily_case_list)

            item.ts_case_input = item.daily_case_list[:target_training_len]

            if item.ts_death_input_full is not None:
                item.ts_death_input = item.ts_death_input_full[:target_training_len]
            else:
                item.ts_death_input = None

            item.meta_input = pandemic_meta_data_imputation(list(item.pandemic_meta_data.values()),
                                                            impute_value=meta_data_impute_value,)

            if augmentation_method == 'shifting':
                item.last_augmentation_idx = max(find_last_augmentation_date(item),pred_len)

            item.time_dependent_weight = list(range(1, pred_len + 1))

            item.loss_weight = loss_weight

        if augmentation:
            pandemic_data = data_augmentation(pandemic_data,
                                              method = augmentation_method,
                                              ts_len=target_training_len,
                                              pred_len=pred_len)            

        # Z-Score normalization of all ts_case_input
        # ts_case_input = [item.ts_case_input for item in pandemic_data]
        # ts_case_input = np.array(ts_case_input)
        # ts_case_input_mean = np.mean(ts_case_input, axis=0)
        # ts_case_input_std = np.std(ts_case_input, axis=0)
        # for item in pandemic_data:
        #     item.ts_case_input = (item.ts_case_input - ts_case_input_mean) / ts_case_input_std

        print(f"Raw Data Num: {len(self.pandemic_data)}")
        print(f"Data Num after {augmentation_method}: {len(pandemic_data)}")

        self.pandemic_data = pandemic_data

    def __len__(self):
        return len(self.pandemic_data)
    
    def __getitem__(self, index):
        return self.pandemic_data[index]
    
    def collate_fn(self, batch):

        pandemic_name = [item.pandemic_name for item in batch]
        population = [float(str(item.population).replace(',', '')) for item in batch]
        cumulative_case_number = [item.cumulative_case_number for item in batch]
        cumulative_death_number = [item.cumulative_death_number for item in batch]
        # model_input = [item.model_input for item in batch]

        country_name = [item.country_name for item in batch]
        domain_name = [item.domain_name for item in batch]
        subdomain_name = [item.subdomain_name for item in batch]
        series_id = [item.series_id for item in batch]

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


        ts_case_input = [item.ts_case_input for item in batch]
        ts_death_input = [item.ts_death_input for item in batch]
        if None in ts_death_input:
            ts_death_input = None
        else:
            torch.tensor(ts_death_input).float()

        meta_input = [item.meta_input for item in batch]
        # true_delphi_params = [item.true_delphi_params for item in batch]

        time_dependent_weight = [item.time_dependent_weight for item in batch]
        sample_weight = [item.loss_weight for item in batch]

        return dict(ts_case_input = torch.tensor(np.array(ts_case_input)).float(),
                    ts_death_input = ts_death_input,
                    meta_input = torch.tensor(meta_input).float(),
                    pandemic_name = pandemic_name,
                    population = torch.tensor(population),
                    cumulative_case_number = cumulative_case_number,
                    cumulative_death_number = cumulative_death_number,
                    country_name = country_name,
                    domain_name = domain_name,
                    subdomain_name = subdomain_name,
                    series_id = series_id,
                    start_date = start_date,
                    first_day_above_hundred = first_day_above_hundred,
                    end_date = end_date,
                    update_frequency = update_frequency,
                    pandemic_meta_data = pandemic_meta_data,
                    case_number = case_number,
                    death_number = death_number,
                    timestamps = timestamps,
                    time_dependent_weight = torch.tensor(time_dependent_weight),
                    sample_weight = torch.tensor(sample_weight),
                    # true_delphi_params = torch.tensor(true_delphi_params),
                    )
    
    
if __name__ == '__main__':

    # with open('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/data_files/processed_data/train/compartment_model_dengue_data_objects.pickle','rb') as file:
    #     data_list = pickle.load(file)
    with open('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/data_files/processed_data/train/toy.pickle','rb') as file:
        data_list = pickle.load(file)

    past_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=data_list,
                                                               target_training_len=56,
                                                               pred_len = 84,
                                                               batch_size=1024,
                                                               meta_data_impute_value=0,
                                                               normalize_by_population=False,
                                                               input_log_transform=True,
                                                               augmentation=True,
                                                               augmentation_method='shifting',
                                                               max_shifting_len=10)
    
    # for item in past_pandemic_dataset.pandemic_data:
    #     print(item.ts_case_input)