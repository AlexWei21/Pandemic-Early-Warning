import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from data.processing import read_pandemic_ts

class Pandemic_Dataset(LightningDataModule):
    def __init__(self, 
                 root_dir, 
                 past_file_path: list, 
                 target_file_path: str, 
                 past_pandemic_name: list,  
                 target_pandemic_name: str, 
                 look_back_len = 100, 
                 pred_len = 200, 
                 with_meta_data = False, 
                 moving_average = 1, 
                 data_smoothing = 'False', 
                 past_data_frequency = [],
                 target_data_frequency = 'Daily', 
                 meta_data_len = 14,  
                 raw_data = False,
                 flag = "train"):
        
        super().__init__

        assert flag in ['train','vali','test']

        self.past_data_dir = [root_dir + x for x in past_file_path]
        print(self.past_data_dir)
        self.target_data_dir = root_dir + target_file_path

        self.look_back_len = look_back_len
        self.pred_len = pred_len
        self.meta_data_len = meta_data_len

        if flag == 'train':
            if raw_data == True:           
                self.data_list = read_pandemic_ts(data_dir=self.past_data_dir, pandemic_name= past_pandemic_name, multiple_files = True,
                                              look_back_len=look_back_len, pred_len=pred_len, with_meta_data=with_meta_data, moving_average=moving_average,
                                              data_smoothing=data_smoothing, target_data_frequency=target_data_frequency, past_data_frequency= past_data_frequency,
                                              meta_data_len=meta_data_len)
            else:
                self.data_list = torch.load(self.past_data_dir)
        
        elif flag == 'test':
            if raw_data == True:
                self.data_list = read_pandemic_ts(data_dir=self.target_data_dir, pandemic_name= target_pandemic_name, multiple_files = False,
                                              look_back_len=look_back_len, pred_len=pred_len, with_meta_data=with_meta_data, moving_average=moving_average,
                                              data_smoothing=data_smoothing, target_data_frequency=target_data_frequency, past_data_frequency=past_data_frequency,
                                              meta_data_len=meta_data_len)
            else:
                self.data_list = torch.load(self.target_data_dir)
        else:
            print("Currently Not Supported")
            exit()
        
    def __getitem__(self,idx):
        return self.data_list[idx]
    
    def __len__(self):
        return len(self.data_list)
    
    def print_data_list(self):
        print(self.data_list)

    def collate_fn(self, batch):       
        
        final_batch = {"pandemic_name":[],
                       "country_name":[],
                       "domain_name":[],
                       "subdomain_name":[],
                       "x":[],
                       "y":[],
                       "decoder_input":[],
                       "time_stamp_x":[],
                       "time_stamp_y":[],
                       "meta_data":[]}
        
        for i in batch:
            final_batch["pandemic_name"].append(i.pandemic_name)
            final_batch["country_name"].append(i.country_name)
            final_batch["domain_name"].append(i.domain_name)
            final_batch["subdomain_name"].append(i.subdomain_name)
            final_batch["x"].append(i.x)
            final_batch["y"].append(i.y)
            final_batch["decoder_input"].append(i.decoder_input)
            final_batch["time_stamp_x"].append(i.time_stamp_x)
            final_batch["time_stamp_y"].append(i.time_stamp_y)
            final_batch["meta_data"].append(i.meta_data)

        return self.to_tensor(final_batch)
    
    def to_tensor(self, target):

        out = {}

        for k,v in target.items():
            if k in ("x","y","meta_data","decoder_input"):
                out[k] = torch.stack(v)
            else:
                out[k] = v

        return out


if __name__ == '__main__':
    # read_pandemic_ts(data_dir='F:/Pandemic-Database/Processed_Data/Covid_19/Covid_World_Domain_Daily_CumCases.csv',
    #                  pandemic_name='Covid',
    #                  data_frequency= 'Daily',
    #                  moving_average=7)
    
    # read_pandemic_ts(root_dir='F:/Pandemic-Database/Processed_Data/Dengue_Fever/',
    #                  file_name='Dengue_AMRO_Country_Weekly_Cases.csv',
    #                  data_frequency= 'Weekly',
    #                  data_smoothing=True)

    a = Pandemic_Dataset(root_dir = 'F:/Pandemic-Database/Processed_Data/', 
                         target_file_path = "Covid_19/Covid_World_Domain_Daily_CumCases.csv", 
                         past_file_path = ["Dengue_Fever/Dengue_AMRO_Country_Weekly_CumCases.csv",
                                           "Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv",
                                           "Monkeypox/Mpox_World_Country_Daily_CumCases.csv",
                                           "SARS/SARS_World_Country_Daily_CumCases.csv",
                                           "Influenza/Influenza_World_Domain_Weekly_CumCases.csv"],
                         target_pandemic_name = 'Covid',
                         past_pandemic_name = ['Dengue',
                                               'Ebola',
                                               'MPox',
                                               'SARS',
                                               'Influenza'],
                         raw_data=True,
                         flag = 'train',
                         look_back_len=30,
                         pred_len=60,
                         data_smoothing=True,
                         target_data_frequency='Daily',
                         past_data_frequency=['Weekly',
                                              'Weekly',
                                              'Daily',
                                              'Daily',
                                              'Weekly'])
    
    b = a.__getitem__(0)
    c = a.__getitem__(2)

    d = a.collate_fn([b,c])

    print(d.shape)