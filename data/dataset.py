import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pytorch_lightning import LightningDataModule
from processing import read_pandemic_ts

class Pandemic_Dataset(LightningDataModule):
    def __init__(self, root_dir, file_name, pandemic_name, look_back_len = 100, pred_len = 200, with_meta_data = False, 
                 moving_average = 1, data_smoothing = 'False', data_frequency = 'Daily', meta_data_len = 14,  raw_data = False):
        
        super().__init__

        self.data_dir = root_dir + file_name

        self.look_back_len = look_back_len
        self.pred_len = pred_len
        self.meta_data_len = meta_data_len

        if raw_data == True:
            self.data_list = read_pandemic_ts(data_dir=self.data_dir, pandemic_name=pandemic_name, look_back_len=look_back_len, 
                                            pred_len=pred_len, with_meta_data=with_meta_data, moving_average=moving_average,
                                            data_smoothing=data_smoothing, data_frequency=data_frequency, meta_data_len=meta_data_len)
        else:
            self.data_list = torch.load(self.data_dir)
        
    def __getitem__(self,idx):
        return self.data_list[idx]
    
    def print_data_list(self):
        print(self.data_list)

    def collate_fn(self, batch):       
        
        final_batch = {"x":[],
                       "y":[],
                       "time_stamp_x":[],
                       "time_stamp_y":[],
                       "meta_data":[]}
        
        for i in batch:
            final_batch["x"].append(i.x)
            final_batch["y"].append(i.y)
            final_batch["time_stamp_x"].append(i.time_stamp_x)
            final_batch["time_stamp_y"].append(i.time_stamp_y)
            final_batch["meta_data"].append(i.meta_data)

        return self.to_tensor(final_batch)
    
    def to_tensor(self, target):

        out = {}

        for k,v in target.items():
            if k in ("x","y","meta_data"):
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

    a = Pandemic_Dataset(root_dir = 'F:/Pandemic-Database/Processed_Data/Covid_19/', 
                         file_name = "Covid_World_Domain_Daily_CumCases.csv", 
                         pandemic_name = 'Covid')
    
    b = a.__getitem__(1)
    c = a.__getitem__(2)

    d = a.collate_fn([b,c])

    print(d)