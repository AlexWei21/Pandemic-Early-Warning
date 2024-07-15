import pandas as pd 
import numpy as np
import copy
import torch

def data_augmentation(data: list,
                      method: str = 'shifting',
                      ts_len: int = 30,):

    if method == 'shifting':
        new_data = []
        for item in data:
            for i in range(item.augmentation_length):
                new_data_point = copy.deepcopy(item)
                new_data_point.ts_case_input = item.ts_case_input_full[i:ts_len+i]
                if item.ts_death_input is not None:
                    new_data_point.ts_death_input = item.ts_death_input_full[i:ts_len+i]
                new_data.append(new_data_point)         
        
        return new_data


    elif method == 'masking':
        new_data = []
        for item in data:
            for i in range(item.augmentation_length):
                new_data_point = copy.deepcopy(item)
                new_data_point.ts_case_input[i:i+7] = [0]*7
                if item.ts_death_input is not None:
                    new_data_point.ts_death_input[i:i+7] = [0]*7
                new_data.append(new_data_point)
        
        return new_data
    else:
        raise NotImplementedError
    