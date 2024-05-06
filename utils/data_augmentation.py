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
                new_data_point.ts_input = item.cumulative_case_number[i:ts_len+i]
                new_data.append(new_data_point)         
        
        return new_data


    elif method == 'sampling':
        raise NotImplementedError
    else:
        raise NotImplementedError
    