from typing import Iterator
from torch.utils.data.sampler import Sampler
from random import sample
import pandas as pd

class Location_Fixed_Batch_Sampler(Sampler):

    """
        Sample one data point from every location, creating a 
        batch with samples from all locations.
    """

    def __init__(self, dataset, batch_size):
        
        self.batch_size = batch_size

        self.dataset = {}

        for item in dataset:
            item.series_id = item.pandemic_name + '_' + item.series_id
            if item.series_id in self.dataset:
                self.dataset[item.series_id].append(item.idx)
            else:
                self.dataset[item.series_id] = [item.idx]

        sample_num = []
        for location, index_list in self.dataset.items():
            sample_num.append(len(index_list))

        self.min_length = min(sample_num)
        self.max_length = max(sample_num)

        print(f"The smallest set has {self.min_length} curves")
        print(f"The largest set has {self.max_length} curves")

        if self.batch_size < len(self.dataset):
            raise Exception(f"Please set batch size = number of (pandemic, locations) pairs, which here is {len(self.dataset)}")
    
    def __iter__(self):
        batch = []
        for i in range(self.max_length):
            for location, index_list in self.dataset.items():

                batch.append(sample(index_list,1)[0])

                if len(batch) == len(self.dataset):
                    yield batch
                    batch = []
    
    def __len__(self):
        return self.max_length

