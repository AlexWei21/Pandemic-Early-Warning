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
            if pd.isna(item.domain_name):
                location = (item.pandemic_name, item.country_name, 'NA')
            else:
                location = (item.pandemic_name, item.country_name, item.domain_name)

            if location in self.dataset:
                self.dataset[location].append(item.idx)
            else:
                self.dataset[location] = [item.idx]

        sample_num = []
        for location, index_list in self.dataset.items():
            print(location, index_list)
            sample_num.append(len(index_list))

        self.min_length = min(sample_num)
        self.max_length = max(sample_num)

        if self.batch_size < len(self.dataset):
            raise Exception(f"Please set batch size = number of (pandemic, locations) pairs, which here is {len(self.dataset)}")

        print(f"The smallest set has {self.min_length} curves")

    
    def __iter__(self):
        batch = []
        for i in range(self.min_length):
            for location, index_list in self.dataset.items():

                batch.append(sample(index_list,1)[0])

                if len(batch) == len(self.dataset):
                    yield batch
                    batch = []
    
    def __len__(self):
        return self.min_length

