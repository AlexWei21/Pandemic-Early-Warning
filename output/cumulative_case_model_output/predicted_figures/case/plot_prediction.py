import pandas as pd 
from matplotlib import pyplot as plt
from data.data_processing_compartment_model import process_data

covid_data = process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_covid_data_objects.pickle',
                                                   raw_data=False)

for item in covid_data:
    if item.domain_name == 'Massachusetts':
        