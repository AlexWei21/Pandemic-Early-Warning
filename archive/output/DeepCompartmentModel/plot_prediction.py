import pandas as pd 
import numpy as np
from data.data_processing_compartment_model import process_data
from matplotlib import pyplot as plt

model_prediction = pd.read_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/output/DeepCompartmentModel/case_prediction.csv')

target_pandemic_data = process_data(processed_data_path='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_covid_data_objects_no_smoothing.pickle',
                                        raw_data=False)

train_len = 46
test_len = 71

for item in target_pandemic_data:
    if (item.domain_name == 'Massachusetts'):
        true_case = item.cumulative_case_number
        pred_case = model_prediction[model_prediction['Domain'] == item.domain_name].values

        if len(pred_case) == 1:
            pred_case = pred_case[0]
            pred_case = pred_case[2:]
        elif len(pred_case) == 0:
            print("No Prediction")
        else:
            print("More than 1 Prediction")

        time = np.arange(1,test_len)
        print(time)

        # plt.figure()
        # plt.plot()

        