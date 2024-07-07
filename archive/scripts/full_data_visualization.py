import pandas as pd 
import numpy as np
from data.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset

pandemic_list = ['covid','sars','dengue','ebola','mpox','influenza']

full_data_df = pd.DataFrame([])

for pandemic in pandemic_list:
    ### Load Perfect Parameters
    perfect_parameter = pd.read_csv(f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_{pandemic}.csv')
    perfect_parameter['pandemic'] = pandemic
    perfect_parameter = perfect_parameter[perfect_parameter['last_15_days_mape'] < 15]
    
    ### Load Time-Series Data and Meta-Data
    pandemic_data_object = process_data(processed_data_path=f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/processed_data/compartment_model_{pandemic}_data_objects.pickle',
                                        raw_data=False)

    ts_meta_data = []

    for location in pandemic_data_object:
        location_data_point = np.append(location.cumulative_case_number[:30], [location.country_name, location.domain_name, pandemic, int(float(location.population.replace(',','')))]).tolist()
        try:
            if len(location_data_point) != 34:
                continue
            meta_data = list(location.pandemic_meta_data.values())
            location_data_point = np.append(location_data_point,meta_data).tolist()
            ts_meta_data.append(location_data_point)
        except:
            print(f"{location.country_name}_{location.domain_name}_doesn't have meta data")

    columns = np.array(["day_" + str(i) for i in range(30)])
    columns = np.append(columns, ['country', 'domain','pandemic','population'])
    columns = np.append(columns, list(pandemic_data_object[0].pandemic_meta_data.keys()))

    ts_meta_data_df = pd.DataFrame(ts_meta_data,
                       columns = columns.tolist(),
                       )
    
    ts_meta_data_df.domain = ts_meta_data_df.domain.astype('str')
    perfect_parameter.domain = perfect_parameter.domain.astype('str')
    
    print(ts_meta_data_df[['country','domain','pandemic']])
    print(perfect_parameter[['country','domain','pandemic']])

    ### Join Dataframes
    combined_df = perfect_parameter.merge(ts_meta_data_df,
                           on = ['country','domain','pandemic'],
                           how = 'inner')

    print(combined_df[['day_1','day_2']])

    full_data_df = pd.concat([full_data_df, combined_df],
                             ignore_index=True)

full_data_df.to_csv('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/clean_model_input_and_delphi_parameters.csv',
                    index = False)

