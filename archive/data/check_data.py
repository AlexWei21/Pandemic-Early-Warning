from data.data_processing_compartment_model import process_data
import pickle

train_year = [2010,2011,2012,2013,2014,2015,2016,2017]
test_year = 2018

train_data = []
test_data = process_data(processed_data_path=f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_{test_year}_influenza_data_objects.pickle',
                            raw_data=False)

for year in train_year:
    new_data = process_data(processed_data_path=f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_{year}_influenza_data_objects.pickle',
                            raw_data=False)
    train_data = train_data + new_data

with open('/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_2010-2017_influenza_data_objects.pickle', 'wb') as handle:
    pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(len(train_data))
# print([item for item in flu_data if item.domain_name == 'Massachusetts'][0])