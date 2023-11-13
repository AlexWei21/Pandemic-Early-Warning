import pandas as pd
import numpy as np

class Pandemic_Data:
    def __init__(self, look_back, pred_len):
        self.x = np.empty((0,look_back), int)
        self.y = np.empty((0,pred_len), int)
        self.meta_data = np.array([])
    
    def get_sample(self,i):
        return self.x[i], self.y[i], self.meta_data[i]


def read_pandemic_ts (file_name, root_dir = None, look_back = 100, pred_len = 200, with_meta_data = False, 
                      moving_average = 1, data_frequency = 'Daily'):

    pandemic_ts_file = pd.read_csv(root_dir + file_name, dtype={'Domain':str})

    data_type = pandemic_ts_file['type'][0]

    country_names = list(set(pandemic_ts_file['Country']))[0:2]

    a = Pandemic_Data(look_back=look_back, pred_len=pred_len)

    for country in country_names:

        processing_country_data = pandemic_ts_file[pandemic_ts_file['Country'] == country]
        
        processing_country_data = processing_country_data.reset_index(drop=True)

        start_idx = processing_country_data.ne(0).idxmax()['number']

        processing_country_data = processing_country_data[start_idx:]

        min_date = min(pd.to_datetime(processing_country_data['date']).dt.date)
        max_date = max(pd.to_datetime(processing_country_data['date']).dt.date)

        print(processing_country_data)

        if (max_date - min_date).days < (look_back + pred_len):
            continue

        look_back_data = processing_country_data['number'][:look_back]
        pred_data = processing_country_data['number'][look_back:look_back + pred_len]

        a.x = np.append(a.x, [look_back_data.to_list()], axis=0)
        a.y = np.append(a.y, [pred_data.to_list()], axis = 0)

    print(a.x)
    print(a.y)

    return 0


if __name__ == '__main__':
    read_pandemic_ts(root_dir='F:/Pandemic-Database/Past_Pandemic_Time_Series_Data/Covid 19/',
                     file_name='Covid_World_Domain_Daily_CumCases.csv')