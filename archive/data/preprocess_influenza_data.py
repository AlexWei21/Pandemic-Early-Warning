import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from data.data_processing_compartment_model import process_data

flu_data = pd.read_csv('/export/home/dor/zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/Influenza/Influenza_World_Domain_Weekly_Cases.csv')

flu_data['Domain'] = flu_data['Domain'].fillna('empty')
flu_data['Sub-Domain'] = flu_data['Sub-Domain'].fillna('empty')

us_data = flu_data[(flu_data['Country'] == 'United States')]
us_data = us_data.reset_index(drop=True)

us_data = us_data[us_data['number'] != 'X']
us_data['number'] = us_data['number'].astype(float)

data_objects = []

# for domain in set(us_data['Domain']):
# local_data = us_data[us_data['Domain'] == domain]
local_data = us_data

start_date = min(local_data['date'])
end_date = max(local_data['date'])

season_cut_date = "07-01"

start_year = pd.to_datetime(start_date).year
end_year = pd.to_datetime(end_date).year

years = [year for year in range(start_year, end_year+1)]

for year in years:
    # Get Season Data for that year
    season_start = str(year) + '-' + season_cut_date
    season_end = str(year+1) + '-' + season_cut_date
    season_df = local_data[(local_data['date'] < season_end) & (local_data['date'] > season_start)].reset_index(drop=True)
        
    # Calculate Cumulative Case Number 
    season_df = season_df.sort_values('date')
    season_df['number']=season_df.groupby(['Country','Domain'],as_index=False)['number'].cumsum()
    season_df['type'] = 'Cumulative_Cases'

    season_df['Sub-Domain'] = np.nan
    season_df.loc[season_df['Domain'] == "empty", "Domain"] = np.nan
        
    # season_df.to_csv(f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/influenza_yearly_data/influenza_{year}.csv',
    #              index = False)

    if year == 2010:
        print(season_df[season_df['Domain'] == 'Texas'])
        exit()
    



for year in [2010,2011,2012,2013,2014,2015,2016,2017]:
    process_data(pandemic_name='Influenza',
                 update_frequency='Weekly',
                 ts_type=['CumCases'],
                 cumulative_case_data_filepath=f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/influenza_yearly_data/influenza_{year}.csv',
                 meta_data_filepath='/export/home/dor/zwei/Documents/GitHub/Pandemic-Database/Meta_Data/past_pandemic_metadata.csv',
                 geological_meta_data_filepath='/export/home/dor/zwei/Documents/GitHub/Pandemic-Database/Meta_Data/Population_Data.csv',
                 save_file_path=f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/data/compartment_model_{year}_influenza_data_objects.pickle',
                 true_delphi_parameter_filepath='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_params_2010-2017_influenza.csv'
                 )