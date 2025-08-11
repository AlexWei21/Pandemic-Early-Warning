import pandas as pd
import numpy as np

def remove_negative_daily_cases(time_series_df:pd.DataFrame):
    
    # Create a column containing information about Country Domain and Sub-Domain
    time_series_df['series_id'] = time_series_df['Country'] + '_' + time_series_df['Domain'].fillna('Overall') + '_' + time_series_df['Sub-Domain'].fillna('Overall')

    data_list = []
    for series in time_series_df['series_id'].unique():

        processing_series_data = time_series_df[time_series_df['series_id'] == series].copy()

        processing_series_data['daily_case'] = np.diff(processing_series_data['number'], prepend=0)
        # Check if there are negative daily cases
        if (processing_series_data['daily_case'] < 0).any():
            print(f"Negative daily cases found in series: {series}")
            # Remove negative daily cases by setting them to zero
            processing_series_data.loc[processing_series_data['daily_case'] < 0, 'daily_case'] = 0
            # Recalculate cumulative cases
        processing_series_data['number'] = processing_series_data['daily_case'].cumsum()

        # Concat to one dataframe
        data_list.append(processing_series_data)
    cleaned_df = pd.concat(data_list, ignore_index=True)

    return cleaned_df

home_dir = '/n/data1/hms/dbmi/farhat/alex/Pandemic-Database/Processed_Time_Series_Data/'
data_paths = [
    'SARS/SARS_World_Country_Daily_CumCases.csv',
    'Covid_19/Covid_World_Domain_Daily_CumCases.csv',
    'Covid_19/Covid_World_Domain_Daily_CumDeaths.csv',
    'Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv',
    'Dengue_Fever/Dengue_AMRO_Country_Weekly_CumCases.csv',
    'Monkeypox/Mpox_World_Country_Daily_CumCases.csv',
    'Monkeypox/Mpox_World_Country_Daily_CumDeaths.csv'
    ]

save_dir = '/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/data_files/cleaned_data/'

for raw_data_path in data_paths:
    time_series_df = pd.read_csv(home_dir + raw_data_path)
    # print(time_series_df)
    cleaned_df = remove_negative_daily_cases(time_series_df)
    # print(cleaned_df)
    cleaned_df.to_csv(save_dir + raw_data_path.split('/')[-1].replace('.csv', '_cleaned.csv'), index=False)

