import pandas as pd
import numpy as np
from tqdm import tqdm
from data.data import Compartment_Model_Pandemic_Data
from utils.data_utils import process_daily_data, process_weekly_data, get_date_from_date_time_list
import pickle
from sklearn.preprocessing import MinMaxScaler
import wbgapi as wb

'''
Function for process raw data into data objects
Input
    pandemic_name: Name for Pandemic
    update_frequency: Daily data update frequency
    ts_type: The type of daily data to include in processing
    meta_data_filepath: Datapath for Meta-data
    population_filepath: File Path for Population
    cumulative_case_data_filepath: File Path for Cumulative Case Number
    cumulative_death_data_filepath: File Path for Cumulative Death Number
    processed_data_path: File Path for saving processed data
    validcase_threshold: The Threshold for considering valid case
    save_file_path: Saving Directory Path
    raw_data: Whether the input is raw data
    true_delphi_parameter_filepath: The file path for true delphi parameter (For Debug and Analysis Only)
    smoothing: Whether to do smoothing for input data
    country_level_metadata_path: Path for country level metadata
'''
def process_data(pandemic_name = 'Covid-19',
                 update_frequency = 'Daily',
                 meta_data_filepath = None, 
                 population_data_filepath = None, 
                 cumulative_case_data_filepath = None, 
                 processed_data_path = None,
                 save_file_path = None,
                 raw_data = True,
                 true_delphi_parameter_filepath = None,
                 smoothing = False,
                 country_level_metadata_path = None,):
    
    if (raw_data == True) & (cumulative_case_data_filepath is None):
        print("raw data paths are needed when raw_data == True")
        exit(1)
    elif (raw_data == False) & (processed_data_path is None):
        print("processed_data_path is needed when raw_data == False")
        exit(1)

    ## Process data if input is raw data
    if raw_data:
        ## Load Cumulative Time Series Data
        if cumulative_case_data_filepath is not None:
            cumulative_case_data = pd.read_csv(cumulative_case_data_filepath)
        if country_level_metadata_path is not None:
            country_level_metadata = pd.read_csv(country_level_metadata_path)

        if true_delphi_parameter_filepath is not None:
            true_parameters = pd.read_csv(true_delphi_parameter_filepath)

        ## Load Meta-Data
        meta_data_file = pd.read_csv(meta_data_filepath,index_col=0)
        population_meta_data = pd.read_csv(population_data_filepath)

        full_data = cumulative_case_data.copy()

        if pandemic_name == 'Influenza':
            full_data['series_id'] = full_data['Country'] + '_' + full_data['Domain'].fillna('Overall') + '_' + full_data['Sub-Domain'].fillna('Overall') + '_' + full_data['season_number'].astype(str)
        else:
            full_data['series_id'] = full_data['Country'] + '_' + full_data['Domain'].fillna('Overall') + '_' + full_data['Sub-Domain'].fillna('Overall')

        full_data = fix_negative_daily_cases(full_data,
                                             pandemic_name=pandemic_name)

        meta_data_file = meta_data_imputation(meta_data_file, normalize=True)


        count = 0
        data_list = []
        for series in full_data['series_id'].unique():
            
            print(f"Processing Series: {series}")
            
            processing_series_data = full_data[full_data['series_id'] == series].copy()
            processing_series_data = data_cleaning(processing_series_data)

            if processing_series_data is None:
                continue

            country = processing_series_data['Country'].iloc[0]
            domain = processing_series_data['Domain'].iloc[0] 
            subdomain = processing_series_data['Sub-Domain'].iloc[0]
            series_id = processing_series_data['series_id'].iloc[0] 

            data_point = Compartment_Model_Pandemic_Data(pandemic_name=pandemic_name,
                                                         country_name=country,
                                                         domain_name=domain,
                                                         subdomain_name=subdomain,
                                                         series_id=series_id,
                                                         update_frequency=update_frequency)

            processing_series_cumcase = processing_series_data[processing_series_data['type']=='Cumulative_Cases'].reset_index(drop=True)

            data_point.start_date = min(pd.to_datetime(processing_series_cumcase['date']).dt.date)
            data_point.end_date = max(pd.to_datetime(processing_series_cumcase['date']).dt.date)

            ## Set the first day that case number exceed 100 as the start date
            first_day_above_hundred = processing_series_cumcase.iloc[np.argmax(processing_series_cumcase['number']>100),:]['date']
            data_point.first_day_above_hundred = pd.to_datetime(first_day_above_hundred).date()

            processing_series_cumcase = processing_series_cumcase[pd.to_datetime(processing_series_cumcase['date']).dt.date >= data_point.first_day_above_hundred]

            data_point.pandemic_meta_data = get_pandemic_meta_data(meta_data_file = meta_data_file,
                                                                   country_level_metadata = country_level_metadata,
                                                                   pandemic_name = pandemic_name,
                                                                   year = data_point.first_day_above_hundred.year,
                                                                   country = country,
                                                                   region = processing_series_data['Region'].iloc[0])

            data_point.population = get_population_data(population_meta_data,country,domain,subdomain)
            if data_point.population is None:
                print(f"[Excluded] {country} | {pandemic_name} | Domain: {domain} — failed Criterion: No Population Data")
                continue

            cumcase_data = None
            cumcase_timestamp = None
            cumdeath_data = None
            
            if update_frequency == 'Daily':
                cumcase_data, _, cumcase_timestamp, _ = process_daily_data(processing_series_cumcase,
                                                                           smoothing = smoothing,
                                                                           look_back = len(processing_series_cumcase),
                                                                           pred_len = 0)
            elif update_frequency == 'Weekly':
                cumcase_data, _, cumcase_timestamp, _ = process_weekly_data(processing_series_cumcase,
                                                                            smoothing = smoothing,
                                                                            look_back = (len(processing_series_cumcase) - 1) * 7 + 1 ,
                                                                            pred_len = 0)            
            else:
                print("Only Daily and Weekly Data are supported.")
                exit(2)
            
            data_point.cumulative_case_number = cumcase_data
            data_point.cumulative_death_number = cumdeath_data
            data_point.timestamps = cumcase_timestamp

            data_list.append(data_point)
            count += 1

        print(f"Total {count} series processed.")

        with open(save_file_path, 'wb') as handle:
            pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(processed_data_path,'rb') as file:
            data_list = pickle.load(file)

    return data_list

def fix_negative_daily_cases(time_series_df:pd.DataFrame, pandemic_name:str):
    
    # Create a column containing information about Country Domain and Sub-Domain
    # if pandemic_name == 'Influenza':
    #     time_series_df['series_id'] = time_series_df['season_id']
    # else:
    #     time_series_df['series_id'] = time_series_df['Country'] + '_' + time_series_df['Domain'].fillna('Overall') + '_' + time_series_df['Sub-Domain'].fillna('Overall')

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

def data_cleaning(data: pd.DataFrame):
    # Focus only on cumulative cases
    cum_case_df = data[data['type'] == 'Cumulative_Cases'].copy()
    cum_case = cum_case_df['number'].values

    # Extract context info for logging
    country = cum_case_df['Country'].iloc[0]
    pandemic = cum_case_df['Microbe Species'].iloc[0]
    domain = cum_case_df['Domain'].iloc[0] if 'Domain' in cum_case_df.columns else None

    # --- Criteria 1: Reject if 5 consecutive negative daily cases ---
    # daily_case = np.diff(cum_case, prepend=cum_case[0])
    # neg_flags = daily_case < 0
    # rolling_neg = np.convolve(neg_flags.astype(int), np.ones(5, dtype=int), mode='valid')
    # if np.any(rolling_neg == 5):
    #     print(f"[Excluded] {country} | {pandemic} | Domain: {domain} — failed Criterion 1: 5 consecutive negative daily cases.")
    #     return None

    # --- Criterion 2: Reject if date span with cum_case > 100 is < 84 days ---
    over_100_mask = cum_case > 100
    if np.any(over_100_mask):
        dates_over_100 = pd.to_datetime(cum_case_df['date'].values[over_100_mask])
        span_days = (dates_over_100.max() - dates_over_100.min()).days
        if span_days < 84:
            print(f"[Excluded] {country} | {pandemic} | Domain: {domain} — failed Criterion 2: date span with cumulative cases > 100 is < 84 days.")
            return None
    else:
        print(f"[Excluded] {country} | {pandemic} | Domain: {domain} — failed Criterion 2: no days with cumulative cases > 100.")
        return None

    # Passed all checks
    return data

def get_pandemic_meta_data(meta_data_file, country_level_metadata, pandemic_name, year, country, region):
    # Manual corrections for country names
    manual_name_map = {
        'Vietnam': 'Viet Nam',
        'Trinidad/Tobago': 'Trinidad and Tobago',
        'St. Vincent/Grenadines': 'St. Vincent and the Grenadines',
        'St. Kitts/Nevis': 'St. Kitts and Nevis',
        'Turks/Caicos': 'Turks and Caicos Islands',
        'Antigua-Barbuda': 'Antigua and Barbuda',
        'Puerto Rico': 'Puerto Rico (US)'
    }

    # Apply manual fix
    fixed_country = manual_name_map.get(country, country)

    # Get the 3-letter WB country code
    country_code_dict = wb.economy.coder([fixed_country])
    country_code = country_code_dict.get(fixed_country)

    if country_code is None:
        print(f"Country code not found for: {country} (fixed as: {fixed_country})")
        return None

    # Ensure year format matches 'YRxxxx'
    year_str = f"YR{min(2022, year)}"

    # Match pandemic metadata
    pandemic_meta_data_row = meta_data_file[
        (meta_data_file['Country'] == country) & 
        (meta_data_file['Pandemic'] == pandemic_name)
    ]

    # Match country-level metadata
    country_meta_data_row = country_level_metadata[
        (country_level_metadata['economy'] == country_code) &
        (country_level_metadata['time'] == year_str)
    ]

    if pandemic_meta_data_row.empty:
        print(f"No pandemic metadata found for {country} - {pandemic_name}")
        return None

    if country_meta_data_row.empty:
        print(f"No country-level metadata found for {country} ({country_code}) in {year_str}")
        country_metadata_dict = {}
    else:
        country_metadata_dict = country_meta_data_row.iloc[0, 3:].to_dict()

    pandemic_metadata_dict = pandemic_meta_data_row.iloc[0, 6:].to_dict()
    pandemic_metadata_dict.update(country_metadata_dict)

    return pandemic_metadata_dict

def get_population_data(population_meta_data,country,domain,subdomain):
    
    if pd.isna(domain):
        population_row = population_meta_data[(population_meta_data['Country'] == country) & (pd.isna(population_meta_data['Domain']))]
    else:
        if pd.isna(subdomain):
            population_row = population_meta_data[(population_meta_data['Country'] == country) & (population_meta_data['Domain'] == domain)]
        else:
            population_row = population_meta_data[(population_meta_data['Country'] == country) & (population_meta_data['Domain'] == domain) & (population_meta_data['Sub-Domain'] == subdomain)]

    population_row = population_row.reset_index(drop=True)

    if len(population_row) == 0:
        return None
    elif len(population_row) > 1:
        print(f"Multiple Population Data found for {country} {domain} {subdomain}, first available data point is provided")
        return population_row.iloc[0,3]
    else:
        return population_row.iloc[0,3]

def meta_data_imputation(meta_data_file: pd.DataFrame,
                         normalize: bool = True):

    meta_data_file['LoS_mean'] = np.where(pd.isna(meta_data_file['LoS_mean']),(meta_data_file['LoS_low'] + meta_data_file['LoS_high'])/2,meta_data_file['LoS_mean'])
    meta_data_file['LoS_high'] = np.where(pd.isna(meta_data_file['LoS_high']),meta_data_file['LoS_mean'],meta_data_file['LoS_high'])
    meta_data_file['LoS_low'] = np.where(pd.isna(meta_data_file['LoS_low']),meta_data_file['LoS_mean'],meta_data_file['LoS_low'])
    meta_data_file['hopitalization_rate_mean'] = np.where(pd.isna(meta_data_file['hopitalization_rate_mean']),(meta_data_file['hopitalization_rate_low'] + meta_data_file['hopitalization_rate_high'])/2,meta_data_file['hopitalization_rate_mean'])
    meta_data_file['hopitalization_rate_high'] = np.where(pd.isna(meta_data_file['hopitalization_rate_high']),meta_data_file['hopitalization_rate_mean'],meta_data_file['hopitalization_rate_high'])
    meta_data_file['hopitalization_rate_low'] = np.where(pd.isna(meta_data_file['hopitalization_rate_low']),meta_data_file['hopitalization_rate_mean'],meta_data_file['hopitalization_rate_low'])
    meta_data_file['R0_mean'] = np.where(pd.isna(meta_data_file['R0_mean']),(meta_data_file['R0_low'] + meta_data_file['R0_high'])/2,meta_data_file['R0_mean'])
    meta_data_file['R0_high'] = np.where(pd.isna(meta_data_file['R0_high']),meta_data_file['R0_mean'],meta_data_file['R0_high'])
    meta_data_file['R0_low'] = np.where(pd.isna(meta_data_file['R0_low']),meta_data_file['R0_mean'],meta_data_file['R0_low'])
    meta_data_file['latent_period_mean'] = np.where(pd.isna(meta_data_file['latent_period_mean']),(meta_data_file['latent_period_low'] + meta_data_file['latent_period_high'])/2,meta_data_file['latent_period_mean'])
    meta_data_file['latent_period_high'] = np.where(pd.isna(meta_data_file['latent_period_high']),meta_data_file['latent_period_mean'],meta_data_file['latent_period_high'])
    meta_data_file['latent_period_low'] = np.where(pd.isna(meta_data_file['latent_period_low']),meta_data_file['latent_period_mean'],meta_data_file['latent_period_low'])
    meta_data_file['incubation_period_mean'] = np.where(pd.isna(meta_data_file['incubation_period_mean']),(meta_data_file['incubation_period_low'] + meta_data_file['incubation_period_high'])/2,meta_data_file['incubation_period_mean'])
    meta_data_file['incubation_period_high'] = np.where(pd.isna(meta_data_file['incubation_period_high']),meta_data_file['incubation_period_mean'],meta_data_file['incubation_period_high'])
    meta_data_file['incubation_period_low'] = np.where(pd.isna(meta_data_file['incubation_period_low']),meta_data_file['incubation_period_mean'],meta_data_file['incubation_period_low'])
    meta_data_file['average_time_to_death_mean'] = np.where(pd.isna(meta_data_file['average_time_to_death_mean']),(meta_data_file['average_time_to_death_low'] + meta_data_file['average_time_to_death_high'])/2,meta_data_file['average_time_to_death_mean'])
    meta_data_file['average_time_to_death_high'] = np.where(pd.isna(meta_data_file['average_time_to_death_high']),meta_data_file['average_time_to_death_mean'],meta_data_file['average_time_to_death_high'])
    meta_data_file['average_time_to_death_low'] = np.where(pd.isna(meta_data_file['average_time_to_death_low']),meta_data_file['average_time_to_death_mean'],meta_data_file['average_time_to_death_low'])
    meta_data_file['average_time_to_discharge_mean'] = np.where(pd.isna(meta_data_file['average_time_to_discharge_mean']),(meta_data_file['average_time_to_discharge_low'] + meta_data_file['average_time_to_discharge_high'])/2,meta_data_file['average_time_to_discharge_mean'])
    meta_data_file['average_time_to_discharge_high'] = np.where(pd.isna(meta_data_file['average_time_to_discharge_high']),meta_data_file['average_time_to_discharge_mean'],meta_data_file['average_time_to_discharge_high'])
    meta_data_file['average_time_to_discharge_low'] = np.where(pd.isna(meta_data_file['average_time_to_discharge_low']),meta_data_file['average_time_to_discharge_mean'],meta_data_file['average_time_to_discharge_low'])
    
    if normalize:
        scaler = MinMaxScaler((0,1))
        meta_data_file[meta_data_file.columns[6:]] = scaler.fit_transform(meta_data_file[meta_data_file.columns[6:]])

    return meta_data_file

if __name__ == '__main__':

    home_dir = '/n/data1/hms/dbmi/farhat/alex/Pandemic-Database'

    # Ebola
    ebola_data = process_data(cumulative_case_data_filepath=f'{home_dir}/Processed_Time_Series_Data/Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv',
                    meta_data_filepath=f'{home_dir}/Meta_Data/past_pandemic_metadata.csv',
                    country_level_metadata_path='data_files/normalized_country_level_meta_data.csv',
                    population_data_filepath=f'{home_dir}/Meta_Data/Population_Data.csv',
                    pandemic_name='Ebola',
                    update_frequency='Weekly',
                    save_file_path='data_files/processed_data/train/compartment_model_ebola_data_objects.pickle',
                    smoothing=True,
                    )

    # Dengue
    dengue_data = process_data(
        cumulative_case_data_filepath=f'{home_dir}/Processed_Time_Series_Data/Dengue_Fever/Dengue_AMRO_Country_Weekly_CumCases.csv',
        meta_data_filepath=f'{home_dir}/Meta_Data/past_pandemic_metadata.csv',
        country_level_metadata_path='data_files/normalized_country_level_meta_data.csv',
        population_data_filepath=f'{home_dir}/Meta_Data/Population_Data.csv',
        pandemic_name='Dengue',
        update_frequency='Weekly',
        save_file_path='data_files/processed_data/train/compartment_model_dengue_data_objects.pickle',
        smoothing=True,
    )

    # SARS
    sars_data = process_data(
        cumulative_case_data_filepath=f'{home_dir}/Processed_Time_Series_Data/SARS/SARS_World_Country_Daily_CumCases.csv',
        # cumulative_case_data_filepath=f'/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/data_files/cleaned_data/SARS_World_Country_Daily_CumCases_cleaned.csv',
        meta_data_filepath=f'{home_dir}/Meta_Data/past_pandemic_metadata.csv',
        country_level_metadata_path='data_files/normalized_country_level_meta_data.csv',
        population_data_filepath=f'{home_dir}/Meta_Data/Population_Data.csv',
        pandemic_name='SARS',
        update_frequency='Daily',
        save_file_path='data_files/processed_data/train/compartment_model_sars_data_objects.pickle',
        smoothing=True,
    )

    # Influenza
    influenza_data = process_data(
        pandemic_name='Influenza',
        update_frequency='Daily',
        cumulative_case_data_filepath='data_files/influenza_seasons.csv',
        meta_data_filepath=f'{home_dir}/Meta_Data/past_pandemic_metadata.csv',
        country_level_metadata_path='data_files/normalized_country_level_meta_data.csv',
        population_data_filepath=f'{home_dir}/Meta_Data/Population_Data.csv',
        save_file_path=f'data_files/processed_data/train/compartment_model_influenza_data_objects.pickle',
        smoothing=True
    )

    # COVID-19
    covid_data = process_data(
        cumulative_case_data_filepath=f'{home_dir}/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumCases.csv',
        meta_data_filepath=f'{home_dir}/Meta_Data/past_pandemic_metadata.csv',
        country_level_metadata_path='data_files/normalized_country_level_meta_data.csv',
        population_data_filepath=f'{home_dir}/Meta_Data/Population_Data.csv',
        pandemic_name='Covid-19',
        update_frequency='Daily',
        save_file_path='data_files/processed_data/train/compartment_model_covid_data_objects.pickle',
        smoothing=True,
    )

    covid_data = process_data(
        cumulative_case_data_filepath=f'{home_dir}/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumCases.csv',
        meta_data_filepath=f'{home_dir}/Meta_Data/past_pandemic_metadata.csv',
        country_level_metadata_path='data_files/normalized_country_level_meta_data.csv',
        population_data_filepath=f'{home_dir}/Meta_Data/Population_Data.csv',
        pandemic_name='Covid-19',
        update_frequency='Daily',
        save_file_path='data_files/processed_data/validation/compartment_model_covid_data_objects.pickle',
        smoothing=False,
    )

    # Mpox
    mpox_data = process_data(
        cumulative_case_data_filepath=f'{home_dir}/Processed_Time_Series_Data/Monkeypox/Mpox_World_Country_Daily_CumCases.csv',
        meta_data_filepath=f'{home_dir}/Meta_Data/past_pandemic_metadata.csv',
        country_level_metadata_path='data_files/normalized_country_level_meta_data.csv',
        population_data_filepath=f'{home_dir}/Meta_Data/Population_Data.csv',
        pandemic_name='MPox',
        update_frequency='Daily',
        save_file_path='data_files/processed_data/validation/compartment_model_mpox_data_objects.pickle',
        smoothing=False,
    )

