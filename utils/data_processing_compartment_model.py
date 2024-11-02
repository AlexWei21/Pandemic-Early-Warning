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
    geological_meta_data_filepath: File Path for Geological Meta-data
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
                 ts_type:list = ['CumCases'],
                 meta_data_filepath = None, 
                 geological_meta_data_filepath = None, 
                 cumulative_case_data_filepath = None, 
                 cumulative_death_data_filepath = None, 
                 processed_data_path = None,
                 validcase_threshold=30,
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
        if cumulative_death_data_filepath is not None:
            cumulative_death_data = pd.read_csv(cumulative_death_data_filepath)
        if country_level_metadata_path is not None:
            country_level_metadata = pd.read_csv(country_level_metadata_path)

        if true_delphi_parameter_filepath is not None:
            true_parameters = pd.read_csv(true_delphi_parameter_filepath)

        ## Load Meta-Data
        meta_data_file = pd.read_csv(meta_data_filepath,index_col=0)
        geological_meta_data = pd.read_csv(geological_meta_data_filepath)

        if (cumulative_case_data_filepath is not None) & (cumulative_death_data_filepath is not None):
            full_data = pd.concat([cumulative_case_data,cumulative_death_data])
        elif (cumulative_case_data_filepath is None) & (cumulative_death_data_filepath is not None):
            full_data = cumulative_death_data
        elif (cumulative_case_data_filepath is not None) & (cumulative_death_data_filepath is None):
            full_data = cumulative_case_data

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
        
        scaler = MinMaxScaler((0,1))
        meta_data_file[meta_data_file.columns[6:]] = scaler.fit_transform(meta_data_file[meta_data_file.columns[6:]])

        country_names = full_data['Country'].unique()

        ## Iterate through counties to get data
        data_list = []
        for country in tqdm(country_names):

            processing_country_data = full_data[full_data['Country'] == country]

            for domain in processing_country_data['Domain'].unique():

                if pd.isna(domain):
                    processing_domain_data = processing_country_data[processing_country_data['Domain'].isna()]
                else:
                    processing_domain_data = processing_country_data[processing_country_data['Domain'] == domain]

                for subdomain in processing_domain_data['Sub-Domain'].unique():

                    data_point = Compartment_Model_Pandemic_Data(pandemic_name=pandemic_name,
                                                                country_name=country,
                                                                domain_name=domain,
                                                                subdomain_name=subdomain,
                                                                update_frequency=update_frequency)

                    if pd.isna(subdomain):
                        if pd.isna(domain):
                            print(f"Processing {country} Overall Data")
                        else:
                            print(f"Processing {country} {domain} Data")
                        processing_subdomain_data = processing_domain_data[processing_domain_data['Sub-Domain'].isna()]
                    else:
                        print(f"Processing {country} {domain} {subdomain} Data")
                        processing_subdomain_data = processing_domain_data[processing_domain_data['Sub-Domain'] == subdomain]

                    population = get_population_data(geological_meta_data,country,domain,subdomain)

                    if population is None:
                        print(f"No Population Data for {country} {domain} {subdomain}")
                        continue

                    data_point.population = population

                    processing_subdomain_data_cumcase = processing_subdomain_data[(processing_subdomain_data['type']=='Cumulative_Cases') | (processing_subdomain_data['type']=='ILI_Total') ]
                    processing_subdomain_data_cumdeath = processing_subdomain_data[processing_subdomain_data['type']=='Cumulative_Deaths']
                    
                    processing_subdomain_data_cumcase = processing_subdomain_data_cumcase.reset_index(drop=True)
                    processing_subdomain_data_cumdeath = processing_subdomain_data_cumdeath.reset_index(drop=True)

                    ## Set the first day that case number exceed 100 as the start date
                    if max(processing_subdomain_data_cumcase['number']) < 100:
                        print(country,domain,subdomain, "data never exceeded 100 cases")
                        continue
                    else:
                        start_idx = np.argmax(processing_subdomain_data_cumcase['number']>100)

                    start_date = min(pd.to_datetime(processing_subdomain_data_cumcase['date']).dt.date)
                    end_date = max(pd.to_datetime(processing_subdomain_data_cumcase['date']).dt.date)

                    data_point.start_date = start_date
                    data_point.end_date = end_date

                    first_day_above_hundred = processing_subdomain_data_cumcase.iloc[start_idx,:]['date']

                    processing_subdomain_data_cumcase = processing_subdomain_data_cumcase[processing_subdomain_data_cumcase['date'] >= first_day_above_hundred]
                    processing_subdomain_data_cumdeath = processing_subdomain_data_cumdeath[processing_subdomain_data_cumdeath['date'] >= first_day_above_hundred]

                    first_day_above_hundred = pd.to_datetime(first_day_above_hundred).date()
                    
                    data_point.first_day_above_hundred = first_day_above_hundred

                    if (end_date - first_day_above_hundred).days < validcase_threshold:
                        print(country,domain,subdomain, "doesn't contain enough valid days to predict")
                        continue
                    
                    data_point.pandemic_meta_data = get_pandemic_meta_data(meta_data_file=meta_data_file,
                                                                           country_level_metadata = country_level_metadata,
                                                                            pandemic_name=pandemic_name,
                                                                            year=first_day_above_hundred.year,
                                                                            country=country,
                                                                            region=processing_subdomain_data['Region'].iloc[0])
                    
                    cumcase_data = None
                    cumcase_timestamp = None
                    cumdeath_data = None
                   
                    if update_frequency == 'Daily':
                        for t in ts_type:
                            if t == 'CumCases':
                                cumcase_data, _, cumcase_timestamp, _ = process_daily_data(processing_subdomain_data_cumcase,
                                                                            smoothing = smoothing,
                                                                            look_back = len(processing_subdomain_data_cumcase),
                                                                            pred_len = 0,
                                                                            avg_len=7)
                            elif t == 'CumDeaths':
                                cumdeath_data,_,cumdeath_timestamp,_= process_daily_data(processing_subdomain_data_cumdeath,
                                                                            smoothing=smoothing,
                                                                            look_back=len(processing_subdomain_data_cumdeath),
                                                                            pred_len=0,
                                                                            avg_len=7)

                    elif update_frequency == 'Weekly':
                        for t in ts_type:
                            if t == 'CumCases':
                                cumcase_data, _, cumcase_timestamp, _ = process_weekly_data(processing_subdomain_data_cumcase,
                                                                            smoothing = smoothing,
                                                                            look_back = (len(processing_subdomain_data_cumcase) - 1) * 7 + 1 ,
                                                                            pred_len = 0)
                            elif t == 'CumDeaths':
                                cumdeath_data,_,cumdeath_timestamp,_= process_weekly_data(processing_subdomain_data_cumdeath,
                                                                            smoothing=smoothing,
                                                                            look_back=len(processing_subdomain_data_cumdeath),
                                                                            pred_len=0)
                    
                    else:
                        print("Only Daily and Weekly Data are supported.")
                        exit(2)
                    
                    data_point.cumulative_case_number = cumcase_data
                    data_point.cumulative_death_number = cumdeath_data
                    data_point.timestamps = cumcase_timestamp

                    if true_delphi_parameter_filepath is not None:
                        if pd.isna(data_point.domain_name):
                            if pandemic_name == 'Influenza':
                                true_value_row = true_parameters[(true_parameters['country'] == data_point.country_name) & (pd.isna(true_parameters['domain'])) & (true_parameters['year'] == data_point.first_day_above_hundred.year)]
                            else:
                                true_value_row = true_parameters[(true_parameters['country'] == data_point.country_name) & (pd.isna(true_parameters['domain']))]
                            assert len(true_value_row) == 1
                            data_point.true_delphi_params = true_value_row.values.flatten().tolist()[:12]
                        else: 
                            if pandemic_name == 'Influenza':
                                true_value_row =  true_parameters[(true_parameters['country'] == data_point.country_name) & (true_parameters['domain'] == data_point.domain_name) & (true_parameters['year'] == data_point.first_day_above_hundred.year)]                                
                            else:
                                true_value_row =  true_parameters[(true_parameters['country'] == data_point.country_name) & (true_parameters['domain'] == data_point.domain_name)]
                            if len(true_value_row) != 1:
                                continue
                            data_point.true_delphi_params =true_value_row.values.flatten().tolist()[:12]
                    
                    data_list.append(data_point)

        with open(save_file_path, 'wb') as handle:

            pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(processed_data_path,'rb') as file:
            data_list = pickle.load(file)

    return data_list
                
## Edited Jan21
def get_pandemic_meta_data(meta_data_file, country_level_metadata, pandemic_name, year, country, region):

    pandemic_meta_data_row = meta_data_file[(meta_data_file['Country'] == country) & (meta_data_file['Pandemic'] == pandemic_name)]

    country_code = 'VNM' if country == 'Vietname' else wb.economy.coder([country])[country]

    year = min(2022, year)

    country_meta_data_row = country_level_metadata[(country_level_metadata['economy'] == country_code) & (country_level_metadata['time'] == 'YR' + str(year))]

    if len(pandemic_meta_data_row) == 0:
        print(f"No pandemic meta data found for {country} {pandemic_name} data")
        return None
    if len(country_meta_data_row) == 0:
        print(f"No Country Meta Data found for {country} data")
    else:
        pandemic_metadata_dict = pandemic_meta_data_row.iloc[0,6:].T.to_dict()
        country_metadata_dict = country_meta_data_row.iloc[0,3:].T.to_dict()
        pandemic_metadata_dict.update(country_metadata_dict)
        return pandemic_metadata_dict

def get_population_data(geological_meta_data,country,domain,subdomain):
    
    if pd.isna(domain):
        population_row = geological_meta_data[(geological_meta_data['Country'] == country) & (pd.isna(geological_meta_data['Domain']))]
    else:
        if pd.isna(subdomain):
            population_row = geological_meta_data[(geological_meta_data['Country'] == country) & (geological_meta_data['Domain'] == domain)]
        else:
            population_row = geological_meta_data[(geological_meta_data['Country'] == country) & (geological_meta_data['Domain'] == domain) & (geological_meta_data['Sub-Domain'] == subdomain)]

    population_row = population_row.reset_index(drop=True)

    if len(population_row) == 0:
        return None
    elif len(population_row) > 1:
        print(f"Multiple Population Data found for {country} {domain} {subdomain}, first available data point is provided")
        return population_row.iloc[0,3]
    else:
        return population_row.iloc[0,3]
    

def get_data(pandemic_name='covid',country_name='United States of America',domain_name=None, sub_domain_name=None):

    covid_cumcase_data = pd.read_csv("F:/Pandemic-Database/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumCases.csv", dtype={'Domain':str})
    covid_cumdeath_data = pd.read_csv("F:/Pandemic-Database/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumDeaths.csv", dtype={'Domain':str})
    covid_cumcase_data = covid_cumcase_data[['Country','Domain','Sub-Domain','date','number']]
    covid_cumcase_data = covid_cumcase_data.rename(columns={'number':'case'})

    covid_cumdeath_data = covid_cumdeath_data[['Country','Domain','Sub-Domain','date','number']]
    covid_cumdeath_data = covid_cumdeath_data.rename(columns={'number':'death'})

    covid_data = covid_cumcase_data.merge(covid_cumdeath_data, on = ['Country','Domain','Sub-Domain','date'], how='inner')

    sample_data = covid_data[covid_data['Country']==country_name]

    if domain_name is None:
        sample_data = sample_data[sample_data['Domain'].isna()]
    else:
        sample_data = sample_data[sample_data['Domain']==domain_name]
    
    if sub_domain_name is None:
        sample_data = sample_data[sample_data['Sub-Domain'].isna()]
    else:
        sample_data = sample_data[sample_data['Sub-Domain']==domain_name]

    sample_data = sample_data[sample_data['case'] > 100]

    sample_data = sample_data.reset_index(drop=True)

    return sample_data

# print(get_data())

# print(get_population())

if __name__ == '__main__':

    # covid_data = process_data(cumulative_case_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumCases.csv',
    #                 cumulative_death_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumDeaths.csv',
    #                 meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/past_pandemic_metadata.csv',
    #                 country_level_metadata_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/normalized_country_level_meta_data.csv',
    #                 geological_meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/Population_Data.csv',
    #                 pandemic_name='Covid-19',
    #                 update_frequency='Daily',
    #                 ts_type=['CumCases','CumDeaths'],
    #                 save_file_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/covid_data_objects.pickle',
    #                 smoothing=False,
    #                 # true_delphi_parameter_filepath='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_covid.csv'
    #                 )

    ebola_data = process_data(cumulative_case_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv',
                    meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/past_pandemic_metadata.csv',
                    country_level_metadata_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/normalized_country_level_meta_data.csv',
                    geological_meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/Population_Data.csv',
                    pandemic_name='Ebola',
                    update_frequency='Weekly',
                    ts_type=['CumCases'],
                    save_file_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data/compartment_model_ebola_data_objects.pickle',
                    smoothing=True,
                    # true_delphi_parameter_filepath='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_ebola.csv',
                    )

    dengue_data = process_data(cumulative_case_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/Dengue_Fever/Dengue_AMRO_Country_Weekly_CumCases.csv',
                    meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/past_pandemic_metadata.csv',
                    country_level_metadata_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/normalized_country_level_meta_data.csv',
                    geological_meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/Population_Data.csv',
                    pandemic_name='Dengue',
                    update_frequency='Weekly',
                    ts_type=['CumCases'],
                    smoothing=True,
                    save_file_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/compartment_model_dengue_data_objects.pickle',
                    # true_delphi_parameter_filepath='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_dengue.csv'
                    )

    mpox_data = process_data(cumulative_case_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/Monkeypox/Mpox_World_Country_Daily_CumCases.csv',
                    cumulative_death_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/Monkeypox/Mpox_World_Country_Daily_CumDeaths.csv',
                    meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/past_pandemic_metadata.csv',
                    country_level_metadata_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/normalized_country_level_meta_data.csv',
                    geological_meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/Population_Data.csv',
                    pandemic_name='MPox',
                    update_frequency='Daily',
                    ts_type=['CumCases','CumDeaths'],
                    smoothing=True,
                    save_file_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/compartment_model_mpox_data_objects.pickle',
                   # true_delphi_parameter_filepath='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_mpox.csv'
                    )

    sars_data = process_data(cumulative_case_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Processed_Time_Series_Data/SARS/SARS_World_Country_Daily_CumCases.csv',
                    meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/past_pandemic_metadata.csv',
                    country_level_metadata_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/normalized_country_level_meta_data.csv',
                    geological_meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/Population_Data.csv',
                    pandemic_name='SARS',
                    update_frequency='Daily',
                    ts_type=['CumCases'],
                    save_file_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/compartment_model_sars_data_objects.pickle',
                    smoothing=True,
                    # true_delphi_parameter_filepath='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_sars.csv'
                    )

    for year in [2010,2011,2012,2013,2014,2015,2016,2017]:
        process_data(pandemic_name='Influenza',
                    update_frequency='Weekly',
                    ts_type=['CumCases'],
                    cumulative_case_data_filepath=f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/influenza_yearly_data/influenza_{year}.csv',
                    meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/past_pandemic_metadata.csv',
                    country_level_metadata_path='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/normalized_country_level_meta_data.csv',
                    geological_meta_data_filepath='/export/home/rcsguest/rcs_zwei/Documents/GitHub/Pandemic-Database/Meta_Data/Population_Data.csv',
                    save_file_path=f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/compartment_model_{year}_influenza_data_objects.pickle',
                    smoothing=True,
                    # true_delphi_parameter_filepath='/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_params_2010-2017_influenza.csv'
                    )

    # print('Covid No Meta Data:', [item.country_name for item in covid_data if item.pandemic_meta_data is None])
    # print('Ebola No Meta Data:', [item.country_name for item in ebola_data if item.pandemic_meta_data is None])
    # print('Dengue No Meta Data:', [item.country_name for item in dengue_data if item.pandemic_meta_data is None])
    # print('Mpox No Meta Data:', [item.country_name for item in mpox_data if item.pandemic_meta_data is None])
    # print('SARS No Meta Data:', [item.country_name for item in sars_data if item.pandemic_meta_data is None])
    ## print('Influenza No Meta Data:', [item.country_name for item in influenza_data if item.pandemic_meta_data is None])

    #print(influenza_data[0].country_name, influenza_data[0].domain_name, influenza_data[0].true_delphi_params)