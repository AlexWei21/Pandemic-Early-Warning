import pandas as pd
import numpy as np
from tqdm import tqdm
from data.data import Compartment_Model_Pandemic_Data
from data.data_utils import process_daily_data, process_weekly_data, get_date_from_date_time_list
import pickle

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
                 raw_data = True):
    
    if (raw_data == True) & (cumulative_case_data_filepath is None):
        print("raw data paths are needed when raw_data == True")
        exit(1)
    elif (raw_data == False) & (processed_data_path is None):
        print("processed_data_path is needed when raw_data == False")
        exit(1)

    if raw_data:
        if cumulative_case_data_filepath is not None:
            cumulative_case_data = pd.read_csv(cumulative_case_data_filepath)
        if cumulative_death_data_filepath is not None:
            cumulative_death_data = pd.read_csv(cumulative_death_data_filepath)
        meta_data = pd.read_csv(meta_data_filepath,index_col=0)
        geological_meta_data = pd.read_csv(geological_meta_data_filepath)

        if (cumulative_case_data_filepath is not None) & (cumulative_death_data_filepath is not None):
            full_data = pd.concat([cumulative_case_data,cumulative_death_data])
        elif (cumulative_case_data_filepath is None) & (cumulative_death_data_filepath is not None):
            full_data = cumulative_death_data
        elif (cumulative_case_data_filepath is not None) & (cumulative_death_data_filepath is None):
            full_data = cumulative_case_data

        country_names = full_data['Country'].unique()

        data_list = []

        for country in tqdm (country_names):

            processing_country_data = full_data[full_data['Country'] == country]
            
            # processing_country_data = processing_country_data.reset_index(drop=True)

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
                    
                    data_point.pandemic_meta_data = get_pandemic_meta_data(meta_data_file=meta_data,
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
                                                                            smoothing = True,
                                                                            look_back = len(processing_subdomain_data_cumcase),
                                                                            pred_len = 0,
                                                                            avg_len=7)
                            elif t == 'CumDeaths':
                                cumdeath_data,_,cumdeath_timestamp,_= process_daily_data(processing_subdomain_data_cumdeath,
                                                                            smoothing=True,
                                                                            look_back=len(processing_subdomain_data_cumdeath),
                                                                            pred_len=0,
                                                                            avg_len=7)

                    elif update_frequency == 'Weekly':
                        for t in ts_type:
                            if t == 'CumCases':
                                cumcase_data, _, cumcase_timestamp, _ = process_weekly_data(processing_subdomain_data_cumcase,
                                                                            smoothing = True,
                                                                            look_back = len(processing_subdomain_data_cumcase),
                                                                            pred_len = 0)
                            elif t == 'CumDeaths':
                                cumdeath_data,_,cumdeath_timestamp,_= process_weekly_data(processing_subdomain_data_cumdeath,
                                                                            smoothing=True,
                                                                            look_back=len(processing_subdomain_data_cumdeath),
                                                                            pred_len=0)
                    
                    else:
                        print("Only Daily and Weekly Data are supported.")
                        exit(2)
                    
                    data_point.cumulative_case_number = cumcase_data
                    data_point.cumulative_death_number = cumdeath_data
                    data_point.timestamps = cumcase_timestamp
                    
                    data_list.append(data_point)

        with open(save_file_path, 'wb') as handle:
            pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(processed_data_path,'rb') as file:
            data_list = pickle.load(file)

    return data_list
                
## Edited Jan21
def get_pandemic_meta_data(meta_data_file, pandemic_name, year, country, region):
    meta_data_row = meta_data_file[(meta_data_file['Country'] == country) & (meta_data_file['Pandemic'] == pandemic_name)]
    if len(meta_data_row) > 0:
        meta_data_row_domain = meta_data_row[(meta_data_file['Region'] == region) & (meta_data_file['Pandemic'] == pandemic_name)]
        if len(meta_data_row_domain) == 0:
            print(f"No meta data found for {country} {pandemic_name} data in {year}")
        else:
            meta_data_row = meta_data_row_domain

    meta_data_row = meta_data_row.iloc[0,6:].T
    return(meta_data_row.to_dict())

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

# a = process_data(cumulative_case_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumCases.csv',
#                    cumulative_death_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumDeaths.csv',
#                    meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/past_pandemic_metadata.csv',
#                    geological_meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/Population_Data.csv',
#                    pandemic_name='Covid-19',
#                    update_frequency='Daily',
#                    ts_type=['CumCases','CumDeaths'],
#                    save_file_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_covid_data_objects.pickle',
#                    )

# a = process_data(processed_data_path='/Users/alex/Documents/Github/Hospitalization_Prediction/compartment_model_covid_data_objects.pickle',
#                  raw_data=False)


# a = process_data(cumulative_case_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv',
#                    # cumulative_death_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumDeaths.csv',
#                    meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/past_pandemic_metadata.csv',
#                    geological_meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/Population_Data.csv',
#                    pandemic_name='Ebola',
#                    update_frequency='Weekly',
#                    ts_type=['CumCases'],
#                    save_file_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_ebola_data_objects.pickle'
#                    )

# a = process_data(cumulative_case_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv',
#                    # cumulative_death_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumDeaths.csv',
#                    meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/past_pandemic_metadata.csv',
#                    geological_meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/Population_Data.csv',
#                    pandemic_name='Ebola',
#                    update_frequency='Weekly',
#                    ts_type=['CumCases'],
#                    save_file_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_ebola_data_objects.pickle'
#                    )

# a = process_data(cumulative_case_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Dengue_Fever/Dengue_AMRO_Country_Weekly_CumCases.csv',
#                   # cumulative_death_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumDeaths.csv',
#                   meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/past_pandemic_metadata.csv',
#                   geological_meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/Population_Data.csv',
#                   pandemic_name='Dengue',
#                   update_frequency='Weekly',
#                   ts_type=['CumCases'],
#                   save_file_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_dengue_data_objects.pickle'
#                   )

# a = process_data(cumulative_case_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Monkeypox/Mpox_World_Country_Daily_CumCases.csv',
#                   cumulative_death_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Monkeypox/Mpox_World_Country_Daily_CumDeaths.csv',
#                   meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/past_pandemic_metadata.csv',
#                   geological_meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/Population_Data.csv',
#                   pandemic_name='MPox',
#                   update_frequency='Daily',
#                   ts_type=['CumCases','CumDeaths'],
#                   save_file_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_mpox_data_objects.pickle'
#                  )

# a = process_data(cumulative_case_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/SARS/SARS_World_Country_Daily_CumCases.csv',
#                  # cumulative_death_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Monkeypox/Mpox_World_Country_Daily_CumDeaths.csv',
#                  meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/past_pandemic_metadata.csv',
#                  geological_meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/Population_Data.csv',
#                  pandemic_name='SARS',
#                  update_frequency='Daily',
#                  ts_type=['CumCases'],
#                  save_file_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_sars_data_objects.pickle'
#                  )

# a = process_data(cumulative_case_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Influenza/Influenza_World_Domain_Weekly_CumCases.csv',# 
#                  # cumulative_death_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Monkeypox/Mpox_World_Country_Daily_CumDeaths.csv',
#                  meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/past_pandemic_metadata.csv',
#                  geological_meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/Population_Data.csv',
#                  pandemic_name='Influenza',
#                  update_frequency='Weekly',
#                  ts_type=['CumCases'],
#                  save_file_path='/Users/alex/Documents/Github/Hospitalization_Prediction/data/compartment_model_influenza_data_objects.pickle'
#                  )

# print(len(a))
# print(a[0])