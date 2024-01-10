import pandas as pd
import numpy as np
from tqdm import tqdm
from data.data import Compartment_Model_Pandemic_Data
from data.data_utils import process_daily_data, process_weekly_data, get_date_from_date_time_list
import pickle

def process_covid_data(meta_data_filepath, geological_meta_data_filepath, cumulative_case_data_filepath, cumulative_death_data_filepath, validcase_threshold=30):
    
    cumulative_case_data = pd.read_csv(cumulative_case_data_filepath)
    cumulative_death_data = pd.read_csv(cumulative_death_data_filepath)
    meta_data = pd.read_csv(meta_data_filepath,index_col=0)
    geological_meta_data = pd.read_csv(geological_meta_data_filepath)

    pandemic_name = 'Covid_19'
    update_frequency = 'Daily'

    full_data = pd.concat([cumulative_case_data,cumulative_death_data])

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

                processing_subdomain_data_cumcase = processing_subdomain_data[processing_subdomain_data['type']=='Cumulative_Cases']
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
                end_date = min(max(pd.to_datetime(processing_subdomain_data_cumcase['date']).dt.date), max(pd.to_datetime(processing_subdomain_data_cumdeath['date']).dt.date))

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
                                                                       country=country)
                
                cumcase_data, _, cumcase_timestamp, _ = process_daily_data(processing_subdomain_data_cumcase,
                                                                           smoothing = True,
                                                                           look_back = len(processing_subdomain_data_cumcase),
                                                                           pred_len = 0,
                                                                           avg_len=7)
                    
                cumdeath_data,_,cumdeath_timestamp,_= process_daily_data(processing_subdomain_data_cumdeath,
                                                                         smoothing=True,
                                                                         look_back=len(processing_subdomain_data_cumdeath),
                                                                         pred_len=0,
                                                                         avg_len=7)
                data_point.cumulative_case_number = cumcase_data
                data_point.cumulative_death_number = cumdeath_data
                data_point.timestamps = cumcase_timestamp
                
                data_list.append(data_point)

    with open('compartment_model_covid_data_objects.pickle', 'wb') as handle:
        pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data_list
                

def get_pandemic_meta_data(meta_data_file, pandemic_name, year, country):
    meta_data_row = meta_data_file[(meta_data_file['Country'] == country) & (meta_data_file['Year'] == year)]
    if len(meta_data_row) == 0:
        print(f"No meta data found for {country} {pandemic_name} data in {year}")
    else:
        meta_data_row = meta_data_row.iloc[0,6:].T
        return(meta_data_row.to_dict())

## TODO: Add population information for: 1. Missing countries or Countries have different names in population file. 2. Domain/Subdomain level population data
def get_population_data(geological_meta_data,year,country,domain,subdomain):
    return 0
    

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

def get_population(country_name='United States', year = 2020):
    population_data = pd.read_csv('F:/Pandemic-Database/Meta_Data/Country_Level_Healthcare_Demographic_Data.csv')
    return(population_data[(population_data['Country'] == country_name) & (population_data['Year'] == year)]['Population'].iloc[0])

# print(get_population())

# process_covid_data(cumulative_case_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumCases.csv',
#                    cumulative_death_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Processed_Time_Series_Data/Covid_19/Covid_World_Domain_Daily_CumDeaths.csv',
#                    meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/past_pandemic_metadata.csv',
#                    geological_meta_data_filepath='/Users/alex/Documents/Github/Past-Pandemic-Metadata/Meta_Data/Country_Level_Healthcare_Demographic_Data.csv')
