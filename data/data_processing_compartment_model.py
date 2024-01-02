import pandas as pd
import numpy as np

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

