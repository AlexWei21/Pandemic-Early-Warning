import wbgapi as wb
from utils.data_processing_compartment_model import process_data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from difflib import get_close_matches

data_file_dir = '/n/data1/hms/dbmi/farhat/alex/Pandemic-Database/Processed_Time_Series_Data/'
data_files = ['Covid_19/Covid_World_Domain_Daily_CumCases.csv',
              'Dengue_Fever/Dengue_AMRO_Country_Weekly_CumCases.csv',
              'Ebola/Ebola_AFRO_Country_Weekly_CumCases.csv',
              'Influenza/Influenza_World_Domain_Weekly_CumCases.csv',
              'Monkeypox/Mpox_World_Country_Daily_CumCases.csv',
              'SARS/SARS_World_Country_Daily_CumCases.csv',
              'Zika/Zika_World_Country_Weekly_Cases.csv']

country_level_metadata = ['EN.POP.SLUM.UR.ZS',
                          'GE.EST',
                          'IS.AIR.DPRT',
                          'NE.DAB.TOTL.CD',
                          'NY.GDP.MKTP.CD',
                          'NY.GDP.PCAP.CD',
                          'SH.MED.PHYS.ZS',
                          'SH.UHC.NOPR.ZS',
                          'SH.XPD.CHEX.PC.CD',
                          'SH.XPD.EHEX.PC.CD',
                          'SH.XPD.GHED.PC.CD',
                          'NY.GNP.PCAP.KD',
                          'EN.POP.DNST']

domain_level_metadata = ['AG.SRF.TOTL.K2',
                         'SP.RUR.TOTL.ZS',
                         'SP.URB.TOTL.IN.ZS']

all_data = pd.DataFrame()
for file in data_files:
    data = pd.read_csv(data_file_dir + file)
    # Concat all files
    all_data = pd.concat([all_data,data],ignore_index=True)

all_data['Year'] = pd.to_datetime(all_data['date']).dt.year
country_year = all_data[['Country','Year']].drop_duplicates(subset=['Country','Year'])
country_list = country_year['Country'].unique().tolist()
year_list = country_year['Year'].unique().tolist()

# Your unique country names from the dataset
dataset_countries = sorted(set(country_list))

# Get World Bank country names and codes
wb_countries = pd.DataFrame(wb.economy.list())
print(wb_countries)
wb_country_names = sorted(wb_countries['value'].tolist())
wb_country_dict = dict(zip(wb_countries['value'], wb_countries['id']))

# Check for missing names (not matched by coder)
unmatched = [name for name in dataset_countries if wb.economy.coder([name])[name] is None]

print("Unmatched countries (not recognized by wbgapi):")
for name in unmatched:
    print(f" - {name}")

print("\nSuggested matches:")
for name in unmatched:
    close = get_close_matches(name, wb_country_names, n=1, cutoff=0.6)
    suggestion = close[0] if close else "No close match"
    print(f" - {name} → {suggestion}")

manual_name_map = {
    'Vietnam': 'Viet Nam',
    'Trinidad/Tobago': 'Trinidad and Tobago',
    'St. Vincent/Grenadines': 'St. Vincent and the Grenadines',
    'St. Kitts/Nevis': 'St. Kitts and Nevis',
    'Turks/Caicos': 'Turks and Caicos Islands',
    'Antigua-Barbuda': 'Antigua and Barbuda',
    'Puerto Rico': 'Puerto Rico (US)'
}

# Apply manual mapping to fix names
fixed_country_list = [manual_name_map.get(name, name) for name in country_list]

# Get country code dict using fixed names
country_code_dict = wb.economy.coder(fixed_country_list)

# Filter valid codes
country_code_list = [code for code in country_code_dict.values() if code is not None]

# Download metadata
print('Downloading Data from WBGAPI')
metadata = wb.data.DataFrame(
    series=country_level_metadata,
    economy=country_code_list,
    time=range(min(year_list) - 5, 2023),
    skipBlanks=True,
    columns='series'
).reset_index()
print('Pulling Data Complete')

# Fill missing values forward by country
metadata = metadata.groupby(['economy'], as_index=False).apply(
    lambda df: df.sort_values('time').ffill(limit=5)
).reset_index(drop=True)

# Reverse mapping from code → country name
name_code_dict = {}
for name, code in country_code_dict.items():
    if code is not None:
        name_code_dict[code] = name

# Insert readable country names
metadata.insert(1, 'country', metadata['economy'].map(name_code_dict))

metadata.to_csv('data_files/country_level_meta_data.csv',
                index=False)

# Normalize values
scaler = MinMaxScaler((0, 1))
metadata[country_level_metadata] = scaler.fit_transform(metadata[country_level_metadata])

# Save to file
metadata.to_csv('data_files/normalized_country_level_meta_data.csv',
                index=False)
