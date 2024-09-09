import wbgapi as wb
from utils.data_processing_compartment_model import process_data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Get names of all series
# wb.series.Series().to_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/sample.csv')

data_file_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/'
    
past_pandemic_data = []
past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_dengue_data_objects.pickle',
                                                   raw_data=False))
past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_ebola_data_objects.pickle',
                                                   raw_data=False))
past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_mpox_data_objects.pickle',
                                                   raw_data=False))
past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_sars_data_objects.pickle',
                                                   raw_data=False))
past_pandemic_data.extend(process_data(processed_data_path = data_file_dir+'compartment_model_covid_data_objects_no_smoothing.pickle',
                                                   raw_data=False))
for year in [2010,2011,2012,2013,2014,2015,2016,2017]:
    data = process_data(processed_data_path = data_file_dir+f'compartment_model_{year}_influenza_data_objects.pickle',
                                    raw_data=False)
    for item in data:
        item.pandemic_name = item.pandemic_name + str(year)
        past_pandemic_data.extend(data)

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

country_list = []
year_list = []
for item in past_pandemic_data:
    if item.country_name == 'Vietnam':
        country_list.append('Viet Nam')
    else:
        country_list.append(item.country_name)
    year_list.append(item.start_date.year)


country_code_dict = wb.economy.coder(country_list)
country_code_list = [country_code_dict[item] for item in country_code_dict]

metadata = wb.data.DataFrame(country_level_metadata,
                             country_code_list,
                             time = range(min(year_list)-5,2023,1),
                             skipBlanks=True,
                             columns='series')
metadata = metadata.reset_index()

# metadata.to_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/country_level_meta_data.csv',
#                 index = False)

# metadata = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/country_level_meta_data.csv')
metadata.groupby(['economy','time'],as_index=False).ffill(limit=5)
scaler = MinMaxScaler((0,1))
metadata[country_level_metadata] = scaler.fit_transform(metadata[country_level_metadata])

name_code_dict = {}
for item in country_code_dict:
    if country_code_dict[item] is not None:
        if country_code_dict[item] == 'VNM':
            name_code_dict[country_code_dict[item]] = 'Vietnam'
        else:
            name_code_dict[country_code_dict[item]] = item

metadata.insert(1,'country',[name_code_dict[item] for item in metadata['economy']])

metadata.to_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/normalized_country_level_meta_data.csv',
               index=False)
