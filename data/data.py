import numpy as np
import pandas as pd

class Pandemic_Data():
    def __init__(self, look_back_len, pred_len, meta_data_len):

        super.__init__

        self.pandemic_name = None
        self.country_name = None
        self.domain_name = None
        self.subdomain_name = None
        self.x = np.empty((0,look_back_len), float)
        self.y = np.empty((0,pred_len), float)
        self.time_stamp_x = np.empty((0,look_back_len), pd.Timestamp)
        self.time_stamp_y = np.empty((0,pred_len), pd.Timestamp)
        self.meta_data = np.empty((0,meta_data_len), float)
        self.decoder_input = np.empty((0,pred_len), float)

class Compartment_Model_Pandemic_Data():
    def __init__(self, pandemic_name:str = None, country_name:str = None, domain_name:str = None, subdomain_name:str = None, 
                 start_date:pd.Timestamp = None, end_date:pd.Timestamp = None, update_frequency:str = None, country_meta_data:dict = None, pandemic_meta_data:dict = None,
                 case_number:list = None, cumulative_case_number:list = None, death_number:list = None, cumulative_death_number:list = None, first_day_above_hundred:pd.Timestamp = None,
                 timestamps:list = None):

        ## Pandemic Information
        self.pandemic_name = pandemic_name

        ## Geological Information
        self.country_name = country_name
        self.domain_name = domain_name
        self.subdomain_name = subdomain_name

        ## Time Information
        self.start_date = start_date
        self.first_day_above_hundred = first_day_above_hundred
        self.end_date = end_date
        self.update_frequency = update_frequency

        ## Meta Data
        self.country_meta_data = country_meta_data
        self.pandemic_meta_data = pandemic_meta_data

        ## Time-Series Data
        self.case_number = case_number
        self.cumulative_case_number = cumulative_case_number
        self.death_number = death_number
        self.cumulative_death_number = cumulative_death_number
        self.timestamps = timestamps

    def __str__(self) -> str:
        Pandemic_information = ('pandemic_name: {0} \ncountry_name: {1} \ndomain_name: {2} \nsubdomain_name: {3}').format(self.pandemic_name, self.country_name, self.domain_name, self.subdomain_name)
        Time_information = ('\nstart_date: {0} \nfirst_day_above_hundred: {1} \nend_date: {2} \nupdate_frequency: {3}').format(self.start_date, self.first_day_above_hundred, self.end_date, self.update_frequency)
        Meta_Information = ('\nCountry_Meta_Date: {0} \nPandemic_Meta_data: {1}'.format(self.country_meta_data, self.pandemic_meta_data))
        Time_Series_Data = ('\ncase_number: {0} \ncumulative_case_number: {1} \ndeath_number: {2} \ncumulative_death_number: {3} \ntime_stamps: {4}' ).format(self.case_number, self.cumulative_case_number, self.death_number, self.cumulative_death_number, self.timestamps)
        return '\n' + Pandemic_information + '\n' + Time_information + '\n' + Meta_Information + '\n' + Time_Series_Data + '\n'
    