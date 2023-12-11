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