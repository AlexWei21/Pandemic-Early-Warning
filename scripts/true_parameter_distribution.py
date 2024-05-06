import pandas as pd 
from matplotlib import pyplot as plt

covid_true_parameter = pd.read_csv("/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_covid.csv")

dengue_true_parameter = pd.read_csv("/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_dengue.csv")
ebola_true_parameter = pd.read_csv("/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_ebola.csv")
influenza_true_parameter = pd.read_csv("/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_influenza.csv")
mpox_true_parameter = pd.read_csv("/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_mpox.csv")
sars_true_parameter = pd.read_csv("/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/DELPHI_params_sars.csv")

past_pandemic_true_parameter = pd.concat([dengue_true_parameter,
                                          ebola_true_parameter,
                                          influenza_true_parameter,
                                          mpox_true_parameter,
                                          sars_true_parameter])

past_pandemic_true_parameter = past_pandemic_true_parameter.reset_index(drop = True)
past_pandemic_true_parameter = past_pandemic_true_parameter[past_pandemic_true_parameter['alpha']!= -999]

parameters = past_pandemic_true_parameter.columns[:12]

bins = 50

for param in parameters:
    
    plt.figure()
    plt.hist(past_pandemic_true_parameter[param],
            alpha = 0.5,
            label = f'Past Pandemic {param} Distribution',
            bins = bins)
    plt.hist(covid_true_parameter[param],
            alpha = 0.5,
            label = f"Covid {param} Distribution",
            bins = bins)
    plt.legend()
    plt.savefig(f'/export/home/dor/zwei/Documents/GitHub/Hospitalization_Prediction/DELPHI_true_parameters/distributions/{param}.png')
