import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.io as pio
import requests
from utils.data_processing_compartment_model import process_data
from data.data import Compartment_Model_Pandemic_Dataset
from lmfit import minimize, Parameters, Parameter, report_fit
from tqdm import tqdm

def ode_model(z, t, beta, sigma, gamma, mu):
    """
    Reference https://www.idmod.org/docs/hiv/model-seir.html
    """
    S, E, I, R, D, C = z
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I

    dCdt = sigma*E
    return [dSdt, dEdt, dIdt, dRdt, dDdt, dCdt]

def ode_solver(t, initial_conditions, params):
    initE, initI, initR, initN, initD, initC = initial_conditions
    beta, sigma, gamma, mu = params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
    initS = initN - (initE + initI + initR + initD)
    res = odeint(ode_model, [initS, initE, initI, initR, initD, initC], t, args=(beta, sigma, gamma, mu))
    return res

def error(params, initial_conditions, tspan, data):
    sol = ode_solver(tspan, initial_conditions, params)
    return sol[:, 5] - data

def get_prediction(data_object,
                   train_len,
                   pred_len,):
    initN = int(float(data_object.population.replace(',','')))
    initE = 1000
    initI = 100
    initR = 0
    initD = 0
    initC = 100
    sigma = 1/5.2
    gamma = 1/2.9
    mu = 0.034
    R0 = 4
    beta = R0 * gamma

    true_case = data_object.cumulative_case_number

    params = Parameters()
    params.add('beta', value=beta, min=0, max=10)
    params.add('sigma', value=sigma, min=0, max=10)
    params.add('gamma', value=gamma, min=0, max=10)
    params.add('mu', value=mu, min=0, max=10)

    initial_conditions = [initE, initI, initR, initN, initD, initC]
    beta = 1.14
    sigma = 0.02
    gamma = 0.02
    mu = 0.01
    params['beta'].value = beta
    params['sigma'].value = sigma
    params['gamma'].value = gamma
    params['mu'].value = mu

    tspan = np.arange(0, train_len , 1)

    result = minimize(error, 
                      params, 
                      args=(initial_conditions, tspan, true_case[:train_len]), 
                      method='leastsq')

    tspan_fit_pred = np.arange(0, pred_len, 1)
    params['beta'].value = result.params['beta'].value
    params['sigma'].value = result.params['sigma'].value
    params['gamma'].value = result.params['gamma'].value
    params['mu'].value = result.params['mu'].value
    fitted_predicted = ode_solver(tspan_fit_pred, initial_conditions, params)

    predicted_case = fitted_predicted[:,5]
    true_case = true_case[:pred_len]

    ### Save plot
    plt.figure()
    plt.plot(tspan_fit_pred,
             predicted_case)
    plt.plot(tspan_fit_pred,
             true_case)
    plt.savefig(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/plots/SEIRD/{train_len}_{pred_len}/{data_object.country_name}_{data_object.domain_name}.png')
    plt.close()

    ### Calculate MAE / MAPE
    outsample_mae = np.mean(np.abs(true_case[train_len:] - predicted_case[train_len:]))

    outsample_mape = np.abs((true_case[train_len:] - predicted_case[train_len:]) / true_case[train_len:]) * 100
    outsample_mape = np.mean(outsample_mape)

    return outsample_mae, outsample_mape


if __name__ == '__main__':

    target_training_len = [14,28,42,56]
    pred_len = 84
    data_file_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/data_with_country_metadata/'

    # target_pandemic_data = process_data(processed_data_path=data_file_dir+'compartment_model_covid_data_objects.pickle',
    #                                     raw_data=False)

    target_pandemic_data = process_data(processed_data_path=data_file_dir+'compartment_model_mpox_data_objects.pickle',
                                        raw_data=False)
    
    for train_len in target_training_len:

        location_df = pd.DataFrame(columns=['Country','Domain','Outsample_MAE','Outsample_MAPE'])

        target_pandemic_dataset = Compartment_Model_Pandemic_Dataset(pandemic_data=target_pandemic_data,
                                                target_training_len=train_len,
                                                pred_len = pred_len,
                                                batch_size=64,
                                                meta_data_impute_value=0,
                                                normalize_by_population=False,
                                                input_log_transform=True,)

        for i in tqdm(range(len(target_pandemic_dataset))):
            location_mae, location_mape = get_prediction(target_pandemic_dataset[i],
                                                         train_len = train_len,
                                                         pred_len = pred_len,)
            location_df.loc[len(location_df)] = [target_pandemic_dataset[i].country_name, 
                                                 target_pandemic_dataset[i].domain_name,
                                                 location_mae,
                                                 location_mape]
        
        location_df.to_csv(f'/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/seird/mpox_performance_{train_len}_{pred_len}.csv',
                           index = False)
    
