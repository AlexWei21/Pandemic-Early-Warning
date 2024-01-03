from data.data_processing_compartment_model import get_data, get_population
from model.Compartment_Models import SEIRD, SEIRD_solver
from lmfit import minimize, Parameters, Parameter, report_fit
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from utils.compartment_utils import plot_compartment

def error(params, initial_conditions, tspan, data, ode_solver):
    sol = ode_solver(tspan, initial_conditions, params)
    return (sol[:,pd.Series([2,4])] - data).ravel()

def run_compartment(look_back_len, pred_len = None, model = 'SEIRD'):
    
    raw_data = get_data()

    population = get_population()

    if model == 'SEIRD':
        ode_solver = SEIRD_solver

    ## Initiate State Values
    initE = 1000
    initI = 100
    initR = 0
    initN = population
    initD = 0

    initial_conditions = [initE, initI, initR, initN, initD]

    ## Random Initiate Values
    beta = 1.14 
    sigma = 0.02
    gamma = 0.02
    mu = 0.01

    params = Parameters()
    params.add('beta', value=beta, min=0, max=10)
    params.add('sigma', value=sigma, min=0, max=10)
    params.add('gamma', value=gamma, min=0, max=10)
    params.add('mu', value=mu, min=0, max=10)

    tspan = np.arange(0, look_back_len, 1)

    data = raw_data.loc[0:(look_back_len-1), ['case','death']].values

    result = minimize(error, params, args=(initial_conditions, tspan, data, ode_solver), method='leastsq')

    print(result.params)

    print(report_fit(result))

    final = data + result.residual.reshape(data.shape)

    fig = plot_compartment(tspan=tspan, data=data, final=final)

    observed_ID = raw_data.loc[:,['case','death']].values

    tspan_fit_pred = np.arange(0, observed_ID.shape[0], 1)
    params['beta'].value = result.params['beta'].value
    params['sigma'].value = result.params['sigma'].value
    params['gamma'].value = result.params['gamma'].value
    params['mu'].value = result.params['mu'].value
    fitted_predicted = ode_solver(tspan_fit_pred, initial_conditions, params)

    fitted_predicted_ID = fitted_predicted[:,pd.Series([2,4])]

    fig = plot_compartment(tspan=tspan_fit_pred, data=observed_ID, final=fitted_predicted_ID)

    print("\nFitted MAE")
    print('Infected: ', mean_absolute_error(fitted_predicted_ID[:look_back_len, 0], observed_ID[:look_back_len, 0]))
    print('Dead: ', mean_absolute_error(fitted_predicted_ID[:look_back_len, 1], observed_ID[:look_back_len, 1]))

    print("\nFitted RMSE")
    print('Infected: ', mean_squared_error(fitted_predicted_ID[:look_back_len, 0], observed_ID[:look_back_len, 0], squared=False))
    print('Dead: ', mean_squared_error(fitted_predicted_ID[:look_back_len, 1], observed_ID[:look_back_len, 1], squared=False))

    print("\nFitted MAPE")
    print('Infected: ', mean_absolute_percentage_error(fitted_predicted_ID[:look_back_len, 0], observed_ID[:look_back_len, 0]))
    print('Dead: ', mean_absolute_percentage_error(fitted_predicted_ID[:look_back_len, 1], observed_ID[:look_back_len, 1]))
    
    print("\nPredicted MAE")
    print('Infected: ', mean_absolute_error(fitted_predicted_ID[look_back_len:observed_ID.shape[0], 0], observed_ID[look_back_len:, 0]))
    print('Dead: ', mean_absolute_error(fitted_predicted_ID[look_back_len:observed_ID.shape[0], 1], observed_ID[look_back_len:, 1]))

    print("\nPredicted RMSE")
    print('Infected: ', mean_squared_error(fitted_predicted_ID[look_back_len:observed_ID.shape[0], 0], observed_ID[look_back_len:, 0], squared=False))
    print('Dead: ', mean_squared_error(fitted_predicted_ID[look_back_len:observed_ID.shape[0], 1], observed_ID[look_back_len:, 1], squared=False))

    print("\nPredicted MAPE")
    print('Infected: ', mean_absolute_percentage_error(fitted_predicted_ID[look_back_len:observed_ID.shape[0], 0], observed_ID[look_back_len:, 0]))
    print('Dead: ', mean_absolute_percentage_error(fitted_predicted_ID[look_back_len:observed_ID.shape[0], 1], observed_ID[look_back_len:, 1]))


