import pandas as pd
import numpy as np
import plotly.graph_objects as go

def DELPHI_evaluation(pred_case, pred_death, true_case, true_death, metric = 'MAPE'):

    total_loss = (compute_mape(true_case[-15:],
                 pred_case[len(true_case)-15:len(true_case)])
    + compute_mape(true_death[-15:],
                   pred_death[len(true_death)-15:len(true_death)])) / 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = np.arange(1,len(pred_case)),
                             y = true_case,
                             mode = 'markers',
                             name='Observed Infections',
                             line = dict(dash='dot')))
    
    fig.add_trace(go.Scatter(x = np.arange(1,len(pred_case)),
                             y = pred_case,
                             mode = 'markers',
                             name='Predicted Infections',
                             line = dict(dash='dot')))

    fig.add_trace(go.Scatter(x = np.arange(1,len(pred_case)),
                             y = true_death,
                             mode = 'markers',
                             name='Observed Deaths',
                             line = dict(dash='dot')))
    
    fig.add_trace(go.Scatter(x = np.arange(1,len(pred_case)),
                             y = pred_death,
                             mode = 'markers',
                             name='Predicted Deaths',
                             line = dict(dash='dot')))
    
    fig.update_layout(title='DELPHI: Observed vs Fitted',
                       xaxis_title='Day',
                       yaxis_title='Counts',
                       title_x=0.5,
                       width=1000, height=600
                     )
    
    fig.show()
    
    return total_loss

def compute_mape(y_true: list, y_pred: list) -> float:
    """
    Compute the Mean Absolute Percentage Error (MAPE) between two lists of values
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a float corresponding to the MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred)[y_true > 0] / y_true[y_true > 0])) * 100
    return mape

result = np.load('x_sol_final.npy', allow_pickle=True)
pred_case = result[15,:]
pred_death = result[14,:]
true_case = np.load('true_case.npy', allow_pickle=True)
true_death = np.load('true_death.npy', allow_pickle=True)

loss = DELPHI_evaluation(pred_case=pred_case,
                         pred_death=pred_death,
                         true_case=true_case,
                         true_death=true_death)

print(pred_case)
print(true_case)

print(loss)