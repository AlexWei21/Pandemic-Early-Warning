import pandas as pd 
import numpy as np

### 56 - 84 SEIRD

performance_df = pd.read_csv('output/cnn/covid_09-22-0000_56-84/validation_location_loss.csv')

print("56_84 Mean MAE:", np.mean(performance_df['OutSample_MAE']))
print("56_84 Median MAE:",np.median(performance_df['OutSample_MAE']))
print("56_84 Mean MAPE:",np.mean(performance_df['OutSample_MAPE']))
print("56_84 Median MAPE:",np.median(performance_df['OutSample_MAPE']))

### 42 - 84 SEIRD

performance_df = pd.read_csv('output/cnn/covid_09-22-0000_42-84/validation_location_loss.csv')

print("42_84 Mean MAE:",np.mean(performance_df['OutSample_MAE']))
print("42_84 Median MAE:",np.median(performance_df['OutSample_MAE']))
print("42_84 Mean MAPE:",np.mean(performance_df['OutSample_MAPE']))
print("42_84 Median MAPE:",np.median(performance_df['OutSample_MAPE']))

### 28 - 84 SEIRD

performance_df = pd.read_csv('output/cnn/covid_09-22-0000_28-84/validation_location_loss.csv')
print("28_84 Mean MAE:",np.mean(performance_df['OutSample_MAE']))
print("28_84 Median MAE:",np.median(performance_df['OutSample_MAE']))
print("28_84 Mean MAPE:",np.mean(performance_df['OutSample_MAPE']))
print("28_84 Median MAPE:",np.median(performance_df['OutSample_MAPE']))

### 14 - 84 SEIRD

performance_df = pd.read_csv('output/cnn/covid_09-22-0000_14-84/validation_location_loss.csv')

print("14_84 Mean MAE:",np.mean(performance_df['OutSample_MAE']))
print("14_84 Median MAE:",np.median(performance_df['OutSample_MAE']))
print("14_84 Mean MAPE:",np.mean(performance_df['OutSample_MAPE']))
print("14_84 Median MAPE:",np.median(performance_df['OutSample_MAPE']))