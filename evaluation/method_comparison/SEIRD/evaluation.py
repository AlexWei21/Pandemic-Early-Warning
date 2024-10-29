import pandas as pd 
import numpy as np

### 56 - 84 SEIRD

performance_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/seird/performance_56_84.csv')

print("56_84 Mean MAE:", np.mean(performance_df['Outsample_MAE']))
print("56_84 Median MAE:",np.median(performance_df['Outsample_MAE']))
print("56_84 Mean MAPE:",np.mean(performance_df['Outsample_MAPE']))
print("56_84 Median MAPE:",np.median(performance_df['Outsample_MAPE']))

### 42 - 84 SEIRD

performance_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/seird/performance_42_84.csv')

print("42_84 Mean MAE:",np.mean(performance_df['Outsample_MAE']))
print("42_84 Median MAE:",np.median(performance_df['Outsample_MAE']))
print("42_84 Mean MAPE:",np.mean(performance_df['Outsample_MAPE']))
print("42_84 Median MAPE:",np.median(performance_df['Outsample_MAPE']))

### 28 - 84 SEIRD

performance_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/seird/performance_28_84.csv')
print("28_84 Mean MAE:",np.mean(performance_df['Outsample_MAE']))
print("28_84 Median MAE:",np.median(performance_df['Outsample_MAE']))
print("28_84 Mean MAPE:",np.mean(performance_df['Outsample_MAPE']))
print("28_84 Median MAPE:",np.median(performance_df['Outsample_MAPE']))

### 14 - 84 SEIRD

performance_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/seird/performance_14_84.csv')

print("14_84 Mean MAE:",np.mean(performance_df['Outsample_MAE']))
print("14_84 Median MAE:",np.median(performance_df['Outsample_MAE']))
print("14_84 Mean MAPE:",np.mean(performance_df['Outsample_MAPE']))
print("14_84 Median MAPE:",np.median(performance_df['Outsample_MAPE']))