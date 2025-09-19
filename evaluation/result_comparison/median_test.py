import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import median_test

## 56 Days - 84 Days

delphi_perf_df = pd.read_csv('output/delphi/covid_56_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_56-84/validation_location_loss.csv')
gru_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/gru/10-08-1300_56-84/validation_location_loss.csv')

combined_df = hgdcm_perf_df.merge(delphi_perf_df, 
                                  left_on = ['Country','Domain'], 
                                  right_on=['country','domain'], 
                                  how = 'inner')

combined_df = combined_df.merge(gru_df, 
                                on = ['Country','Domain'])

print("########## 56 - 84 ##########")
print("DELPHI vs HG-DCM:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_x']).pvalue)
print("GRU vs HG-DCM:", median_test(combined_df['OutSample_MAE_y'],
                                       combined_df['OutSample_MAE_x']).pvalue)
print("DELPHI vs GRU:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_y']).pvalue)

## 42 Days - 84 Days

delphi_perf_df = pd.read_csv('output/delphi/covid_42_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_42-84/validation_location_loss.csv')
gru_df = pd.read_csv('output/gru/10-08-1600_42-84/validation_location_loss.csv')

combined_df = hgdcm_perf_df.merge(delphi_perf_df, 
                                  left_on = ['Country','Domain'], 
                                  right_on=['country','domain'], 
                                  how = 'inner')

combined_df = combined_df.merge(gru_df, 
                                on = ['Country','Domain'])

print("########## 42 - 84 ##########")
print("DELPHI vs HG-DCM:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_x']).pvalue)
print("GRU vs HG-DCM:", median_test(combined_df['OutSample_MAE_y'],
                                       combined_df['OutSample_MAE_x']).pvalue)
print("DELPHI vs GRU:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_y']).pvalue)

### 28 - 84 SEIRD
delphi_perf_df = pd.read_csv('output/delphi/covid_28_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('past_guided/covid_09-17-1000_28-84/validation_location_loss.csv')
gru_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/gru/10-08-1600_28-84/validation_location_loss.csv')

combined_df = hgdcm_perf_df.merge(delphi_perf_df, 
                                  left_on = ['Country','Domain'], 
                                  right_on=['country','domain'], 
                                  how = 'inner')

combined_df = combined_df.merge(gru_df, 
                                on = ['Country','Domain'])

print("########## 28 - 84 ##########")
print("DELPHI vs HG-DCM:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_x']).pvalue)
print("GRU vs HG-DCM:", median_test(combined_df['OutSample_MAE_y'],
                                       combined_df['OutSample_MAE_x']).pvalue)
print("DELPHI vs GRU:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_y']).pvalue)

### 14 - 84 SEIRD
delphi_perf_df = pd.read_csv('output/delphi/covid_14_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_14-84/validation_location_loss.csv')
gru_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/gru/10-08-1600_14-84/validation_location_loss.csv')

combined_df = hgdcm_perf_df.merge(delphi_perf_df, 
                                  left_on = ['Country','Domain'], 
                                  right_on=['country','domain'], 
                                  how = 'inner')

combined_df = combined_df.merge(gru_df, 
                                on = ['Country','Domain'])

print("########## 14 - 84 ##########")
print("DELPHI vs HG-DCM:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_x']).pvalue)
print("GRU vs HG-DCM:", median_test(combined_df['OutSample_MAE_y'],
                                       combined_df['OutSample_MAE_x']).pvalue)
print("DELPHI vs GRU:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_y']).pvalue)