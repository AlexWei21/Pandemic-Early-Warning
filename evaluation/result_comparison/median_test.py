import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import median_test, ttest_rel, wilcoxon

## 56 Days - 84 Days

delphi_perf_df = pd.read_csv('output/delphi/covid_56_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_56-84/best_epoch_validation_location_loss.csv')
gru_df = pd.read_csv('output/gru/covid_09-19-2000_56-84/validation_location_loss.csv')

combined_df = hgdcm_perf_df.merge(delphi_perf_df, 
                                  left_on = ['Country','Domain'], 
                                  right_on=['country','domain'], 
                                  how = 'inner')

combined_df = combined_df.merge(gru_df, 
                                on = ['Country','Domain'],
                                suffixes=('_hgdcm', '_gru'))

print("########## 56 - 84 ##########")
print("DELPHI vs HG-DCM:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_hgdcm']).pvalue)
print("DELPHI vs HG-DCM:", ttest_rel(combined_df['outsample_mae'],
                                     combined_df['OutSample_MAE_hgdcm']).pvalue)
print("DELPHI vs HG-DCM:", wilcoxon(combined_df['outsample_mae'],
                                    combined_df['OutSample_MAE_hgdcm']).pvalue)
print("GRU vs HG-DCM:", wilcoxon(combined_df['OutSample_MAE_hgdcm'],
                                 combined_df['OutSample_MAE_gru']).pvalue)
print("DELPHI vs GRU:", wilcoxon(combined_df['outsample_mae'],
                                 combined_df['OutSample_MAE_gru']).pvalue)

## 42 Days - 84 Days

delphi_perf_df = pd.read_csv('output/delphi/covid_42_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_42-84/best_epoch_validation_location_loss.csv')
gru_df = pd.read_csv('output/gru/covid_09-19-2000_42-84/validation_location_loss.csv')

combined_df = hgdcm_perf_df.merge(delphi_perf_df, 
                                  left_on = ['Country','Domain'], 
                                  right_on=['country','domain'], 
                                  how = 'inner')

combined_df = combined_df.merge(gru_df, 
                                on = ['Country','Domain'],
                                suffixes=('_hgdcm', '_gru'))

print("########## 42 - 84 ##########")
print("DELPHI vs HG-DCM:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_hgdcm']).pvalue)
print("DELPHI vs HG-DCM:", ttest_rel(combined_df['outsample_mae'],
                                     combined_df['OutSample_MAE_hgdcm']).pvalue)
print("DELPHI vs HG-DCM:", wilcoxon(combined_df['outsample_mae'],
                                    combined_df['OutSample_MAE_hgdcm']).pvalue)
print("GRU vs HG-DCM:", wilcoxon(combined_df['OutSample_MAE_hgdcm'],
                                 combined_df['OutSample_MAE_gru']).pvalue)
print("DELPHI vs GRU:", wilcoxon(combined_df['outsample_mae'],
                                 combined_df['OutSample_MAE_gru']).pvalue)


### 28 - 84 SEIRD
delphi_perf_df = pd.read_csv('output/delphi/covid_28_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_28-84/best_epoch_validation_location_loss.csv')
gru_df = pd.read_csv('output/gru/covid_09-19-2000_28-84/validation_location_loss.csv')

combined_df = hgdcm_perf_df.merge(delphi_perf_df, 
                                  left_on = ['Country','Domain'], 
                                  right_on=['country','domain'], 
                                  how = 'inner')

combined_df = combined_df.merge(gru_df, 
                                on = ['Country','Domain'],
                                suffixes=('_hgdcm', '_gru'))

print("########## 28 - 84 ##########")
print("DELPHI vs HG-DCM:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_hgdcm']).pvalue)
print("DELPHI vs HG-DCM:", ttest_rel(combined_df['outsample_mae'],
                                     combined_df['OutSample_MAE_hgdcm']).pvalue)
print("DELPHI vs HG-DCM:", wilcoxon(combined_df['outsample_mae'],
                                    combined_df['OutSample_MAE_hgdcm']).pvalue)
print("GRU vs HG-DCM:", wilcoxon(combined_df['OutSample_MAE_hgdcm'],
                                 combined_df['OutSample_MAE_gru']).pvalue)
print("DELPHI vs GRU:", wilcoxon(combined_df['outsample_mae'],
                                 combined_df['OutSample_MAE_gru']).pvalue)


### 14 - 84 SEIRD
delphi_perf_df = pd.read_csv('output/delphi/covid_14_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_14-84/best_epoch_validation_location_loss.csv')
gru_df = pd.read_csv('output/gru/covid_09-19-2000_14-84/validation_location_loss.csv')

combined_df = hgdcm_perf_df.merge(delphi_perf_df, 
                                  left_on = ['Country','Domain'], 
                                  right_on=['country','domain'], 
                                  how = 'inner')

combined_df = combined_df.merge(gru_df, 
                                on = ['Country','Domain'],
                                suffixes=('_hgdcm', '_gru'))

print("########## 14 - 84 ##########")
print("DELPHI vs HG-DCM:", median_test(combined_df['outsample_mae'],
                                       combined_df['OutSample_MAE_hgdcm']).pvalue)
print("DELPHI vs HG-DCM:", ttest_rel(combined_df['outsample_mae'],
                                     combined_df['OutSample_MAE_hgdcm']).pvalue)
print("DELPHI vs HG-DCM:", wilcoxon(combined_df['outsample_mae'],
                                    combined_df['OutSample_MAE_hgdcm']).pvalue)
print("GRU vs HG-DCM:", wilcoxon(combined_df['OutSample_MAE_hgdcm'],
                                 combined_df['OutSample_MAE_gru']).pvalue)
print("DELPHI vs GRU:", wilcoxon(combined_df['outsample_mae'],
                                 combined_df['OutSample_MAE_gru']).pvalue)

