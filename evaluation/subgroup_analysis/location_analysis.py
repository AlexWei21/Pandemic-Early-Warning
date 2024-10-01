import pandas as pd
import numpy as np
from utils.data_processing_compartment_model import process_data
from tqdm import tqdm
from matplotlib import pyplot as plt

## Covid Data Object
covid_data = process_data(processed_data_path = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/data_files/compartment_model_covid_data_objects_no_smoothing.pickle',
                          raw_data = False)

## HGDCM Predictions
hgdcm_two_week_predictions = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/09-24-1300_14-84/validation_location_loss.csv')
hgdcm_four_week_predictions = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/09-12-0900_28-84/validation_location_loss.csv')
hgdcm_six_week_predictions = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/09-12-0900_42-84/validation_location_loss.csv')
hgdcm_eight_week_predictions = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/09-12-0900_56-84/validation_location_loss.csv')

## Delphi Predictions
delphi_two_week_predictions = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_14_84_case_only_performance.csv')
delphi_four_week_predictions = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_28_84_case_only_performance.csv')
delphi_six_week_predictions = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_42_84_case_only_performance.csv')
delphi_eight_week_predictions = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_56_84_case_only_performance.csv')

## Get Population Information
population_list = []
for loc in covid_data:
    population_list.append([loc.country_name, loc.domain_name,loc.population])
population_df = pd.DataFrame(population_list,
                             columns = ['Country','Domain','population'])

population_df['population'] = pd.to_numeric(population_df['population'].str.replace(',',''))

## 2 Weeks
hgdcm_two_week_predictions = hgdcm_two_week_predictions.merge(population_df,
                                                              on = ['Country','Domain'],
                                                              how = 'left')
two_week_comparison_df = hgdcm_two_week_predictions.merge(delphi_two_week_predictions,
                                                          left_on = ['Country','Domain'],
                                                          right_on = ['country','domain'],
                                                          how = 'left')
two_week_comparison_df['close_perf'] = np.where(abs(two_week_comparison_df['OutSample_MAPE'] - two_week_comparison_df['outsample_mape']) < 1, 1, 0)
two_week_comparison_df['hgdcm_better'] = np.where(two_week_comparison_df['OutSample_MAPE'] < two_week_comparison_df['outsample_mape'], 1, 0)

two_week_comparison_df['log_population'] = np.log10(two_week_comparison_df['population'])
two_week_comparison_df['bin'] = pd.cut(two_week_comparison_df['log_population'], 
                                       bins=[-np.inf, 6, 6.5, 7, 7.5, 8, np.inf], 
                                       labels=['< {:.1f}'.format(5.5), 
                                               # '{:.1f} to {:.1f}'.format(5.5, 6), 
                                               '{:.1f} to {:.1f}'.format(6, 6.6),
                                               '{:.1f} to {:.1f}'.format(6.5, 7),
                                               '{:.1f} to {:.1f}'.format(7, 7.5),
                                               '{:.1f} to {:.1f}'.format(7.5, 8),
                                               '> {:.1f}'.format(8)])

proportion_positive = two_week_comparison_df.groupby('bin')['hgdcm_better'].mean()

plt.figure()
ax = plt.subplot(111)
plt.bar(proportion_positive.index, proportion_positive)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Population")
plt.ylabel("Proportion HGDCM Perform Better (%)")

plt.savefig('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/subgroup_analysis/14-84_population_plot.png')

## 4 Weeks
hgdcm_four_week_predictions = hgdcm_four_week_predictions.merge(population_df,
                                                              on = ['Country','Domain'],
                                                              how = 'left')
four_week_comparison_df = hgdcm_four_week_predictions.merge(delphi_four_week_predictions,
                                                          left_on = ['Country','Domain'],
                                                          right_on = ['country','domain'],
                                                          how = 'left')
four_week_comparison_df['close_perf'] = np.where(abs(four_week_comparison_df['OutSample_MAPE'] - four_week_comparison_df['outsample_mape']) < 1, 1, 0)
four_week_comparison_df['hgdcm_better'] = np.where(four_week_comparison_df['OutSample_MAPE'] < four_week_comparison_df['outsample_mape'], 1, 0)

four_week_comparison_df['log_population'] = np.log10(four_week_comparison_df['population'])
four_week_comparison_df['bin'] = pd.cut(four_week_comparison_df['log_population'], 
                                       bins=[-np.inf, 6, 6.5, 7, 7.5, 8, np.inf], 
                                       labels=['< {:.1f}'.format(5.5), 
                                               # '{:.1f} to {:.1f}'.format(5.5, 6), 
                                               '{:.1f} to {:.1f}'.format(6, 6.6),
                                               '{:.1f} to {:.1f}'.format(6.5, 7),
                                               '{:.1f} to {:.1f}'.format(7, 7.5),
                                               '{:.1f} to {:.1f}'.format(7.5, 8),
                                               '> {:.1f}'.format(8)])

proportion_positive = four_week_comparison_df.groupby('bin')['hgdcm_better'].mean()

plt.figure()
ax = plt.subplot(111)
plt.bar(proportion_positive.index, proportion_positive)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Population")
plt.ylabel("Proportion HGDCM Perform Better (%)")

plt.savefig('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/subgroup_analysis/28-84_population_plot.png')

## 6 Weeks
hgdcm_six_week_predictions = hgdcm_six_week_predictions.merge(population_df,
                                                              on = ['Country','Domain'],
                                                              how = 'left')
six_week_comparison_df = hgdcm_six_week_predictions.merge(delphi_six_week_predictions,
                                                          left_on = ['Country','Domain'],
                                                          right_on = ['country','domain'],
                                                          how = 'left')
six_week_comparison_df['close_perf'] = np.where(abs(six_week_comparison_df['OutSample_MAPE'] - six_week_comparison_df['outsample_mape']) < 1, 1, 0)
six_week_comparison_df['hgdcm_better'] = np.where(six_week_comparison_df['OutSample_MAPE'] < six_week_comparison_df['outsample_mape'], 1, 0)

six_week_comparison_df['log_population'] = np.log10(six_week_comparison_df['population'])
six_week_comparison_df['bin'] = pd.cut(six_week_comparison_df['log_population'], 
                                       bins=[-np.inf, 6, 6.5, 7, 7.5, 8, np.inf], 
                                       labels=['< {:.1f}'.format(5.5), 
                                               # '{:.1f} to {:.1f}'.format(5.5, 6), 
                                               '{:.1f} to {:.1f}'.format(6, 6.6),
                                               '{:.1f} to {:.1f}'.format(6.5, 7),
                                               '{:.1f} to {:.1f}'.format(7, 7.5),
                                               '{:.1f} to {:.1f}'.format(7.5, 8),
                                               '> {:.1f}'.format(8)])

proportion_positive = six_week_comparison_df.groupby('bin')['hgdcm_better'].mean()

plt.figure()
ax = plt.subplot(111)
plt.bar(proportion_positive.index, proportion_positive)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Population")
plt.ylabel("Proportion HGDCM Perform Better (%)")

plt.savefig('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/subgroup_analysis/42-84_population_plot.png')

## 8 Weeks
hgdcm_eight_week_predictions = hgdcm_eight_week_predictions.merge(population_df,
                                                              on = ['Country','Domain'],
                                                              how = 'left')
eight_week_comparison_df = hgdcm_eight_week_predictions.merge(delphi_eight_week_predictions,
                                                          left_on = ['Country','Domain'],
                                                          right_on = ['country','domain'],
                                                          how = 'left')
eight_week_comparison_df['close_perf'] = np.where(abs(eight_week_comparison_df['OutSample_MAPE'] - eight_week_comparison_df['outsample_mape']) < 1, 1, 0)
eight_week_comparison_df['hgdcm_better'] = np.where(eight_week_comparison_df['OutSample_MAPE'] < eight_week_comparison_df['outsample_mape'], 1, 0)

eight_week_comparison_df['log_population'] = np.log10(eight_week_comparison_df['population'])
eight_week_comparison_df['bin'] = pd.cut(eight_week_comparison_df['log_population'], 
                                       bins=[-np.inf, 6, 6.5, 7, 7.5, 8, np.inf], 
                                       labels=['< {:.1f}'.format(5.5), 
                                               # '{:.1f} to {:.1f}'.format(5.5, 6), 
                                               '{:.1f} to {:.1f}'.format(6, 6.6),
                                               '{:.1f} to {:.1f}'.format(6.5, 7),
                                               '{:.1f} to {:.1f}'.format(7, 7.5),
                                               '{:.1f} to {:.1f}'.format(7.5, 8),
                                               '> {:.1f}'.format(8)])

proportion_positive = eight_week_comparison_df.groupby('bin')['hgdcm_better'].mean()

plt.figure()
ax = plt.subplot(111)
plt.bar(proportion_positive.index, proportion_positive)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Population")
plt.ylabel("Proportion HGDCM Perform Better (%)")

plt.savefig('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/subgroup_analysis/56-84_population_plot.png')

delphi_two_week_predictions = delphi_two_week_predictions[delphi_two_week_predictions['outsample_mape']!=99999]

