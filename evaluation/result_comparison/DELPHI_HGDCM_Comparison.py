import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

## 56 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_56_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/08-28-1100/validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("56 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log(combined_df['OutSample_MAPE'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mape",
            label = "DELPHI",
            linewidth = 3)
# plt.hist(combined_df['log_outsample_mape'],
#          density=True,
#          alpha = 0.5,
#          bins=20)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE',
            label = "HG-DCM",
            linewidth = 3)
# plt.hist(np.log(combined_df['OutSample_MAPE']),
#          density=True,
#          alpha = 0.5,
#          bins=20)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Log Out Sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mape_distribution_plots/56_84.png")

## 42 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_42_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/08-28-1500/validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("42 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log(combined_df['OutSample_MAPE'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mape",
            label = "DELPHI",
            linewidth = 3)
# plt.hist(combined_df['log_outsample_mape'],
#          density=True,
#          alpha = 0.5,
#          bins=20)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE',
            label = "HG-DCM",
            linewidth = 3)
# plt.hist(np.log(combined_df['OutSample_MAPE']),
#          density=True,
#          alpha = 0.5,
#          bins=20)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Log Out Sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mape_distribution_plots/42_84.png")

## 28 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_28_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/08-29-0800_28-84/validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("28 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log(combined_df['OutSample_MAPE'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mape",
            label = "DELPHI",
            linewidth = 3)
# plt.hist(combined_df['log_outsample_mape'],
#          density=True,
#          alpha = 0.5,
#          bins=20)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE',
            label = "HG-DCM",
            linewidth = 3)
# plt.hist(np.log(combined_df['OutSample_MAPE']),
#          density=True,
#          alpha = 0.5,
#          bins=20)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Log Out Sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mape_distribution_plots/28_84.png")