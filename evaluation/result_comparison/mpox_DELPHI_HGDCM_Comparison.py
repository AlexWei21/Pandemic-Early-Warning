import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.robust import mad

## 56 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/mpox_56_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/mpox_forecasting/past_guided/09-16-1100_56-84/validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("56 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']) ,"+/-", np.std(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']), "+/-", np.std(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Median MAE:", np.median(combined_df['outsample_mae']),"+/-",mad(combined_df['outsample_mae']), np.median(combined_df['OutSample_MAE']), "+/-", mad(combined_df['OutSample_MAE']))
print("Median MAPE:", np.median(combined_df['outsample_mape']), np.median(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log(combined_df['OutSample_MAPE'])
combined_df['log_outsample_mae'] = np.log(combined_df['outsample_mae'])
combined_df['log_OutSample_MAE'] = np.log(combined_df['OutSample_MAE'])

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
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mape_distribution_plots/mpox_56_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mae",
            label = "DELPHI",
            linewidth = 3)
# plt.hist(combined_df['log_outsample_mape'],
#          density=True,
#          alpha = 0.5,
#          bins=20)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE',
            label = "HG-DCM",
            linewidth = 3)
# plt.hist(np.log(combined_df['OutSample_MAPE']),
#          density=True,
#          alpha = 0.5,
#          bins=20)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Log Out Sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mae_distribution_plots/mpox_56_84.png")

## 42 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/mpox_42_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/mpox_forecasting/past_guided/09-14-2300_42-84/validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("42 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']) ,"+/-", np.std(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']), "+/-", np.std(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Median MAE:", np.median(combined_df['outsample_mae']),"+/-",mad(combined_df['outsample_mae']), np.median(combined_df['OutSample_MAE']), "+/-", mad(combined_df['OutSample_MAE']))
print("Median MAPE:", np.median(combined_df['outsample_mape']), np.median(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log(combined_df['OutSample_MAPE'])
combined_df['log_outsample_mae'] = np.log(combined_df['outsample_mae'])
combined_df['log_OutSample_MAE'] = np.log(combined_df['OutSample_MAE'])

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
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mape_distribution_plots/mpox_42_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mae",
            label = "DELPHI",
            linewidth = 3)
# plt.hist(combined_df['log_outsample_mape'],
#          density=True,
#          alpha = 0.5,
#          bins=20)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE',
            label = "HG-DCM",
            linewidth = 3)
# plt.hist(np.log(combined_df['OutSample_MAPE']),
#          density=True,
#          alpha = 0.5,
#          bins=20)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Log Out Sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mae_distribution_plots/mpox_42_84.png")

## 28 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/mpox_28_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/mpox_forecasting/past_guided/10-01-1900_28-84/validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("28 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']) ,"+/-", np.std(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']), "+/-", np.std(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Median MAE:", np.median(combined_df['outsample_mae']),"+/-",mad(combined_df['outsample_mae']), np.median(combined_df['OutSample_MAE']), "+/-", mad(combined_df['OutSample_MAE']))
print("Median MAPE:", np.median(combined_df['outsample_mape']), np.median(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log(combined_df['OutSample_MAPE'])
combined_df['log_outsample_mae'] = np.log(combined_df['outsample_mae'])
combined_df['log_OutSample_MAE'] = np.log(combined_df['OutSample_MAE'])

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
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mape_distribution_plots/mpox_28_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mae",
            label = "DELPHI",
            linewidth = 3)
# plt.hist(combined_df['log_outsample_mape'],
#          density=True,
#          alpha = 0.5,
#          bins=20)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE',
            label = "HG-DCM",
            linewidth = 3)
# plt.hist(np.log(combined_df['OutSample_MAPE']),
#          density=True,
#          alpha = 0.5,
#          bins=20)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Log Out Sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mae_distribution_plots/mpox_28_84.png")
## 14 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/mpox_14_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/mpox_forecasting/past_guided/09-15-2100_14-84/validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("14 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']) ,"+/-", np.std(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']), "+/-", np.std(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Median MAE:", np.median(combined_df['outsample_mae']),"+/-",mad(combined_df['outsample_mae']), np.median(combined_df['OutSample_MAE']), "+/-", mad(combined_df['OutSample_MAE']))
print("Median MAPE:", np.median(combined_df['outsample_mape']), np.median(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log(combined_df['OutSample_MAPE'])
combined_df['log_outsample_mae'] = np.log(combined_df['outsample_mae'])
combined_df['log_OutSample_MAE'] = np.log(combined_df['OutSample_MAE'])

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
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mape_distribution_plots/mpox_14_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mae",
            label = "DELPHI",
            linewidth = 3)
# plt.hist(combined_df['log_outsample_mape'],
#          density=True,
#          alpha = 0.5,
#          bins=20)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE',
            label = "HG-DCM",
            linewidth = 3)
# plt.hist(np.log(combined_df['OutSample_MAPE']),
#          density=True,
#          alpha = 0.5,
#          bins=20)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Log Out Sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/mae_distribution_plots/mpox_14_84.png")