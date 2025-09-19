import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Create a function to convert log values back to real values for the x-axis
def scientific_format(x, _):
    return r'$10^{{{}}}$'.format(int(x))

## 56 Days - 84 Days
# DELPHI Performance
delphi_perf_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/delphi/covid_56_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/past_guided/covid_09-17-1000_56-84/best_epoch_validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("56 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Median MAE:", np.median(combined_df['outsample_mae']), np.median(combined_df['OutSample_MAE']))
print("Median MAPE:", np.median(combined_df['outsample_mape']), np.median(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log10(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log10(combined_df['OutSample_MAPE'])
combined_df['log_outsample_mae'] = np.log10(combined_df['outsample_mae'])
combined_df['log_OutSample_MAE'] = np.log10(combined_df['OutSample_MAE'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mape",
            label = "DELPHI",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/mape_distribution_plots/56_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mae",
            label = "DELPHI",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/mae_distribution_plots/56_84.png")

## 42 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/delphi/covid_42_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/past_guided/covid_09-17-1000_42-84/best_epoch_validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("42 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Median MAE:", np.median(combined_df['outsample_mae']), np.median(combined_df['OutSample_MAE']))
print("Median MAPE:", np.median(combined_df['outsample_mape']), np.median(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log10(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log10(combined_df['OutSample_MAPE'])
combined_df['log_outsample_mae'] = np.log10(combined_df['outsample_mae'])
combined_df['log_OutSample_MAE'] = np.log10(combined_df['OutSample_MAE'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mape",
            label = "DELPHI",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)

# Apply the custom tick formatter to the x-axis
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))

plt.xlabel("Out-sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/mape_distribution_plots/42_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mae",
            label = "DELPHI",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/mae_distribution_plots/42_84.png")

## 28 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/delphi/covid_28_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/past_guided/covid_09-17-1000_28-84/best_epoch_validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("28 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Median MAE:", np.median(combined_df['outsample_mae']), np.median(combined_df['OutSample_MAE']))
print("Median MAPE:", np.median(combined_df['outsample_mape']), np.median(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log10(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log10(combined_df['OutSample_MAPE'])
combined_df['log_outsample_mae'] = np.log10(combined_df['outsample_mae'])
combined_df['log_OutSample_MAE'] = np.log10(combined_df['OutSample_MAE'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mape",
            label = "DELPHI",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/mape_distribution_plots/28_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mae",
            label = "DELPHI",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/mae_distribution_plots/28_84.png")

## 14 Days - 84 Days

# DELPHI Performance
delphi_perf_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/delphi/covid_14_84_case_only_performance.csv')
hgdcm_perf_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/past_guided/covid_09-17-1000_14-84/best_epoch_validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(delphi_perf_df, left_on = ['Country','Domain'], right_on=['country','domain'], how = 'inner')

# Print Metric Comparison Table
print("14 Days - 84 Days DELPHI vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['outsample_mae']), np.mean(combined_df['OutSample_MAE']))
print("Mean MAPE:", np.mean(combined_df['outsample_mape']), np.mean(combined_df['OutSample_MAPE']))
print("Median MAE:", np.median(combined_df['outsample_mae']), np.median(combined_df['OutSample_MAE']))
print("Median MAPE:", np.median(combined_df['outsample_mape']), np.median(combined_df['OutSample_MAPE']))
print("Perform Better:", 
      len(combined_df[combined_df['outsample_mape'] < combined_df['OutSample_MAPE']]),
      len(combined_df[combined_df['outsample_mape'] > combined_df['OutSample_MAPE']]))

combined_df['log_outsample_mape'] = np.log10(combined_df['outsample_mape'])
combined_df['log_OutSample_MAPE'] = np.log10(combined_df['OutSample_MAPE'])
combined_df['log_outsample_mae'] = np.log10(combined_df['outsample_mae'])
combined_df['log_OutSample_MAE'] = np.log10(combined_df['OutSample_MAE'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mape",
            label = "DELPHI",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/mape_distribution_plots/14_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_outsample_mae",
            label = "DELPHI",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/mae_distribution_plots/14_84.png")