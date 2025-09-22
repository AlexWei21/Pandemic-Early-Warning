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
dcm_perf_df = pd.read_csv('output/self_tune/covid_09-18-0900_56-84/best_epoch_validation_location_loss.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_56-84/best_epoch_validation_location_loss.csv')
# hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-20-1000_56-84/best_epoch_validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(dcm_perf_df,
                                  on = ['Country','Domain'], 
                                  how = 'inner')

# Print Metric Comparison Table
print("56 Days - 84 Days DCM vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['OutSample_MAE_y']), np.mean(combined_df['OutSample_MAE_x']))
print("Mean MAPE:", np.mean(combined_df['OutSample_MAPE_y']), np.mean(combined_df['OutSample_MAPE_x']))
print("Median MAE:", np.median(combined_df['OutSample_MAE_y']), np.median(combined_df['OutSample_MAE_x']))
print("Median MAPE:", np.median(combined_df['OutSample_MAPE_y']), np.median(combined_df['OutSample_MAPE_x']))
print("Perform Better:", 
      len(combined_df[combined_df['OutSample_MAPE_y'] < combined_df['OutSample_MAPE_x']]),
      len(combined_df[combined_df['OutSample_MAPE_y'] > combined_df['OutSample_MAPE_x']]))

combined_df['log_OutSample_MAPE_y'] = np.log10(combined_df['OutSample_MAPE_y'])
combined_df['log_OutSample_MAPE_x'] = np.log10(combined_df['OutSample_MAPE_x'])
combined_df['log_OutSample_MAE_y'] = np.log10(combined_df['OutSample_MAE_y'])
combined_df['log_OutSample_MAE_x'] = np.log10(combined_df['OutSample_MAE_x'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_OutSample_MAPE_y",
            label = "DCM",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE_x',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/ablation/mape_distribution_plots/56_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_OutSample_MAE_y",
            label = "DCM",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE_x',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/ablation/mae_distribution_plots/56_84.png")

## 42 Days - 84 Days
# DELPHI Performance
dcm_perf_df = pd.read_csv('output/self_tune/covid_09-18-0900_42-84/best_epoch_validation_location_loss.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_42-84/best_epoch_validation_location_loss.csv')
# hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-20-1000_42-84/best_epoch_validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(dcm_perf_df,
                                  on = ['Country','Domain'], 
                                  how = 'inner')

# Print Metric Comparison Table
print("42 Days - 84 Days DCM vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['OutSample_MAE_y']), np.mean(combined_df['OutSample_MAE_x']))
print("Mean MAPE:", np.mean(combined_df['OutSample_MAPE_y']), np.mean(combined_df['OutSample_MAPE_x']))
print("Median MAE:", np.median(combined_df['OutSample_MAE_y']), np.median(combined_df['OutSample_MAE_x']))
print("Median MAPE:", np.median(combined_df['OutSample_MAPE_y']), np.median(combined_df['OutSample_MAPE_x']))
print("Perform Better:", 
      len(combined_df[combined_df['OutSample_MAPE_y'] < combined_df['OutSample_MAPE_x']]),
      len(combined_df[combined_df['OutSample_MAPE_y'] > combined_df['OutSample_MAPE_x']]))

combined_df['log_OutSample_MAPE_y'] = np.log10(combined_df['OutSample_MAPE_y'])
combined_df['log_OutSample_MAPE_x'] = np.log10(combined_df['OutSample_MAPE_x'])
combined_df['log_OutSample_MAE_y'] = np.log10(combined_df['OutSample_MAE_y'])
combined_df['log_OutSample_MAE_x'] = np.log10(combined_df['OutSample_MAE_x'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_OutSample_MAPE_y",
            label = "DCM",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE_x',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/ablation/mape_distribution_plots/42_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_OutSample_MAE_y",
            label = "DCM",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE_x',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/ablation/mae_distribution_plots/42_84.png")

## 28 Days - 84 Days

# DELPHI Performance
dcm_perf_df = pd.read_csv('output/self_tune/covid_09-18-0900_28-84/best_epoch_validation_location_loss.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_28-84/best_epoch_validation_location_loss.csv')
# hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-20-1000_28-84/best_epoch_validation_location_loss.csv')

# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(dcm_perf_df,
                                  on = ['Country','Domain'], 
                                  how = 'inner')

# Print Metric Comparison Table
print("28 Days - 84 Days DCM vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['OutSample_MAE_y']), np.mean(combined_df['OutSample_MAE_x']))
print("Mean MAPE:", np.mean(combined_df['OutSample_MAPE_y']), np.mean(combined_df['OutSample_MAPE_x']))
print("Median MAE:", np.median(combined_df['OutSample_MAE_y']), np.median(combined_df['OutSample_MAE_x']))
print("Median MAPE:", np.median(combined_df['OutSample_MAPE_y']), np.median(combined_df['OutSample_MAPE_x']))
print("Perform Better:", 
      len(combined_df[combined_df['OutSample_MAPE_y'] < combined_df['OutSample_MAPE_x']]),
      len(combined_df[combined_df['OutSample_MAPE_y'] > combined_df['OutSample_MAPE_x']]))

combined_df['log_OutSample_MAPE_y'] = np.log10(combined_df['OutSample_MAPE_y'])
combined_df['log_OutSample_MAPE_x'] = np.log10(combined_df['OutSample_MAPE_x'])
combined_df['log_OutSample_MAE_y'] = np.log10(combined_df['OutSample_MAE_y'])
combined_df['log_OutSample_MAE_x'] = np.log10(combined_df['OutSample_MAE_x'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_OutSample_MAPE_y",
            label = "DCM",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE_x',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/ablation/mape_distribution_plots/28_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_OutSample_MAE_y",
            label = "DCM",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE_x',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/ablation/mae_distribution_plots/28_84.png")

## 14 Days - 84 Days

# DELPHI Performance
dcm_perf_df = pd.read_csv('output/self_tune/covid_09-18-1000_14-84/best_epoch_validation_location_loss.csv')
hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-17-1000_14-84/best_epoch_validation_location_loss.csv')
# hgdcm_perf_df = pd.read_csv('output/past_guided/covid_09-20-1000_14-84/best_epoch_validation_location_loss.csv')


# Valid Comparison Locations
combined_df = hgdcm_perf_df.merge(dcm_perf_df,
                                  on = ['Country','Domain'], 
                                  how = 'inner')

# Print Metric Comparison Table
print("14 Days - 84 Days DCM vs. HGDCM")
print("Mean MAE:", np.mean(combined_df['OutSample_MAE_y']), np.mean(combined_df['OutSample_MAE_x']))
print("Mean MAPE:", np.mean(combined_df['OutSample_MAPE_y']), np.mean(combined_df['OutSample_MAPE_x']))
print("Median MAE:", np.median(combined_df['OutSample_MAE_y']), np.median(combined_df['OutSample_MAE_x']))
print("Median MAPE:", np.median(combined_df['OutSample_MAPE_y']), np.median(combined_df['OutSample_MAPE_x']))
print("Perform Better:", 
      len(combined_df[combined_df['OutSample_MAPE_y'] < combined_df['OutSample_MAPE_x']]),
      len(combined_df[combined_df['OutSample_MAPE_y'] > combined_df['OutSample_MAPE_x']]))

combined_df['log_OutSample_MAPE_y'] = np.log10(combined_df['OutSample_MAPE_y'])
combined_df['log_OutSample_MAPE_x'] = np.log10(combined_df['OutSample_MAPE_x'])
combined_df['log_OutSample_MAE_y'] = np.log10(combined_df['OutSample_MAE_y'])
combined_df['log_OutSample_MAE_x'] = np.log10(combined_df['OutSample_MAE_x'])

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_OutSample_MAPE_y",
            label = "DCM",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAPE_x',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAPE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/ablation/mape_distribution_plots/14_84.png")

plt.figure(figsize=(6.4,3.2))
ax = plt.subplot(111)
sns.kdeplot(data=combined_df,
            x = "log_OutSample_MAE_y",
            label = "DCM",
            linewidth = 3)
sns.kdeplot(data=combined_df,
            x = 'log_OutSample_MAE_x',
            label = "HG-DCM",
            linewidth = 3)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
plt.xlabel("Out-sample MAE")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/ablation/mae_distribution_plots/14_84.png")
