import pandas as pd 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

### Analyze Parameter Distribution
## Parameter Dir
delphi_dir = '/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/delphi/'
delphi_parameter_dir = delphi_dir + 'covid_28_84_case_only_parameters.csv'
# selftune_parameter_dir = selftune_model_dir + 'predicted_params.csv'

past_pandemic_guided_model_dir = '/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/output/past_guided/'
guided_parameter_dir = past_pandemic_guided_model_dir + 'covid_09-17-1000_28-84/' + 'predicted_params.csv'

delphi_parameter_df = pd.read_csv(delphi_parameter_dir)
# selftune_parameter_df = pd.read_csv(selftune_parameter_dir)
guided_parameter_df = pd.read_csv(guided_parameter_dir)

params = ['alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']

delphi_parameter_df = delphi_parameter_df[['country','domain','alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']]
delphi_parameter_df.columns = ['Country','Domain','alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']
delphi_parameter_df = delphi_parameter_df.sort_values(by=['Country','Domain'])
# selftune_parameter_df.columns = ['Country','Domain','alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']
guided_parameter_df.columns = ['Country','Domain','alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']

delphi_parameter_df = delphi_parameter_df[delphi_parameter_df['alpha']!=-999]

# def add_significance_annotation(ax, x1, x2, y, p_val, text="*"):
#     ax.plot([x1, x2], [y, y], color='black')  # Draw horizontal line
#     ax.plot([x1,x1],[y,y-0.2], color='black')
#     ax.plot([x2,x2],[y,y-0.2], color='black')
#     # ax.text((x1 + x2) * 0.5, y, text, ha='center', va='bottom')  # Add * or text
#     if p_val < 0.05:
#         ax.text((x1 + x2) * 0.5, y + 0.2, '*', ha='center')

print("---------- Parameter Analysis ----------")
fig, axs = plt.subplots(3,4,
                        figsize=(12.8,9.6))
fig.tight_layout(pad=5.0)

for param, ax in zip(params,axs.ravel()):
    # delphi_mean = np.mean(delphi_parameter_df[param])
    # selftune_mean = np.mean(selftune_parameter_df[param])
    # guided_mean = np.mean(guided_parameter_df[param])
    # print(f"{param} mean: DELPHI = {delphi_mean}, Selftune = {selftune_mean}, Guided = {guided_mean}")
    print(f"##### {param}##### ")
    ax.boxplot([delphi_parameter_df[param], guided_parameter_df[param]],
                labels = ['DELPHI', 'HG-DCM'])
    ax.set_ylabel(param)
    ax.set_ylim(top = max(delphi_parameter_df[param]) * 1.5)
    ax.spines[['right', 'top']].set_visible(False)

    res = stats.mannwhitneyu(delphi_parameter_df[param],
                             # selftune_parameter_df[param],
                             guided_parameter_df[param])
    if res.pvalue < 0.05:
        print("*")

    # if res.pvalue < 0.05:
    #     y_max = max(max(delphi_parameter_df[param]), max(guided_parameter_df[param]))  # Max y-value
    #     add_significance_annotation(ax, 1, 2, y_max * 1.3, res.pvalue)

plt.savefig('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/parameter_analysis/parameter_analysis_28-84.png')
plt.savefig('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/evaluation/parameter_analysis/parameter_analysis_28-84.pdf')
