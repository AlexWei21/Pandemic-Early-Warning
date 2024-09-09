import pandas as pd 
import numpy as np
from scipy.stats import tukey_hsd
from matplotlib import pyplot as plt

def compare_results(delphi_dir:str,
                    selftune_model_dir:str,
                    past_pandemic_guided_model_dir:str,
                    out_dir:str):

    ## Performance Dir
    delphi_performance_dir = delphi_dir + 'covid_46_71_case_only_performance.csv'
    selftune_model_performance_dir = selftune_model_dir + 'validation_location_loss.csv'
    past_pandemic_guided_model_performance_dir = past_pandemic_guided_model_dir + 'validation_location_loss.csv'
    
    ## Read Performance Dir
    delphi_baseline = pd.read_csv(delphi_performance_dir)
    delphi_baseline = delphi_baseline[['country','domain','train_mae','outsample_mae','train_mape','outsample_mape']]
    delphi_baseline.columns = ['country','domain','delphi_insample_mae','delphi_outsample_mae','delphi_insample_mape','delphi_outsample_mape']

    selftune_model_performance = pd.read_csv(selftune_model_performance_dir)
    selftune_model_performance.columns = ['country','domain','selftune_insample_mae','selftune_outsample_mae','selftune_insample_mape','selftune_outsample_mape']

    guided_model_performance = pd.read_csv(past_pandemic_guided_model_performance_dir)
    guided_model_performance.columns = ['country','domain','guided_insample_mae','guided_outsample_mae','guided_insample_mape','guided_outsample_mape']

    combined_df = delphi_baseline.merge(selftune_model_performance, on = ['country','domain'])
    combined_df = combined_df.merge(guided_model_performance, on = ['country','domain'])
    
    ## Compute Mean MAE & MAPE
    print("---------- MAE / MAPE Comparison ----------")
    delphi_valid_predictions = delphi_baseline[delphi_baseline['delphi_outsample_mae'] != 999] 
    delphi_mean_mae = np.mean(delphi_valid_predictions['delphi_outsample_mae'])
    delphi_mean_mape = np.mean(delphi_valid_predictions['delphi_outsample_mape'])

    selftune_mean_mae = np.mean(selftune_model_performance['selftune_outsample_mae'])
    selftune_mean_mape = np.mean(selftune_model_performance['selftune_outsample_mape'])

    guided_mean_mae = np.mean(guided_model_performance['guided_outsample_mae'])
    guided_mean_mape = np.mean(guided_model_performance['guided_outsample_mape'])

    print(f"DELPHI: Mean MAE = {delphi_mean_mae}, Mean MAPE = {delphi_mean_mape}")
    print(f"Selftune: Mean MAE = {selftune_mean_mae}, Mean MAPE = {selftune_mean_mape}")
    print(f"Past Guided: Mean MAE = {guided_mean_mae}, Mean MAPE = {guided_mean_mape}")

    ## Rank Results
    print("---------- Rank Comparison -----------")
    rank_df = combined_df[['delphi_outsample_mae','selftune_outsample_mae','guided_outsample_mae']].rank(axis=1,
                                                                                                         numeric_only=True)

    delphi_total_rank = sum(rank_df['delphi_outsample_mae']) / len(rank_df)
    selftune_total_rank = sum(rank_df['selftune_outsample_mae']) / len(rank_df)
    guided_total_rank = sum(rank_df['guided_outsample_mae']) / len(rank_df)
    print("DELPHI Total Rank:", delphi_total_rank)
    print("Selftune Total Rank:", selftune_total_rank)
    print("Guided Total Rank:", guided_total_rank)

    ## Compare Results
    combined_df['selftune_delphi_diff'] = combined_df['delphi_outsample_mae'] - combined_df['selftune_outsample_mae']
    combined_df['guided_selftune_diff'] = combined_df['selftune_outsample_mae'] - combined_df['guided_outsample_mae']
    combined_df['guided_delphi_diff'] = combined_df['delphi_outsample_mae'] - combined_df['guided_outsample_mae']

    combined_df['best_method'] = np.nan

    conditions = [(combined_df['delphi_outsample_mae'] < combined_df['selftune_outsample_mae']) & (combined_df['delphi_outsample_mae'] < combined_df['guided_outsample_mae']),
                  (combined_df['selftune_outsample_mae'] < combined_df['delphi_outsample_mae']) & (combined_df['selftune_outsample_mae'] < combined_df['guided_outsample_mae']),
                  (combined_df['guided_outsample_mae'] < combined_df['delphi_outsample_mae']) & (combined_df['guided_outsample_mae'] < combined_df['selftune_outsample_mae'])]

    values = ['DELPHI','Self-tune','Past-Guided']

    combined_df['best_method'] = np.select(conditions,values)
    
    combined_df.to_csv(out_dir,
                       index=False)

    # print(f"Selftune Model does better in these {len(combined_df[combined_df['selftune_delphi_diff']>0])} Locations than DELPHI")
    # print(f"Guided Model does the better in these {len(combined_df[combined_df['guided_selftune_diff']>0])} Locations than Self-tune Model")
    # print(f"Guided Model does the better in these {len(combined_df[combined_df['guided_delphi_diff']>0])} Locations than DELPHI Model")

    equal_threshold = 0.05

    combined_df['delphi_significantly_better_than_self-tune'] = np.where((combined_df['selftune_delphi_diff'] / combined_df['delphi_outsample_mae']) < -equal_threshold, 1, 0)
    combined_df['selftune_significantly_better_than_delphi'] = np.where((combined_df['selftune_delphi_diff'] / combined_df['delphi_outsample_mae']) > equal_threshold, 1, 0)
    combined_df['delphi_selftune_equal'] = np.where((combined_df['delphi_significantly_better_than_self-tune'] + combined_df['selftune_significantly_better_than_delphi']) == 0, 1, 0)

    combined_df['delphi_significantly_better_than_guided'] = np.where((combined_df['guided_delphi_diff'] / combined_df['delphi_outsample_mae']) < -equal_threshold, 1, 0)
    combined_df['guided_significantly_better_than_delphi'] = np.where((combined_df['guided_delphi_diff'] / combined_df['delphi_outsample_mae']) > equal_threshold, 1, 0)
    combined_df['delphi_guided_equal'] = np.where((combined_df['delphi_significantly_better_than_guided'] + combined_df['guided_significantly_better_than_delphi']) == 0, 1, 0)

    print("########## Self-Tune Results ##########")
    print(f"Self-Tune Better: {sum(combined_df['selftune_significantly_better_than_delphi'])}")
    print(f"Self-Tune Equal to DELPHI: {sum(combined_df['delphi_selftune_equal'])}")
    print(f"DELPHI Better: {sum(combined_df['delphi_significantly_better_than_self-tune'])}")

    print("########## Self-Tune Results ##########")
    print(f"Guided Better: {sum(combined_df['guided_significantly_better_than_delphi'])}")
    print(f"Guided Equal to DELPHI: {sum(combined_df['delphi_guided_equal'])}")
    print(f"DELPHI Better: {sum(combined_df['delphi_significantly_better_than_guided'])}")

    ### Analyze Parameter Distribution
    ## Parameter Dir
    delphi_parameter_dir = delphi_dir + 'covid_46_71_case_only_parameters.csv'
    selftune_parameter_dir = selftune_model_dir + 'predicted_params.csv'
    guided_parameter_dir = past_pandemic_guided_model_dir + 'predicted_params.csv'

    delphi_parameter_df = pd.read_csv(delphi_parameter_dir)
    selftune_parameter_df = pd.read_csv(selftune_parameter_dir)
    guided_parameter_df = pd.read_csv(guided_parameter_dir)

    params = ['alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']

    delphi_parameter_df = delphi_parameter_df[['country','domain','alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']]
    delphi_parameter_df.columns = ['Country','Domain','alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']
    delphi_parameter_df = delphi_parameter_df.sort_values(by=['Country','Domain'])
    selftune_parameter_df.columns = ['Country','Domain','alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']
    guided_parameter_df.columns = ['Country','Domain','alpha','days', 'r_s', 'r_dth' ,'p_dth','r_dthdecay','k1','k2' ,'jump' ,'t_jump','std_normal','k3']

    delphi_parameter_df = delphi_parameter_df[delphi_parameter_df['alpha']!=-999]

    print("---------- Parameter Analysis ----------")
    for param in params:
        # delphi_mean = np.mean(delphi_parameter_df[param])
        # selftune_mean = np.mean(selftune_parameter_df[param])
        # guided_mean = np.mean(guided_parameter_df[param])
        # print(f"{param} mean: DELPHI = {delphi_mean}, Selftune = {selftune_mean}, Guided = {guided_mean}")
        print(f"##### {param}##### ")
        res = tukey_hsd(delphi_parameter_df[param],
                        selftune_parameter_df[param],
                        guided_parameter_df[param])
        print(res)

    ### Plot MAE Distribution
    plt.figure()
    ax = plt.subplot(111)
    plt.hist(np.log(combined_df['delphi_outsample_mae']),
            bins = 20,
            alpha = 0.5,
            label = 'DELPHI')
    plt.hist(np.log(combined_df['guided_outsample_mae']),
            bins = 20,
            alpha = 0.5,
            label = 'HG-DCM')
    ax.spines[['right', 'top']].set_visible(False)
    plt.xlabel("Log MAE")
    plt.ylabel("Num of Locations")
    plt.show()
    
    print(combined_df.sort_values(by=['delphi_outsample_mae'],
                                  ascending=False))
    print(combined_df.sort_values(by = ['guided_outsample_mae'],
                                  ascending=False))


if __name__ == '__main__':

    compare_results(delphi_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/',
                    # selftune_model_performance_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/self_tune/07-07-18/validation_location_loss.csv',
                    # selftune_model_performance_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/self_tune/07-19-17/validation_location_loss.csv',
                    selftune_model_dir = '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/self_tune/07-28-16/',
                    past_pandemic_guided_model_dir= '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/07-29-0900/',
                    # past_pandemic_guided_model_performance_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/07-15-1200/validation_location_loss.csv',
                    out_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/result_comparison/result_comparison.csv')