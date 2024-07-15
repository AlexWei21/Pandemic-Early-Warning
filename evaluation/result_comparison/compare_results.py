import pandas as pd 
import numpy as np

def compare_results(delphi_performance_dir:str,
                    selftune_model_performance_dir:str,
                    past_pandemic_guided_model_performance_dir:str,
                    out_dir:str):

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
    rank_df = combined_df[['delphi_outsample_mae','selftune_outsample_mae','guided_outsample_mae']].rank(axis=1,
                                                                                                         numeric_only=True)
    
    delphi_total_rank = sum(rank_df['delphi_outsample_mae'])
    selftune_total_rank = sum(rank_df['selftune_outsample_mae'])
    guided_total_rank = sum(rank_df['guided_outsample_mae'])
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

    print(f"Selftune Model does better in these {len(combined_df[combined_df['selftune_delphi_diff']>0])} Locations than DELPHI")
    print(f"Selftune Model does significantly better in these {len(combined_df[combined_df['selftune_delphi_diff']>500])} Locations than DELPHI")
    print(combined_df.sort_values(by=['selftune_delphi_diff'], ascending=False))
    print(f"Guided Model does the better in these {len(combined_df[combined_df['guided_selftune_diff']>0])} Locations than Self-tune Model")
    print(combined_df.sort_values(by=['guided_selftune_diff'], ascending=False))
    print(f"Guided Model does the better in these {len(combined_df[combined_df['guided_delphi_diff']>0])} Locations than DELPHI Model")
    print(combined_df.sort_values(by=['guided_delphi_diff'], ascending=False))


if __name__ == '__main__':

    compare_results(delphi_performance_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_46_71_case_only_performance.csv',
                    # selftune_model_performance_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/self_tune/validation_location_loss.csv',
                    selftune_model_performance_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/self_tune/07-07-18/validation_location_loss.csv',
                    past_pandemic_guided_model_performance_dir= '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/07-10-0100/validation_location_loss.csv',
                    out_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/result_comparison/result_comparison.csv')