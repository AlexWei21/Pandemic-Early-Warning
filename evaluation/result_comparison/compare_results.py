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
    
    combined_df.to_csv(out_dir,
                       index=False)
    
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

    ## Compare Results
    combined_df['selftune_better'] = np.where((combined_df['selftune_outsample_mae'] < combined_df['delphi_outsample_mae']) & (combined_df['selftune_outsample_mae'] < combined_df['guided_outsample_mae']), 1, 0)
    combined_df['guided_better'] = np.where((combined_df['guided_outsample_mae'] < combined_df['delphi_outsample_mae']) & (combined_df['guided_outsample_mae'] < combined_df['selftune_outsample_mae']), 1, 0)

    print(f"Selftune Model does the best in {sum(combined_df['selftune_better'])} / {len(combined_df)} Locations than DELPHI")
    print(f"Guided Model does the best in {sum(combined_df['guided_better'])} / {len(combined_df)} Locations than DELPHI")


if __name__ == '__main__':

    compare_results(delphi_performance_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_46_71_case_only_performance.csv',
                    selftune_model_performance_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/self_tune/validation_location_loss.csv',
                    past_pandemic_guided_model_performance_dir= '/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/validation_location_loss.csv',
                    out_dir='/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/result_comparison/result_comparison.csv')