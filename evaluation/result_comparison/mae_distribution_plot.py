import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

### 46-days Prediction
delphi_performance = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/delphi/covid_46_71_case_only_performance.csv')

failed_num = delphi_performance[delphi_performance['outsample_mape'] == 999]

print(delphi_performance.sort_values(by=['outsample_mae'], ascending=False))

print("Delphi 14 Days - 71 Days Forecasting")

print(f"Failed on {len(failed_num)} locations")
print(f"Mean MAE: {np.mean(delphi_performance[delphi_performance['outsample_mape'] != 999]['outsample_mae'])}")
print(f"Mean MAPE: {np.mean(delphi_performance[delphi_performance['outsample_mape'] != 999]['outsample_mape'])}")


hgdcm_performance = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/output/past_guided/07-29-0900/validation_location_loss.csv')
print(hgdcm_performance)

plt.figure()
ax = plt.subplot(111)
plt.hist(np.log(delphi_performance[delphi_performance['outsample_mape'] != 999]['outsample_mae']),
         bins = 20)
plt.hist(np.log(hgdcm_performance['OutSample_MAE']),
         bins=20)
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel("Log MAE")
plt.ylabel("Num of Locations")
plt.show()