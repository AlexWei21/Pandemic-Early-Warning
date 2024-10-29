import pandas as pd
import numpy as np

# Four Week Prediction
four_week_prediction = pd.read_csv('/export/home/rcsguest/rcs_zwei/Pandemic-Early-Warning/evaluation/covidhub_comparison/predictions/2020-04-06-COVIDhub-baseline.csv')
four_week_prediction = four_week_prediction[four_week_prediction['target'].str.contains('case')]
four_week_prediction = four_week_prediction[four_week_prediction['type'] == 'point' ]
print(four_week_prediction[four_week_prediction['location'] == 'US'])