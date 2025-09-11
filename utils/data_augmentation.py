import pandas as pd 
import numpy as np
import copy
import torch
from scipy.signal import find_peaks

def find_last_augmentation_date(pandemic_data_item):
    # Calculate daily cases
    daily_cases = np.diff(pandemic_data_item.cumulative_case_number, prepend=pandemic_data_item.cumulative_case_number[0])

    # Optional: smooth to avoid false peaks in noisy data
    window = 7
    smoothed_daily = pd.Series(daily_cases).rolling(window, center=True, min_periods=1).mean().values

    # Step 1: Find the first major peak (first wave's peak)
    peaks, _ = find_peaks(smoothed_daily, height=np.max(smoothed_daily)*0.25)  # Adjust 0.25 if needed
    if len(peaks) == 0:
        raise ValueError("No peaks found in daily cases, check your data!")

    first_peak_idx = peaks[0]

    # Find the first local minimum after the first peak (end of first wave)
    minima, _ = find_peaks(-smoothed_daily[first_peak_idx:])
    if len(minima) == 0:
        wave_end_idx = len(smoothed_daily)  # Use the whole series if no minimum
    else:
        wave_end_idx = first_peak_idx + minima[0]

    # Step 2: Find the max daily case IN the first wave period
    max_idx_in_wave = np.argmax(smoothed_daily[:wave_end_idx+1])
    # last_day_of_augmentation = dates[max_idx_in_wave]

    # print("Last day of augmentation (max daily case in first wave):", last_day_of_augmentation)

    return max_idx_in_wave

def data_augmentation(data: list,
                      method: str = 'shifting',
                      ts_len: int = 28,
                      pred_len: int = 84,):

    if method == 'shifting':
        new_data = []
        for item in data:
            for i in range(item.last_augmentation_idx - pred_len + 1):
                new_data_point = copy.deepcopy(item)
                new_data_point.ts_case_input = item.daily_case_list[i:ts_len+i]
                if item.ts_death_input is not None:
                    new_data_point.ts_death_input = item.ts_death_input_full[i:ts_len+i]
                new_data.append(new_data_point)         
        
        return new_data


    elif method == 'masking':
        new_data = []
        for item in data:
            new_data.append(copy.deepcopy(item))
            for i in range(ts_len - 7 + 1):
                new_data_point = copy.deepcopy(item)
                new_data_point.ts_case_input[i:i+7] = [0] * 7
                if item.ts_death_input is not None:
                    new_data_point.ts_death_input[i:i+7] = [0] * 7
                new_data.append(new_data_point)
        return new_data
    else:
        raise NotImplementedError
    