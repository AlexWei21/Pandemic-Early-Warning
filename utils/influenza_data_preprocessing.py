import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import numpy as np

df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Database/Processed_Time_Series_Data/Influenza/Influenza_World_Domain_Weekly_CumCases.csv')
df['date'] = pd.to_datetime(df['date'])
df['number'] = pd.to_numeric(df['number'], errors='coerce')

# Fill NAs in group keys if needed
group_cols = ['Region', 'Country', 'Domain', 'Sub-Domain']
df[group_cols] = df[group_cols].fillna('')

season_case_records = []
# Process each location
for keys, group in df.groupby(group_cols):
    group = group.sort_values('date')
    # Set index to date
    group = group.set_index('date')
    group = group.groupby(group.index).agg({'number': 'sum'})

    # Only cumulative data assumed, interpolate over full date range
    full_range = pd.date_range(group.index.min(), group.index.max())
    cum = group['number'].reindex(full_range)
    
    # Interpolate cumulative cases linearly
    cum_interp = cum.interpolate(method='linear')
    
    # Get daily new cases (difference, first value is NaN or 0)
    daily_cases = cum_interp.diff().fillna(0)

    # Smooth daily cases a bit to avoid noise in peak finding
    smooth = daily_cases.rolling(window=7, min_periods=1, center=True).mean()
    
    # Identify peaks (you can tweak 'distance' or 'prominence' as needed)
    peaks, _ = find_peaks(smooth, 
                          distance=180, 
                          prominence=0.2 * smooth.max())  # at least 20% of max value (scales for large/small countries))  # adjust distance/prominence to taste

    # For each peak, find the wave start (previous local min) and end (next local min)
    window = 180  # days before/after to search for minima

    for i, peak_idx in enumerate(peaks):
        peak_date = full_range[peak_idx]
        
        # Define search bounds
        start_search = max(0, peak_idx - window)
        end_search = min(len(smooth) - 1, peak_idx + window)
        
        # Search for previous local minimum (start of wave)
        pre_window = smooth[start_search:peak_idx]
        if len(pre_window) > 0:
            wave_start_idx = pre_window.idxmin()
            wave_start_idx = smooth.index.get_loc(wave_start_idx)
        else:
            wave_start_idx = start_search

        # Search for next local minimum (end of wave)
        post_window = smooth[peak_idx:end_search]
        if len(post_window) > 0:
            wave_end_idx = post_window.idxmin()
            wave_end_idx = smooth.index.get_loc(wave_end_idx)
        else:
            wave_end_idx = end_search

        season_number = i + 1
        # Extract the date interval
        season_dates = full_range[wave_start_idx:wave_end_idx+1]
        season_cum_cases = cum_interp.loc[season_dates]
        for d, c in zip(season_dates, season_cum_cases):
            season_case_records.append({
                'Region': keys[0],
                'Country': keys[1],
                'Domain': keys[2],
                'Sub-Domain': keys[3],
                'season_number': season_number,
                'date': d,
                'cumulative_cases': c
            })


# Compile results into a DataFrame
season_case_df = pd.DataFrame(season_case_records)

# for name, group in season_case_df.groupby(['Region', 'Country', 'Domain', 'Sub-Domain']):
#     plt.figure(figsize=(10,5))
#     for season, season_group in group.groupby('season_number'):
#         plt.plot(season_group['date'], season_group['cumulative_cases'], label=f'Season {season}')
#     title_str = ', '.join([str(n) for n in name if n != ''])
#     plt.title(f"Cumulative Cases by Season: {title_str}")
#     plt.xlabel('Date')
#     plt.ylabel('Cumulative Cases')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"data_files/visualization/influenza_season_plots_new/{name[1]}_{name[2]}_{name[3]}.png")

season_case_df = season_case_df.sort_values(['Region', 'Country', 'Domain', 'Sub-Domain', 'season_number', 'date'])

def zero_relative(group):
    first_cum = group['cumulative_cases'].iloc[0]
    group = group.copy()
    group['season_cumcase'] = group['cumulative_cases'] - first_cum
    return group

# Apply to each location+season group
season_case_df = season_case_df.groupby(
    ['Region', 'Country', 'Domain', 'Sub-Domain', 'season_number'],
    group_keys=False
).apply(zero_relative)

season_case_df['number'] = season_case_df['season_cumcase']
season_case_df['type'] = 'Cumulative_Cases'
season_case_df['Microbe Family'] = 'Orthomyxovididae'
season_case_df['Microbe Genus'] = 'Influenza A, B, and C Virus'
season_case_df['Microbe Species'] = None

season_case_df = season_case_df[['Microbe Family','Microbe Genus','Microbe Species',
                                 'Region', 'Country', 'Domain', 'Sub-Domain', 
                                 'season_number', 'date', 'type','number']]

season_case_df.to_csv('/n/data1/hms/dbmi/farhat/alex/Pandemic-Early-Warning/data_files/influenza_seasons.csv', index=False)

# for name, group in season_case_df.groupby(['Region', 'Country', 'Domain', 'Sub-Domain']):
#     plt.figure(figsize=(10,5))
#     for season, season_group in group.groupby('season_number'):
#         # X axis: days since start of season
#         days_since_start = (season_group['date'] - season_group['date'].min()).dt.days
#         plt.plot(days_since_start, season_group['season_cumcase'], label=f'Season {season}')
#     title_str = ', '.join([str(n) for n in name if n != ''])
#     plt.title(f"Cumulative Cases (from 0) by Season: {title_str}")
#     plt.xlabel('Days Since Start of Season')
#     plt.ylabel('Cumulative Cases (Season)')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"data_files/visualization/influenza_season_plots_combined_new/{name[1]}_{name[2]}_{name[3]}.png")
