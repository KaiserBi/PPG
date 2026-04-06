import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import hilbert, savgol_filter, find_peaks, welch, detrend
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import pyhrv
import pyhrv.frequency_domain as fd

# === File paths ===
#file_path = os.path.join("PPG", "day1test10400am.csv")

# === Load IR data ===



def process_ppg_file(file_path):

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        
        # Read all rows once and extract timestamps and IR data
        rawtimestamps = []
        ir_data = []
        for row in reader:
            if len(row) >= 2 and row[1].strip() != '':  # Ensure row has at least 2 columns and column 1 is non-empty
                rawtimestamps.append(float(row[0]))  # Column 0 = timestamp
                ir_data.append(float(row[1]))        # Column 1 = PPG signal


    time_interval = 1/750  # 1/200
    time_interval_us = (int)(time_interval * 1e6)  # Convert to microseconds
    x_values = [i * time_interval for i in range(len(ir_data))]
    actual_sampling_rate = 1/time_interval

    corrected_timestamps = np.array(rawtimestamps, dtype=np.int64)  # Convert to numpy array for easier manipulation
    print(len(corrected_timestamps))
    for i in range(1, len(corrected_timestamps)):
            time_diff = corrected_timestamps[i] - corrected_timestamps[i-1]

            if abs(time_diff) > 10000:
                
                excess_time_us = time_diff - time_interval_us

              
                corrected_timestamps[i:] -= excess_time_us
                


    timestamps_seconds = np.array(corrected_timestamps) / 1e6  # µs → seconds
    first_time = timestamps_seconds[0]
    timestamps = timestamps_seconds - first_time  # Now starts at 0
    first_time = timestamps[0]

    end_time = timestamps[-1]
    num_samples = int(np.ceil((end_time - first_time) * actual_sampling_rate))
    uniform_timestamps = np.linspace(first_time, end_time, num_samples)
    interpolator = interp1d(timestamps, ir_data, kind='cubic', fill_value='extrapolate')
    resampled_signal = interpolator(uniform_timestamps)

   
    # === Time axis (assuming 200 Hz sampling) ===


    # === CONTROL: Time range to plot (in seconds) ===
    start_time = 0   # set this to your desired start
    end_time = 900  # set this to your desired end

    custom_time_ranges = {}

    with open("Metadata\\PPG_time_ranges.csv", mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                time_ranges_filename = row['filename']
                start = float(row['start'])  # or int() if all times are integers
                end = float(row['end'])
                custom_time_ranges[time_ranges_filename] = (start, end)
        
    time_ranges_filename = os.path.splitext(os.path.basename(file_path))[0]
    start_time, end_time = custom_time_ranges.get(time_ranges_filename, custom_time_ranges['default'])

    # === Filter data based on time range ===
    filtered_indices = [i for i, t in enumerate(uniform_timestamps) if start_time <= t <= end_time]

    x_filtered = [uniform_timestamps[i] for i in filtered_indices]

    ir_filtered = [resampled_signal[i] for i in filtered_indices]

    # === Raw Points ===


    # === Remove Outliers ===
    y = np.array(ir_filtered)
    y_median = pd.Series(y).rolling(window=15, center=True, min_periods=1).median()
    gap = abs(y_median - y)
    threshold = 4.25 * np.std(gap.dropna())
    outliers = gap > threshold
    x_filtered = np.array(x_filtered)
    bad_x = x_filtered[outliers]
    bad_y = y[outliers]


    y_fixed = np.copy(y)

    y_fixed[outliers.to_numpy()] = np.interp(
        x_filtered[outliers.to_numpy()],  # x positions of bad points
        x_filtered[~outliers.to_numpy()], # x positions of good points
        y[~outliers.to_numpy()]           # y values of good points
    )

    # === Smoothing (moving average) ===
    window_size = 0
    smooth = lambda data: [np.mean(data[max(0, i-window_size): min(len(data), i+window_size + 1)]) for i in range(len(data))]

    ir_smooth = smooth(y_fixed)
    x_smoothed = uniform_timestamps[:len(ir_smooth)]

    # === High Pass Detrend ===
    from scipy.signal import butter, filtfilt
    def bandpass(data, lowcut=0.5, highcut=10, fs=50, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    hpfiltered = bandpass(ir_smooth, lowcut=0.6, highcut=3.3, fs=actual_sampling_rate)

    # === Peak Finding ===
    peaks, props = find_peaks(hpfiltered, prominence=1.1, width = 0.13, distance = 0.4*actual_sampling_rate)


    # === Biometric Calculation ===
    peak_times = np.array(peaks * time_interval) + start_time
    rr_intervals = np.diff(peak_times)
    median_rr = np.median(rr_intervals)

    filtered_peaks = [peaks[0]]
    for i in range(1, len(peak_times)):
        rr = peak_times[i] - peak_times[i - 1]
        if 0.5 * median_rr < rr:
            filtered_peaks.append(peaks[i])

    # === Remove Ranges ===
    df = pd.read_csv("Metadata\\RemoveRanges.csv")
    # Filter only the rows relevant to this PPG file
    ranges = df[df['filename'] == time_ranges_filename][['start_idx', 'end_idx']].values


    # Remove segments from the filtered peak times based on the ranges
    filtered_peaks = np.array(filtered_peaks)
    # Recalculate using filtered peaks
    filtered_peak_times = np.array(filtered_peaks) * time_interval + start_time

    keep_mask = np.ones_like(filtered_peak_times, dtype=bool)

    # Remove peaks inside any of the specified ranges
    for start, end in ranges:
        keep_mask &= ~((filtered_peak_times >= start) & (filtered_peak_times <= end))

    # Apply mask to keep only peaks outside all ranges
    ranges_removed_peak_times = filtered_peak_times[keep_mask]
    ranges_removed_peaks = ((ranges_removed_peak_times - start_time) / time_interval).astype(int)

    final_peaks = []
    custom_peaks = []
    # Add custom peaks
    df = pd.read_csv("Metadata\\Manual_peaks.csv")
    file_ranges = df[df['filename'].str.strip() == time_ranges_filename]
    for _, row in file_ranges.iterrows():
        start_idx = float(row['startidx'])
        end_idx = float(row['endidx'])
        # Ensure the range is within bounds
        start_idx = max(0, int(((start_idx - start_time) / time_interval)))
        end_idx = min(len(hpfiltered) - 1, int(((end_idx - start_time) / time_interval)))


        if start_idx < end_idx:
            # Find the index of max value in the range
            local_max_idx = start_idx + np.argmax(hpfiltered[start_idx:end_idx+1])

            # Add to peaks if not already present
            if local_max_idx not in ranges_removed_peaks:
                custom_peaks.append(local_max_idx)


    final_peaks = np.sort(np.concatenate([ranges_removed_peaks, np.array(custom_peaks, dtype=int)]))

    final_peaks = np.array(final_peaks, dtype=int)  # or dtype=int if they're indices

    final_peak_times = final_peaks * time_interval + start_time

    custom_peak_times = np.array(custom_peaks) * time_interval + start_time


    filtered_rr_intervals = np.diff(final_peak_times)
    median_rr = np.median(filtered_rr_intervals)

    re_filtered_intervals = [filtered_rr_intervals[0]]  # Start with the first peak
    for i in range(1, len(filtered_rr_intervals)):
        if 2.5 * median_rr > filtered_rr_intervals[i]:
            re_filtered_intervals.append(filtered_rr_intervals[i])


    heart_rate = 60 / np.mean(re_filtered_intervals)
    bps = heart_rate / 60
    hrv = np.std(re_filtered_intervals)
    re_filtered_intervals = np.array(re_filtered_intervals)


    
    # === Show both windows ===
    nni = np.array(re_filtered_intervals)
    # Compute the PSD and frequency domain parameters
    result = fd.welch_psd(nni=nni, show=False)
    # Access peak frequencies using the key 'fft_peak'
    lfhf = result['fft_ratio']
    freqbands = result['fft_abs']
    vlf = freqbands[0]
    lf = freqbands[1]
    hf = freqbands[2]

    return os.path.basename(file_path), bps, hrv, lfhf, vlf, lf, hf

def main():
    print("hi")
    data_folder = os.path.join("./data", "PPGPVTtests", "day3")

    filenames = []
    bps, hrv, lfhf, vlf, lf, hf = [], [], [], [], [], []
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            file_name, bps_val, hrv_val, lfhf_val, vlf_val, lf_val, hf_val = process_ppg_file(file_path)
            filenames.append(file_name)
            
            bps.append(bps_val)
            hrv.append(hrv_val)
            lfhf.append(lfhf_val)
            vlf.append(vlf_val)
            lf.append(lf_val)
            hf.append(hf_val)

    df = pd.DataFrame({
        "Filename": filenames,
        "BPS": bps,
        "HRV": hrv,
        "LF/HF": lfhf,
        "VLF": vlf,
        "LF": lf,
        "HF": hf
    })

    # Save to CSV
    df.to_csv("ppg_results.csv", index=False)
    print("CSV file saved as ppg_results.csv")

    # Now you can use the aggregated lists (bps, hrv, etc.) as needed



if __name__ == "__main__":
    main()