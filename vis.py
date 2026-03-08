import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import pyhrv.frequency_domain as fd

# === File paths ===
file_path = "totaldata.csv"
time_ranges_filename = os.path.splitext(os.path.basename(file_path))[0]

# === 1. Load IR data ===
rawtimestamps = []
ir_data = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # Ensure row has at least 2 columns and column 1 is non-empty
        if len(row) >= 2 and row[1].strip() != '':
            try:
                # Attempt to convert the strings to floats
                timestamp = float(row[0])
                ir_value = float(row[1])
                
                # If successful, append to our arrays
                rawtimestamps.append(timestamp)
                ir_data.append(ir_value)
            except ValueError:
                # If a ValueError occurs (e.g., text like '1/Mean'), skip this row
                continue

# Define sampling variables necessary for downstream calculations
time_interval = 1 / 750
time_interval_us = int(time_interval * 1e6)
actual_sampling_rate = 1 / time_interval

# === 2. Timestamp Correction ===
corrected_timestamps = np.array(rawtimestamps, dtype=np.int64)

for i in range(1, len(corrected_timestamps)):
    time_diff = corrected_timestamps[i] - corrected_timestamps[i-1]
    if abs(time_diff) > 10000:
        excess_time_us = time_diff - time_interval_us
        corrected_timestamps[i:] -= excess_time_us

timestamps_seconds = corrected_timestamps / 1e6
first_time = timestamps_seconds[0]
timestamps = timestamps_seconds - first_time
end_time = timestamps[-1]

# === 3. Resampling ===

# Extract unique timestamps and their original indices
timestamps_unique, unique_indices = np.unique(timestamps, return_index=True)

# Map the unique indices to the IR data to remove the corresponding duplicate readings
ir_data_unique = np.array(ir_data)[unique_indices]

# Recalculate signal boundaries using the sanitized arrays
end_time = timestamps_unique[-1]
start_time = timestamps_unique[0]

# Generate the uniform time axis based on the 750 Hz actual_sampling_rate
num_samples = int(np.ceil((end_time - start_time) * actual_sampling_rate))
uniform_timestamps = np.linspace(start_time, end_time, num_samples)

# Execute interpolation with strictly monotonically increasing x-values
interpolator = interp1d(timestamps_unique, ir_data_unique, kind='cubic', fill_value='extrapolate')
resampled_signal = interpolator(uniform_timestamps)

# === 4. Time Range Filtering ===
custom_time_ranges = {'default': (0, 900)}
if os.path.exists("Metadata/PPG_time_ranges.csv"):
    with open("Metadata/PPG_time_ranges.csv", mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            custom_time_ranges[row['filename']] = (float(row['start']), float(row['end']))

start_time, end_time = custom_time_ranges.get(time_ranges_filename, custom_time_ranges['default'])
filtered_indices = [i for i, t in enumerate(uniform_timestamps) if start_time <= t <= end_time]

x_filtered = np.array([uniform_timestamps[i] for i in filtered_indices])
ir_filtered = np.array([resampled_signal[i] for i in filtered_indices])

# === 5. Outlier Removal ===
y = np.array(ir_filtered)
y_median = pd.Series(y).rolling(window=15, center=True, min_periods=1).median()
gap = np.abs(y_median - y)
threshold = 4.25 * np.std(gap.dropna())
outliers = (gap > threshold).to_numpy()

bad_x = x_filtered[outliers]
bad_y = y[outliers]

y_fixed = np.copy(y)
y_fixed[outliers] = np.interp(x_filtered[outliers], x_filtered[~outliers], y[~outliers])

# === 6. High Pass Detrend ===
def bandpass(data, lowcut=0.5, highcut=10, fs=50, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

hpfiltered = bandpass(y_fixed, lowcut=0.6, highcut=3.3, fs=actual_sampling_rate)

# === 7. Peak Finding ===
peaks, props = find_peaks(hpfiltered, prominence=1.1, width=0.13, distance=0.4*actual_sampling_rate)
peak_times = (peaks * time_interval) + start_time
rr_intervals = np.diff(peak_times)
median_rr = np.median(rr_intervals)

filtered_peaks = [peaks[0]]
for i in range(1, len(peak_times)):
    rr = peak_times[i] - peak_times[i - 1]
    if 0.5 * median_rr < rr:
        filtered_peaks.append(peaks[i])

filtered_peaks = np.array(filtered_peaks)
filtered_peak_times = filtered_peaks * time_interval + start_time

# === 8. Range Removal and Custom Peaks ===
keep_mask = np.ones_like(filtered_peak_times, dtype=bool)

if os.path.exists("Metadata/RemoveRanges.csv"):
    df_remove = pd.read_csv("Metadata/RemoveRanges.csv")
    ranges = df_remove[df_remove['filename'] == time_ranges_filename][['start_idx', 'end_idx']].values
    for start, end in ranges:
        keep_mask &= ~((filtered_peak_times >= start) & (filtered_peak_times <= end))

ranges_removed_peak_times = filtered_peak_times[keep_mask]
ranges_removed_peaks = ((ranges_removed_peak_times - start_time) / time_interval).astype(int)

custom_peaks = []
if os.path.exists("Metadata/Manual_peaks.csv"):
    df_manual = pd.read_csv("Metadata/Manual_peaks.csv")
    file_ranges = df_manual[df_manual['filename'].str.strip() == time_ranges_filename]
    for _, row in file_ranges.iterrows():
        s_idx = max(0, int(((float(row['startidx']) - start_time) / time_interval)))
        e_idx = min(len(hpfiltered) - 1, int(((float(row['endidx']) - start_time) / time_interval)))
        
        if s_idx < e_idx:
            local_max_idx = s_idx + np.argmax(hpfiltered[s_idx:e_idx+1])
            if local_max_idx not in ranges_removed_peaks:
                custom_peaks.append(local_max_idx)

final_peaks = np.sort(np.concatenate([ranges_removed_peaks, np.array(custom_peaks, dtype=int)]))
final_peak_times = final_peaks * time_interval + start_time
custom_peak_times = np.array(custom_peaks) * time_interval + start_time

# === 9. Biometric Calculation ===
filtered_rr_intervals = np.diff(final_peak_times)
median_rr = np.median(filtered_rr_intervals)

re_filtered_intervals = [filtered_rr_intervals[0]]
for i in range(1, len(filtered_rr_intervals)):
    if 2.5 * median_rr > filtered_rr_intervals[i]:
        re_filtered_intervals.append(filtered_rr_intervals[i])

re_filtered_intervals = np.array(re_filtered_intervals)
heart_rate = 60 / np.mean(re_filtered_intervals)
bps = heart_rate / 60
hrv = np.std(re_filtered_intervals)

# === 10. FFT & SNR Calculation ===
nperseg_snr = min(8192, len(y_fixed))
noverlap_snr = nperseg_snr // 2

frequencies, psd = welch(y_fixed, fs=actual_sampling_rate, nperseg=nperseg_snr, noverlap=noverlap_snr, nfft=65536, scaling='density')

plusminusrange = 0.2
heart_rate_range = (bps - plusminusrange, bps + plusminusrange)
first_harmonic = (bps * 2 - plusminusrange, bps * 2 + plusminusrange)
second_harmonic = (bps * 3 - plusminusrange, bps * 3 + plusminusrange)

zero_mask = (frequencies >= heart_rate_range[0]) & (frequencies <= heart_rate_range[1])
first_mask = (frequencies >= first_harmonic[0]) & (frequencies <= first_harmonic[1])
second_mask = (frequencies >= second_harmonic[0]) & (frequencies <= second_harmonic[1])

# Explicitly assign x values to Simpson's rule for accuracy
zero_h_area = simpson(psd[zero_mask], x=frequencies[zero_mask])
first_h_area = simpson(psd[first_mask], x=frequencies[first_mask])
second_h_area = simpson(psd[second_mask], x=frequencies[second_mask])
heart_area = zero_h_area + first_h_area + second_h_area

mask_all = ((frequencies >= 0.5) & (frequencies <= 15))
dominant_freq = frequencies[mask_all][np.argmax(psd[mask_all])]
heart_rate_bpm = dominant_freq * 60

all_area = simpson(psd[mask_all], x=frequencies[mask_all])
noise_area = all_area - heart_area
snr = heart_area / noise_area if noise_area > 0 else np.inf

# === 11. pyHRV execution ===
nni_ms = re_filtered_intervals * 1000
result = fd.welch_psd(nni=nni_ms, show=False)

# === 12. Plotting ===
plt.figure(1, figsize=(12, 6))
plt.plot(x_filtered, ir_filtered, label='IR Smoothed')
plt.plot(x_filtered, ir_filtered, marker='o', linestyle='None', label='IR Raw', markersize=3)
plt.plot(bad_x, bad_y, marker='o', linestyle='None', label='Outliers', markersize=4, color='black')
plt.title(f"PPG Signal - Smoothed and Raw - {start_time}s to {end_time}s")
plt.xlabel("Time (seconds)")
plt.ylabel("Sensor Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(2, figsize=(12, 6))
plt.plot(x_filtered, hpfiltered, label='High Pass Detrended', color='purple')
plt.plot(filtered_peak_times, hpfiltered[filtered_peaks], marker='o', linestyle='None', color='red', label="Peaks", markersize=4)
plt.plot(peak_times, hpfiltered[peaks], marker='o', linestyle='None', color='green', label="Unfiltered peaks", markersize=2)
plt.plot(ranges_removed_peak_times, hpfiltered[ranges_removed_peaks], marker='o', linestyle='None', color='black', label="Removed ranges peaks", markersize=4)
plt.plot(custom_peak_times, hpfiltered[custom_peaks], marker='o', linestyle='None', color='orange', label="Added peaks", markersize=4)

stats_text = f"Heart Rate: {heart_rate:.1f} BPM\nHRV (std RR): {hrv:.3f} s"
plt.text(0.83, 0.6, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle="round"))
plt.title(f"PPG Signal (HP Detrended) - {start_time}s to {end_time}s")
plt.xlabel("Time (seconds)")
plt.ylabel("HP Detrended Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(3, figsize=(12, 6))
plt.plot(frequencies, psd, label='PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.title(f'PPG Power Spectral Density\nDominant HR: {heart_rate_bpm:.1f} BPM | SNR: {snr:.2f}')
plt.axvspan(heart_rate_range[0], heart_rate_range[1], color='green', alpha=0.1, label='HR Band')
plt.axvspan(first_harmonic[0], first_harmonic[1], color='green', alpha=0.1, label='1st Harmonic')
plt.axvline(dominant_freq, color='red', linestyle='--', label=f'Dominant Freq ({dominant_freq:.2f} Hz)')
plt.xlim(0, 15)
plt.ylim(0, np.max(psd) * 1.1)
plt.grid(True)
plt.tight_layout()

plt.show()