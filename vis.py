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
file_path = "data\\PPGPVTtests\\day1test2530.csv"

# === Load IR data ===

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
            print(f"\n--- Detected jump at index {i} ---")
            print(f"  Previous timestamp: {corrected_timestamps[i-1]} us")
            print(f"  Observed difference: {time_diff} us")

            # Calculate the actual "missing" time or excessive delay
            # This is the amount we need to subtract from subsequent timestamps.
            # We assume the two points involved in the jump *should* have been
            # exactly one expected_period_us apart.
            # So, the "excess" time is (observed_diff - expected_period).
            # This excess is what causes everything after it to be shifted.
            excess_time_us = time_diff - time_interval_us

            print(f"  Expected period: {time_interval_us} us")
            print(f"  Excess time to correct: {excess_time_us} us")

            # Subtract this excess time from all *subsequent* timestamps
            corrected_timestamps[i:] -= excess_time_us
            print(f"  Timestamp at index {i} corrected to: {corrected_timestamps[i]} us")
            print(f"  New difference from previous: {corrected_timestamps[i] - corrected_timestamps[i-1]} us")
            print(f"  Remaining timestamps adjusted by {excess_time_us} us.")
            print("---------------------------------")


timestamps_seconds = np.array(corrected_timestamps) / 1e6  # µs → seconds
first_time = timestamps_seconds[0]
timestamps = timestamps_seconds - first_time  # Now starts at 0
first_time = timestamps[0]
print("useless")
print("start time:", first_time)
end_time = timestamps[-1]
num_samples = int(np.ceil((end_time - first_time) * actual_sampling_rate))
uniform_timestamps = np.linspace(first_time, end_time, num_samples)
interpolator = interp1d(timestamps, ir_data, kind='cubic', fill_value='extrapolate')
resampled_signal = interpolator(uniform_timestamps)

print("new time")
print(len(uniform_timestamps))
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
print(time_ranges_filename)
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


# === FFT ===
y = hpfiltered  # your filtered signal
fs = actual_sampling_rate  # your sampling rate, e.g., 50 Hz

N = len(y)
yf = np.fft.fft(y)
xf = np.fft.fftfreq(N, 1/fs)

idx = xf >= 0
xf = xf[idx]
yf = np.abs(yf[idx])  # magnitude


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

print(ranges)

# Remove segments from the filtered peak times based on the ranges
filtered_peaks = np.array(filtered_peaks)
# Recalculate using filtered peaks
filtered_peak_times = np.array(filtered_peaks) * time_interval + start_time
print(filtered_peak_times)

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
    print("goon")
    # Ensure the range is within bounds
    start_idx = max(0, int(((start_idx - start_time) / time_interval)))
    end_idx = min(len(hpfiltered) - 1, int(((end_idx - start_time) / time_interval)))

    print(start_idx, end_idx)

    if start_idx < end_idx:
        # Find the index of max value in the range
        local_max_idx = start_idx + np.argmax(hpfiltered[start_idx:end_idx+1])

        # Add to peaks if not already present
        if local_max_idx not in ranges_removed_peaks:
            custom_peaks.append(local_max_idx)


print(custom_peaks)
print(final_peaks)
final_peaks = np.sort(np.concatenate([ranges_removed_peaks, np.array(custom_peaks, dtype=int)]))

final_peaks = np.array(final_peaks, dtype=int)  # or dtype=int if they're indices

print(final_peaks)
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
print(re_filtered_intervals)
re_filtered_intervals = np.array(re_filtered_intervals)


# === LF/HF Ratio ===
sampling_rate_hz=4
segment_length_s=570
nfft_points=1024

r_peak_times_s = np.cumsum(re_filtered_intervals)
# Adjust to start time from 0 if it's not already
r_peak_times_s = r_peak_times_s - r_peak_times_s[0]

# Check if the recording is long enough
if r_peak_times_s[-1] < segment_length_s:
    print(f"Warning: Recording is only {r_peak_times_s[-1]:.2f} seconds long, "
            f"which is shorter than the desired segment length of {segment_length_s} seconds.")
    segment_length_s = r_peak_times_s[-1] # Adjust segment length to actual duration

# --- 2. Interpolation/Resampling ---

time_interp = np.arange(0, segment_length_s, 1 / sampling_rate_hz)


unique_r_peak_times, unique_rr_intervals = np.unique(r_peak_times_s, return_index=True)
f_interp = interp1d(unique_r_peak_times, re_filtered_intervals[unique_rr_intervals], kind='linear', fill_value="extrapolate")

# Get the interpolated RR interval series (in seconds)
interpolated_rr_s = f_interp(time_interp)


# --- 3. Detrending (Optional but Recommended) ---
# Remove linear trend from the interpolated signal to improve PSD estimation
detrended_signal = detrend(interpolated_rr_s)


# --- 4. LF/HF (PSD) Calculation ---

nperseg_actual = min(nfft_points, len(detrended_signal))
nperseg_actual = int(2**np.floor(np.log2(nperseg_actual)))

noverlap_actual = 0 # No overlap for single segment approach

fafrequencies, psd = welch(detrended_signal,
                            fs=sampling_rate_hz,
                            nperseg=nperseg_actual,
                            noverlap=noverlap_actual,
                            nfft=nfft_points, 
                            window='hann', 
                            scaling='spectrum') 

psd_ms2_per_hz = psd * (1000**2)

# --- 5. Define Frequency Bands and Calculate Power ---
# Frequencies in Hz
VLF_band = (0.003, 0.04) # Note: For short-term, VLF is often not interpreted or included in normalization base
LF_band = (0.04, 0.15)
HF_band = (0.15, 0.4)

# Find indices corresponding to each band
vlf_indices = np.where((fafrequencies >= VLF_band[0]) & (fafrequencies < VLF_band[1]))[0]
lf_indices = np.where((fafrequencies >= LF_band[0]) & (fafrequencies < LF_band[1]))[0]
hf_indices = np.where((fafrequencies >= HF_band[0]) & (fafrequencies < HF_band[1]))[0]


df = fafrequencies[1] - fafrequencies[0] 

total_power = np.sum(psd_ms2_per_hz) * df 

vlf_power = np.sum(psd_ms2_per_hz[vlf_indices]) * df
lf_power = np.sum(psd_ms2_per_hz[lf_indices]) * df
hf_power = np.sum(psd_ms2_per_hz[hf_indices]) * df

# Calculate normalized units (excluding VLF from the denominator, as per document)
lf_norm = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
hf_norm = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0

# Calculate LF/HF Ratio
lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf # Handle division by zero


print("\n--- HRV Frequency Domain Analysis Results ---")
print(f"Total Power: {total_power:.2f} ms^2")
print(f"VLF Power: {vlf_power:.2f} ms^2 (Band: {VLF_band[0]}-{VLF_band[1]} Hz)")
print(f"LF Power: {lf_power:.2f} ms^2 (Band: {LF_band[0]}-{LF_band[1]} Hz)")
print(f"HF Power: {hf_power:.2f} ms^2 (Band: {HF_band[0]}-{HF_band[1]} Hz)")
print(f"Normalized LF (LFnu): {lf_norm:.2f}")
print(f"Normalized HF (HFnu): {hf_norm:.2f}")
print(f"LF/HF Ratio: {lf_hf_ratio:.2f}")


# === SNR ===
fs = actual_sampling_rate  # Sampling frequency
frequencies, psd = welch(y_fixed, 
                        fs=fs, 
                        nperseg=8192,  # Window size
                        noverlap=4096,  # Overlap between segments
                        nfft=65536,
                        scaling='density')



# Calculate SNR (Heart band vs Noise band)
plusminusrange = 0.2
print("BPS")
print(bps)
heart_rate_range = (bps-plusminusrange, bps+plusminusrange)   # Very low frequency noise
first_harmonic = (bps*2-plusminusrange, bps*2+plusminusrange)
second_harmonic = (bps*3-plusminusrange, bps*3+plusminusrange)


zero_mask = (frequencies >= heart_rate_range[0]) & (frequencies <= heart_rate_range[1])
first_mask = (frequencies >= first_harmonic[0]) & (frequencies <= first_harmonic[1])
second_mask = (frequencies >= second_harmonic[0]) & (frequencies <= second_harmonic[1])
zero_h_area = simpson(psd[zero_mask], frequencies[zero_mask])
first_h_area = simpson(psd[first_mask], frequencies[first_mask])
second_h_area = simpson(psd[second_mask], frequencies[second_mask])
heart_area = zero_h_area + first_h_area + second_h_area

mask_all = ((frequencies >= 0.5) & (frequencies <= 15))


dominant_freq = frequencies[mask_all][np.argmax(psd[mask_all])]
heart_rate_bpm = dominant_freq * 60  # Convert Hz to BPM



all_area = simpson(psd[mask_all], frequencies[mask_all])
noise_area = simpson(psd[mask_all], frequencies[mask_all]) - heart_area

snr = heart_area / noise_area
print(snr)

print("httr")
httr = heart_area / all_area
print(httr)


# === calculating APA ===
peak_values = hpfiltered[filtered_peaks]

# 2. Calculate the average of these peak values
average_peak_value = np.mean(peak_values)

# 3. Print the result
print(average_peak_value)

# Convert list to numpy array


# Access peak frequencies using the key 'fft_peak'

# === Plot 1: Smoothed and Raw in Window 1 ===

plt.figure(1, figsize=(12, 6))
plt.plot(x_filtered, ir_filtered, label='IR Smoothed')
plt.plot(x_filtered, ir_filtered, 'o', label='IR Raw', markersize=3)
plt.plot(bad_x, bad_y, 'o', label='IR Raw', markersize=4, color = 'black')

plt.title(f"PPG Signal — Smoothed & Raw — {start_time}s to {end_time}s")
plt.xlabel("Time (seconds)")
plt.ylabel("Sensor Value")
plt.legend()
plt.xlim(start_time, max(x_filtered)+5)

plt.grid(True)
plt.tight_layout()


# === Plot 2: High Pass Detrend in Window 2 ===
plt.figure(2, figsize=(12, 6))
plt.plot(x_filtered, hpfiltered, label='High Pass Detrended', color='purple')
plt.plot(filtered_peak_times, hpfiltered[filtered_peaks], 'ro', label="Peaks", markersize = 4)
plt.plot(peak_times, hpfiltered[peaks], 'ro', label="unfiltered peaks", markersize = 2, color = 'green')

plt.plot(ranges_removed_peak_times, hpfiltered[ranges_removed_peaks], 'ro', label="removed ranges peaks", markersize = 4, color = 'black')
plt.plot(custom_peak_times, hpfiltered[custom_peaks], 'ro', label="added peaks", markersize = 4, color = 'red')


stats_text = f"Heart Rate: {heart_rate:.1f} BPM\nHRV (std RR): {hrv:.3f} s"
plt.text(0.83, 0.6, stats_text, transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, boxstyle="round"))

plt.title(f"PPG Signal (HP Detrended) — {start_time}s to {end_time}s")
plt.xlabel("Time (seconds)")
plt.ylabel("HP Detrended Value")
plt.legend()
plt.xlim(start_time, max(x_filtered)+5)
plt.grid(True)
plt.tight_layout()




# === Plot 3: FFT ===
plt.figure(3, figsize=(12, 6))
plt.plot(xf, yf, label='High Pass Detrended', color='purple')
plt.title("FFT of the Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 9)  # Limit x-axis to 10 Hz for better visibility
plt.grid(True)

# === Plot 4: PSD ===
plt.figure(4, figsize=(12, 6))
plt.plot(frequencies, psd, label='PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V²/Hz)')
plt.title(f'PPG Power Spectral Density\nDominant HR: {heart_rate_bpm:.1f} BPM | SNR: {snr:.2f}')

# Add heart rate band shading
plt.axvspan(heart_rate_range[0], heart_rate_range[1], color='green', alpha=0.1, label='Heart Rate Band')
plt.axvspan(first_harmonic[0], first_harmonic[1], color='green', alpha=0.1, label='First Harmonic Heart Rate Band')

plt.axvline(dominant_freq, color='red', linestyle='--', label=f'Dominant Frequency ({dominant_freq:.2f} Hz)')
plt.axvline(dominant_freq*2, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*2:.2f} Hz)')
plt.axvline(dominant_freq*3, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*3:.2f} Hz)')
plt.axvline(dominant_freq*4, color='purple', linestyle='--', label=f'Dominant Frequency ({dominant_freq*4:.2f} Hz)')
plt.xlim(0, 15)
plt.ylim(0, np.max(psd) * 1.1)  # Set y-axis limit to 110% of max PSD value


# Formatting
print("hiiii")

# === Show both windows ===
nni = np.array(re_filtered_intervals)
# Compute the PSD and frequency domain parameters
result = fd.welch_psd(nni=nni, show=False)

print(result['fft_ratio'])
freqbands = result['fft_abs']
print(freqbands)
print("hi")


plt.show() 

