import numpy as np
import pyhrv
import pyhrv.frequency_domain as fd
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import hilbert, savgol_filter, find_peaks, welch, detrend
import pandas as pd
from scipy.integrate import simpson
from scipy.interpolate import interp1d


with open("output_column.csv", 'r') as file:
    reader = csv.reader(file)
    
    # Read all rows once and extract timestamps and IR data
    nnir = []
    for row in reader:
        nnir.append(float(row[0]))        # Column 1 = PPG signal


# Convert list to numpy array
nni = np.array(nnir)
# Compute the PSD and frequency domain parameters
result = fd.welch_psd(nni=nni)

# Access peak frequencies using the key 'fft_peak'
print(result['fft_peak'])

print("hi")