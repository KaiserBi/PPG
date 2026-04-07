"""
zcr_sqi.py  --  Zero Crossing Rate SQI for PPG signals
=======================================================
SEAL Lab | Embedded Body Sensors Team
Metric:  ZSQI = (number of sign changes in mean-subtracted signal) / (N - 1)
         A stable ZSQI indicates consistent sensor contact.
         A spike in ZSQI indicates jostling / lost contact.

Modes:
    Single file:
        python zcr_sqi.py --csv data.csv --col ppg_raw --fs 100

    Folder batch (no headers, col0=time_ms, col1=ppg -- fs auto-detected):
        python zcr_sqi.py --folder path/to/Results

Arguments:
    --csv       Path to a single input CSV file (use with --col)
    --col       Column name (string) or 0-based index for single-file mode
    --folder    Path to folder -- processes every *.csv inside it
                Assumes no header row: col0 = time (s or ms, auto-detected), col1 = PPG signal
    --fs        Sampling frequency in Hz (default: 100, ignored in folder mode -- auto-detected)
    --window    Analysis window length in seconds (default: 5.0)
    --step      Window step/stride in seconds (default: 1.0)
    --plot      Flag: show a plot per file
    --out       Output CSV path for single-file results (ignored in folder mode)
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd


# --- Core computation --------------------------------------------------------

def compute_zcr(segment: np.ndarray) -> float:
    if len(segment) < 2:
        return np.nan
    x = segment - np.mean(segment)
    s = np.sign(x)
    for i in range(len(s)):
        if s[i] == 0:
            s[i] = s[i - 1] if i > 0 else 1
    crossings = np.sum(np.diff(s) != 0)
    return float(crossings / (len(segment) - 1))


def windowed_zcr(signal: np.ndarray, fs: float, window_sec: float, step_sec: float):
    win_samples  = int(window_sec * fs)
    step_samples = int(step_sec   * fs)
    if win_samples < 2:
        raise ValueError(f"Window ({window_sec} s @ {fs} Hz) is too short.")
    if step_samples < 1:
        raise ValueError(f"Step ({step_sec} s) must be >= 1 sample.")
    times, zcrs = [], []
    start = 0
    while start + win_samples <= len(signal):
        seg = signal[start : start + win_samples]
        times.append((start + win_samples / 2) / fs)
        zcrs.append(compute_zcr(seg))
        start += step_samples
    return np.array(times), np.array(zcrs)


# --- I/O helpers -------------------------------------------------------------

def load_signal_named(csv_path: str, col):
    """Single-file mode: CSV has headers."""
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded: {csv_path}  |  shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    if isinstance(col, int):
        if col >= df.shape[1]:
            sys.exit(f"[ERROR] Column index {col} out of range.")
        series = df.iloc[:, col]
        print(f"[INFO] Using column index {col}: '{series.name}'")
    else:
        if col not in df.columns:
            sys.exit(f"[ERROR] Column '{col}' not found. Available: {list(df.columns)}")
        series = df[col]
        print(f"[INFO] Using column: '{col}'")
    return pd.to_numeric(series, errors='coerce').dropna().to_numpy(dtype=float)


def load_signal_no_header(csv_path: str):
    """
    Folder mode: headerless CSV, col0=time_ms, col1=ppg.
    Estimates fs from median inter-sample interval in the time column.
    """
    df = pd.read_csv(csv_path, header=None, names=["time_ms", "ppg"])
    time_ms = pd.to_numeric(df["time_ms"], errors='coerce').dropna().to_numpy(dtype=float)
    signal  = pd.to_numeric(df["ppg"],     errors='coerce').dropna().to_numpy(dtype=float)
    if len(time_ms) > 1:
        median_dt = float(np.median(np.diff(time_ms)))
        # auto-detect seconds vs milliseconds: if median step < 1 it's in seconds
        fs = float(1.0 / median_dt) if median_dt < 1 else float(1000.0 / median_dt)
    else:
        fs = 100.0
    return signal, fs


def maybe_int(value: str):
    try:
        return int(value)
    except ValueError:
        return value


# --- Print summary -----------------------------------------------------------

def print_summary(label: str, times: np.ndarray, zcrs: np.ndarray, fs: float):
    mean_zcr  = np.mean(zcrs)
    std_zcr   = np.std(zcrs)
    threshold = mean_zcr + 2 * std_zcr
    flagged_t = times[zcrs > threshold]
    flagged_z = zcrs[zcrs > threshold]

    print(f"\n{'=' * 60}")
    print(f"  File      : {os.path.basename(label)}")
    print(f"  fs        : {fs:.1f} Hz")
    print(f"{'-' * 60}")
    print(f"  Windows   : {len(zcrs)}")
    print(f"  ZSQI mean : {mean_zcr:.4f}")
    print(f"  ZSQI std  : {std_zcr:.4f}")
    print(f"  ZSQI min  : {np.min(zcrs):.4f}  @ t = {times[np.argmin(zcrs)]:.1f} s")
    print(f"  ZSQI max  : {np.max(zcrs):.4f}  @ t = {times[np.argmax(zcrs)]:.1f} s")
    print(f"  Threshold : mean + 2*std = {threshold:.4f}")
    print(f"{'-' * 60}")
    if len(flagged_t):
        print(f"  Flagged windows (possible jostling / contact loss):")
        for t, z in zip(flagged_t, flagged_z):
            print(f"    t = {t:6.1f} s  |  ZSQI = {z:.4f}")
    else:
        print(f"  No flagged windows -- contact appears stable.")
    print(f"{'=' * 60}")

    return float(threshold)


# --- Optional plot -----------------------------------------------------------

def plot_file(label: str, signal: np.ndarray, fs: float,
              times: np.ndarray, zcrs: np.ndarray, threshold: float):
    try:
        import matplotlib.pyplot as plt
        mean_zcr = np.mean(zcrs)
        t_axis   = np.arange(len(signal)) / fs
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6))
        fig.suptitle(os.path.basename(label), fontsize=11)
        ax1.plot(t_axis, signal, color='steelblue', linewidth=0.7)
        ax1.set_title("Raw PPG Signal")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        ax2.plot(times, zcrs, color='darkorange', linewidth=1.2, marker='o', markersize=3)
        ax2.axhline(mean_zcr,  color='gray', linestyle='--', label=f'Mean = {mean_zcr:.4f}')
        ax2.axhline(threshold, color='red',  linestyle=':',  label=f'Threshold = {threshold:.4f}')
        ax2.fill_between(times, zcrs, threshold,
                         where=(zcrs > threshold), alpha=0.3, color='red', label='Flagged')
        ax2.set_title("ZSQI per Window")
        ax2.set_xlabel("Window Centre Time (s)")
        ax2.set_ylabel("ZSQI")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("[WARNING] matplotlib not installed -- skipping plot.")


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ZCR SQI calculator for PPG signals.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--csv",    help="Path to a single CSV file (requires --col)")
    mode.add_argument("--folder", help="Path to folder of headerless CSVs (col0=time_ms, col1=ppg)")
    parser.add_argument("--col",    default=None, help="Column name or index (single-file mode only)")
    parser.add_argument("--fs",     type=float, default=100, help="Sampling frequency Hz (single-file mode, default: 100)")
    parser.add_argument("--window", type=float, default=5.0, help="Window length in seconds (default: 5.0)")
    parser.add_argument("--step",   type=float, default=1.0, help="Window stride in seconds (default: 1.0)")
    parser.add_argument("--plot",   action="store_true",     help="Show plot(s)")
    parser.add_argument("--out",    default=None,            help="Output CSV (single-file mode only)")
    args = parser.parse_args()

    # FOLDER MODE
    if args.folder:
        csv_files = sorted(glob.glob(os.path.join(args.folder, "*.csv")))
        if not csv_files:
            sys.exit(f"[ERROR] No .csv files found in: {args.folder}")
        print(f"[INFO] Found {len(csv_files)} CSV file(s) in: {args.folder}")
        for filepath in csv_files:
            try:
                signal, fs = load_signal_no_header(filepath)
                times, zcrs = windowed_zcr(signal, fs, args.window, args.step)
                threshold = print_summary(filepath, times, zcrs, fs)
                if args.plot:
                    plot_file(filepath, signal, fs, times, zcrs, threshold)
            except Exception as e:
                print(f"[ERROR] Failed to process {os.path.basename(filepath)}: {e}")
        return

    # SINGLE FILE MODE
    if args.col is None:
        sys.exit("[ERROR] --col is required in single-file mode.")
    col    = maybe_int(args.col)
    signal = load_signal_named(args.csv, col)
    times, zcrs = windowed_zcr(signal, args.fs, args.window, args.step)
    threshold = print_summary(args.csv, times, zcrs, args.fs)
    if args.out:
        pd.DataFrame({"time_s": times, "ZSQI": zcrs}).to_csv(args.out, index=False)
        print(f"[INFO] Results saved to: {args.out}")
    if args.plot:
        plot_file(args.csv, signal, args.fs, times, zcrs, threshold)


if __name__ == "__main__":
    main()
