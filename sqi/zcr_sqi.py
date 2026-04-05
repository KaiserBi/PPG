"""
zcr_sqi.py  --  Zero Crossing Rate SQI for PPG signals
=======================================================
SEAL Lab | Embedded Body Sensors Team
Metric:  ZSQI = (number of sign changes in mean-subtracted signal) / (N - 1)
         A stable ZSQI indicates consistent sensor contact.
         A spike in ZSQI indicates jostling / lost contact.

Usage:
    python zcr_sqi.py --csv data.csv --col ppg --fs 100
    python zcr_sqi.py --csv data.csv --col 0   --fs 100 --window 5.0 --step 1.0

Arguments:
    --csv       Path to input CSV file
    --col       Column name (string) OR 0-based column index (int) containing PPG data
    --fs        Sampling frequency in Hz (default: 100)
    --window    Analysis window length in seconds (default: 5.0)
    --step      Window step/stride in seconds (default: 1.0)  [set equal to --window for non-overlapping]
    --plot      Flag: show a plot of the signal and ZSQI over time
    --out       Optional path to save results as CSV
"""

import argparse
import sys
import numpy as np
import pandas as pd


# ─── Core computation ────────────────────────────────────────────────────────

def compute_zcr(segment: np.ndarray) -> float:
    """
    Compute Zero Crossing Rate for a single 1-D PPG segment.

    Steps:
      1. Remove DC offset (subtract mean) so the signal oscillates around zero.
      2. Compute sign of each sample.
      3. Count transitions (sign changes between adjacent samples).
      4. Normalize by (N - 1) to get rate in [0, 1].

    Parameters
    ----------
    segment : np.ndarray
        1-D array of raw PPG samples.

    Returns
    -------
    float
        ZSQI value in [0, 1].  Higher = more zero crossings per sample.
    """
    if len(segment) < 2:
        return np.nan

    # Step 1: remove DC (mean subtraction)
    x = segment - np.mean(segment)

    # Step 2: sign array  (+1, 0, -1)
    s = np.sign(x)

    # Step 3: count sign changes
    # np.diff gives s[n] - s[n-1]; a crossing occurs when the result is non-zero
    # Edge case: sign = 0 (sample exactly at zero) — treat as previous sign to avoid
    # spurious counts.  Replace zeros with the last non-zero sign.
    for i in range(len(s)):
        if s[i] == 0:
            s[i] = s[i - 1] if i > 0 else 1  # default to +1 at the start

    crossings = np.sum(np.diff(s) != 0)

    # Step 4: normalize
    zcr = crossings / (len(segment) - 1)
    return float(zcr)


def windowed_zcr(signal: np.ndarray, fs: float,
                 window_sec: float, step_sec: float):
    """
    Slide a window over the signal and compute ZCR per window.

    Parameters
    ----------
    signal     : 1-D PPG array
    fs         : sampling frequency (Hz)
    window_sec : window length in seconds
    step_sec   : stride in seconds

    Returns
    -------
    times : np.ndarray  -- centre time of each window (seconds)
    zcrs  : np.ndarray  -- ZSQI value per window
    """
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
        centre_time = (start + win_samples / 2) / fs
        times.append(centre_time)
        zcrs.append(compute_zcr(seg))
        start += step_samples

    return np.array(times), np.array(zcrs)


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def load_signal(csv_path: str, col) -> np.ndarray:
    """
    Load a PPG column from a CSV file.

    Parameters
    ----------
    csv_path : str   -- path to CSV
    col      : str or int -- column name or 0-based index

    Returns
    -------
    np.ndarray  1-D float array of PPG samples
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded CSV: {csv_path}  |  shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Resolve column
    if isinstance(col, int):
        if col >= df.shape[1]:
            sys.exit(f"[ERROR] Column index {col} out of range (CSV has {df.shape[1]} columns).")
        series = df.iloc[:, col]
        print(f"[INFO] Using column index {col}: '{series.name}'")
    else:
        if col not in df.columns:
            sys.exit(f"[ERROR] Column '{col}' not found. Available: {list(df.columns)}")
        series = df[col]
        print(f"[INFO] Using column: '{col}'")

    signal = pd.to_numeric(series, errors='coerce').dropna().to_numpy(dtype=float)
    print(f"[INFO] Signal length: {len(signal)} samples")
    return signal


def maybe_int(value: str):
    """Try to parse as int; fall back to string (column name)."""
    try:
        return int(value)
    except ValueError:
        return value


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZCR SQI calculator for PPG signals.")
    parser.add_argument("--csv",    required=True,          help="Path to input CSV")
    parser.add_argument("--col",    default="0",            help="Column name or index (default: 0)")
    parser.add_argument("--fs",     type=float, default=100, help="Sampling frequency in Hz (default: 100)")
    parser.add_argument("--window", type=float, default=5.0, help="Window length in seconds (default: 5.0)")
    parser.add_argument("--step",   type=float, default=1.0, help="Window step in seconds (default: 1.0)")
    parser.add_argument("--plot",   action="store_true",    help="Show plot")
    parser.add_argument("--out",    default=None,           help="Optional output CSV path for results")
    args = parser.parse_args()

    col = maybe_int(args.col)

    # 1. Load
    signal = load_signal(args.csv, col)

    # 2. Compute windowed ZCR
    times, zcrs = windowed_zcr(signal, args.fs, args.window, args.step)

    # 3. Summary statistics
    print(f"\n── ZCR SQI Results ──────────────────────────────────")
    print(f"  Windows analysed : {len(zcrs)}")
    print(f"  ZSQI mean        : {np.mean(zcrs):.4f}")
    print(f"  ZSQI std         : {np.std(zcrs):.4f}")
    print(f"  ZSQI min         : {np.min(zcrs):.4f}  @ t = {times[np.argmin(zcrs)]:.1f} s")
    print(f"  ZSQI max         : {np.max(zcrs):.4f}  @ t = {times[np.argmax(zcrs)]:.1f} s")
    print(f"─────────────────────────────────────────────────────\n")

    # Interpretation note
    mean_zcr = np.mean(zcrs)
    std_zcr  = np.std(zcrs)
    print("[INTERPRETATION]")
    print(f"  Baseline (mean ZSQI): {mean_zcr:.4f}")
    print(f"  Windows > mean + 2*std flagged as HIGH ZCR (possible jostling/contact loss):")
    threshold = mean_zcr + 2 * std_zcr
    flagged = times[zcrs > threshold]
    if len(flagged):
        for t in flagged:
            print(f"    t = {t:.1f} s  (ZSQI = {zcrs[times == t][0]:.4f})")
    else:
        print("    None — signal contact appears stable throughout.")
    print()

    # 4. Optionally save results
    if args.out:
        out_df = pd.DataFrame({"time_s": times, "ZSQI": zcrs})
        out_df.to_csv(args.out, index=False)
        print(f"[INFO] Results saved to: {args.out}")

    # 5. Optionally plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            t_axis = np.arange(len(signal)) / args.fs

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

            # Raw PPG
            ax1.plot(t_axis, signal, color='steelblue', linewidth=0.7)
            ax1.set_title("Raw PPG Signal")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True, alpha=0.3)

            # ZCR over time
            ax2.plot(times, zcrs, color='darkorange', linewidth=1.2, marker='o', markersize=3)
            ax2.axhline(mean_zcr, color='gray', linestyle='--', label=f'Mean = {mean_zcr:.4f}')
            ax2.axhline(threshold, color='red', linestyle=':', label=f'Threshold (mean+2σ) = {threshold:.4f}')
            ax2.fill_between(times, zcrs, threshold,
                             where=(zcrs > threshold), alpha=0.3, color='red', label='Flagged')
            ax2.set_title("Zero Crossing Rate SQI (ZSQI) per Window")
            ax2.set_xlabel("Window Centre Time (s)")
            ax2.set_ylabel("ZSQI")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("[WARNING] matplotlib not installed. Skipping plot.")


if __name__ == "__main__":
    main()
