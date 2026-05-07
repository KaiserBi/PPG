import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def visualize_ecg(csv_path, title="ECG Signal", xlabel="Time (s)", ylabel="Amplitude"):
    """
    Simple ECG/signal visualization from CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with two columns (time, signal values)
    title : str
        Title for the plot
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    """
    # Read CSV file
    data = pd.read_csv(csv_path)
    
    # Extract time and signal (handles both headerless and header-based CSVs)
    time = data.iloc[:, 0].values.astype(float)
    signal = data.iloc[:, 1].values.astype(float)
    
    # Normalize time to start from 0 and convert to seconds for readability
    time = (time - time[0]) / 1000  # Convert to seconds
    
    # Create figure and plot
    plt.figure(figsize=(12, 5))
    plt.plot(time, signal, linewidth=1)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_ecg_multiple(csv_paths, titles=None):
    """
    Visualize multiple ECG signals in subplots.
    
    Parameters:
    -----------
    csv_paths : list
        List of paths to CSV files
    titles : list
        List of titles for each subplot
    """
    n = len(csv_paths)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4*n))
    
    if n == 1:
        axes = [axes]
    
    for idx, csv_path in enumerate(csv_paths):
        data = pd.read_csv(csv_path)
        time = data.iloc[:, 0].values.astype(float)
        signal = data.iloc[:, 1].values.astype(float)
        time = (time - time[0]) / 1000
        
        axes[idx].plot(time, signal, linewidth=1)
        if titles:
            axes[idx].set_title(titles[idx], fontweight='bold')
        axes[idx].set_xlabel("Time (s)")
        axes[idx].set_ylabel("Amplitude")
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    csv_file = "ecg_data.csv"
    visualize_ecg(csv_file, title="ecg")
