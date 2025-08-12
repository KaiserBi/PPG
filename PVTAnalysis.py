import os
import pandas as pd
from datetime import datetime

def process_file(file_path):
    # 1. Get the time from the metadata
    with open(file_path, "r") as f:
        lines = f.readlines()
    time_value = lines[4].split(",")[1].strip()  # 5th line (0-based index 4)

    # 2. Find the line number where data starts
    data_start = None
    for i, line in enumerate(lines):
        if line.strip() == "---DATA---":
            data_start = i + 2  # skip the ---DATA--- line and the column headers
            break

    # 3. Read numeric data from that line onward
    df = pd.read_csv(file_path, skiprows=data_start)
    col1 = pd.to_numeric(df.iloc[:, 0], errors="coerce")  # reaction times
    over_500 = col1[col1 > 500]

    # 4. Calculate lapses and 1/mean
    lapses_count = len(over_500)
    inv_mean = 1 / over_500.mean() if not over_500.empty else None

    return time_value, lapses_count, inv_mean

def extract_datetime_from_filename(filename):
    # Split by underscores, take the second-to-last segment
    date_str = filename.split("_")[-2] + "_" + filename.split("_")[-1].replace(".csv", "")
    # Parse into datetime
    return datetime.strptime(date_str, "%m-%d-%Y_%H-%M")

def main():
    data_folder = os.path.join("./data", "PVTs")
    results = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            time_value, lapses_count, inv_mean = process_file(file_path)
            results.append([filename, time_value, lapses_count, inv_mean])

    # Save all results
    results_df = pd.DataFrame(results, columns=["Filename", "Time", "Lapses", "1/Mean"])
    results_df["Datetime"] = results_df["Filename"].apply(extract_datetime_from_filename)

    start_time = results_df["Datetime"].min()
    results_df["HoursSinceStart"] = results_df["Datetime"].apply(lambda x: (x - start_time).total_seconds() / 3600)

    results_df = results_df.sort_values(by="Datetime").drop(columns="Datetime")
    results_df.to_csv("pvt_results.csv", index=False)
    print("Saved results to pvt_results.csv")


if __name__ == "__main__":
    main()