import pandas as pd
import pingouin as pg

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("hrv_data.csv")

# -----------------------------
# Compute ICC per site vs ECG
# -----------------------------
def compute_icc_vs_ecg(df, ecg_label="ECG"):

    results = []

    sites = [s for s in df["site"].unique() if s != ecg_label]

    for site in sites:

        # Keep only ECG + current site
        sub = df[df["site"].isin([ecg_label, site])].copy()

        # ICC needs "long format"
        # targets = subject, raters = site type
        icc_table = pg.intraclass_corr(
            data=sub,
            targets="subject",
            raters="site",
            ratings="hrv"
        )

        icc3 = icc_table[icc_table["Type"] == "ICC3"]["ICC"].values[0]
        ci95 = icc_table[icc_table["Type"] == "ICC3"]["CI95%"].values[0]

        results.append({
            "site": site,
            "icc3": icc3,
            "ci95_low": ci95[0],
            "ci95_high": ci95[1],
            "n_subjects": sub["subject"].nunique()
        })

    return pd.DataFrame(results).sort_values("icc3", ascending=False)


# -----------------------------
# Run
# -----------------------------
results = compute_icc_vs_ecg(df)

print(results)
results.to_csv("icc_results.csv", index=False)