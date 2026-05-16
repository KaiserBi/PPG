import csv
with open("ecg_data.csv") as f:
    rows = [r for r in csv.reader(f) if len(r) >= 3]
ts = [int(r[0]) for r in rows]
print(f"{len(ts)} rows")
print(f"first 10 timestamps: {ts[:10]}")
print(f"last 5 timestamps: {ts[-5:]}")
print(f"span: {(ts[-1]-ts[0])/1e6:.3f} s  -> implied fs: {len(ts)/((ts[-1]-ts[0])/1e6):.1f} Hz")
diffs = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
print(f"median Δt: {sorted(diffs)[len(diffs)//2]} µs (target 2500)")
print(f"max Δt: {max(diffs)} µs")