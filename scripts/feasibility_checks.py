"""The three feasibility checks reported in Section 3.4, recomputed from the
round-2 outputs so that every figure in that table is reproducible."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from pipeline_common import robust_scale

RT = Path("processed/cnc_run_tool_features.csv")
TS = Path("processed/cnc_time_series_features.csv.gz")
MIN_RUNS, HALF, MIN_DROP = 20, 5, 3.0

rt = pd.read_csv(RT, low_memory=False)
rt["tool"] = rt["tool"].astype(str); rt["program"] = rt["program"].astype(str)
rt["started_at"] = pd.to_datetime(rt["started_at"], utc=True, errors="coerce")
for c in ["current_mean", "cutting_s", "machine_id"]:
    rt[c] = pd.to_numeric(rt[c], errors="coerce")
ready = rt[rt["analysis_ready_run"].astype(str).str.lower().isin(["true", "1"])]

print("=" * 68)
print("CHECK 1 - TOOL LIFE OVER alarm")
cols = ["machine_id", "tool", "alarms", "current1", "is_cutting", "created_at"]
rows, machines, before, after = 0, set(), [], []
for chunk in pd.read_csv(TS, usecols=lambda c: c in cols, chunksize=200_000, low_memory=False):
    chunk["is_cutting"] = chunk["is_cutting"].astype(str).str.lower().isin(["true", "1"])
    hit = chunk["alarms"].astype(str).str.contains("TOOL LIFE OVER", regex=False)
    rows += int(hit.sum()); machines |= set(chunk.loc[hit, "machine_id"].unique())
    cut = chunk[chunk["is_cutting"]]
    h = cut["alarms"].astype(str).str.contains("TOOL LIFE OVER", regex=False)
    after.append(cut.loc[h, "current1"].dropna()); before.append(cut.loc[~h, "current1"].dropna())
a = pd.concat(after); b = pd.concat(before)
print(f"  rows carrying the alarm : {rows:,}")
print(f"  machines affected       : {sorted(machines)}")
print(f"  cutting current, alarm active   : median {a.median():.2f} A (n={len(a):,})")
print(f"  cutting current, alarm inactive : median {b.median():.2f} A (n={len(b):,})")
print(f"  difference              : {100*(a.median()-b.median())/b.median():+.2f}%")

print("=" * 68)
print("CHECK 2 - tool-change resets (sustained downward current steps)")
seqs, steps, detail = 0, 0, []
for (m, p, t), g in ready.groupby(["machine_id", "program", "tool"], sort=True):
    if t == "-" or len(g) < MIN_RUNS: continue
    seqs += 1
    y = g.sort_values("started_at")["current_mean"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2 * HALF: continue
    scale = robust_scale(np.diff(y))
    if not np.isfinite(scale) or scale == 0: continue
    for i in range(HALF, y.size - HALF):
        drop = np.median(y[i - HALF:i]) - np.median(y[i:i + HALF])
        if drop / scale >= MIN_DROP:
            steps += 1; detail.append((int(m), p, t, i, round(float(drop / scale), 1))); break
print(f"  sequences with >= {MIN_RUNS} analysis-ready runs : {seqs}")
print(f"  sustained downward steps (>= {MIN_DROP} robust SD, sustained {HALF} runs) : {steps}")
for d in detail: print(f"    machine {d[0]} / {d[1]} / tool {d[2]}: at run {d[3]}, {d[4]} SD")

print("=" * 68)
print("CHECK 3 - tool-life coverage")
cov = ready[ready["tool"] != "-"].groupby(["machine_id", "program", "tool"])["cutting_s"].sum() / 3600
print(f"  tool sequences (any length)  : {len(cov)}")
print(f"  cutting hours per sequence   : median {cov.median():.3f}  "
      f"p90 {cov.quantile(0.9):.2f}  max {cov.max():.2f}")
long = cov[cov.index.map(lambda k: (ready.groupby(['machine_id','program','tool']).size().get(k,0) >= MIN_RUNS))]
print(f"  restricted to >= {MIN_RUNS} runs ({len(long)} sequences): "
      f"median {long.median():.2f}  range {long.min():.2f}-{long.max():.2f} hours")
w = pd.read_csv("processed/cnc_dtw_best_windows.csv")
wh = w.groupby(["machine_id", "program", "tool"]).apply(
    lambda g: float(g["median_cutting_s"].iloc[0] * g["num_runs"].iloc[0] / 3600), include_groups=False)
print(f"  the {len(w)} comparable windows : {wh.min():.2f}-{wh.max():.2f} cutting hours "
      f"(median {wh.median():.2f})")
print("=" * 68)
