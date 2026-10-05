"""Alignment-based features against scalar baselines (RQ4).

Applies the identical inference chain -- Kendall tau, circular block bootstrap,
Benjamini-Hochberg -- to the DTW residual and to features computed without any
alignment, so that the contribution of the alignment machinery can be measured
rather than assumed.  Benchmarking an elastic similarity measure against a
simple baseline is established practice (Bagnall et al., 2017).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr
from pipeline_common import benjamini_hochberg, block_bootstrap_trend

FEATURES = {
    "residual_mean":        "DTW residual mean (full alignment)",
    "dtw_shape_distance":   "DTW shape distance (alignment, amplitude-free)",
    "warping_amount":       "Warping amount (time-axis distortion)",
    "current_mean":         "Mean cutting current (no alignment)",
    "current_p95":          "95th percentile current (no alignment)",
    "current_integral_As":  "Current-time integral (no alignment)",
}
B, SEED = 2000, 0

f = pd.read_csv("processed/dtw_features_all_best/cnc_dtw_features.csv", low_memory=False)
rows = []
for wid, g in f.groupby("comparable_window_id", sort=True):
    g = g.sort_values("run_order")
    x = g["run_order"].to_numpy(float)
    rec = {"comparable_window_id": int(wid), "machine_id": int(g.machine_id.iloc[0]),
           "tool": str(g.tool.iloc[0]), "n": len(g)}
    for col in FEATURES:
        if col not in g.columns: continue
        y = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6:
            rec[f"{col}_tau"] = np.nan; rec[f"{col}_block_p"] = np.nan; continue
        tau, _ = kendalltau(x[ok], y[ok])
        rec[f"{col}_tau"] = float(tau)
        rec[f"{col}_block_p"] = block_bootstrap_trend(y[ok], n_resamples=B, seed=SEED)["p"]
    rows.append(rec)
out = pd.DataFrame(rows)
for col in FEATURES:
    if f"{col}_block_p" in out.columns:
        out[f"{col}_block_q"] = benjamini_hochberg(out[f"{col}_block_p"])
        out[f"{col}_sig"] = out[f"{col}_block_q"].le(0.05) & out[f"{col}_tau"].gt(0)
out.to_csv("processed/baseline_benchmark.csv", index=False)

print("=" * 74)
print("RQ4 - DETECTION: windows with a significant rising trend (block bootstrap + FDR)")
print("=" * 74)
print(f"{'Feature':<48}{'Sig':>5}{'  mean |tau|':>12}")
base = out["residual_mean_sig"]
for col, lab in FEATURES.items():
    if f"{col}_sig" not in out.columns: continue
    print(f"{lab:<48}{int(out[f'{col}_sig'].sum()):>5}{np.nanmean(np.abs(out[f'{col}_tau'])):>12.3f}")
print()
print("Agreement with the DTW residual, over all 40 windows:")
for col, lab in FEATURES.items():
    if col == "residual_mean" or f"{col}_sig" not in out.columns: continue
    both = int((base & out[f"{col}_sig"]).sum())
    dtw_only = int((base & ~out[f"{col}_sig"]).sum())
    other_only = int((~base & out[f"{col}_sig"]).sum())
    r = spearmanr(out["residual_mean_tau"], out[f"{col}_tau"], nan_policy="omit")
    print(f"  {lab:<46} both {both}  DTW-only {dtw_only}  other-only {other_only}"
          f"   rho(tau) {r.statistic:+.3f}")
print()
print("Windows found significant by any feature:")
cols = [c for c in out.columns if c.endswith("_sig")]
anysig = out[out[cols].any(axis=1)]
show = ["comparable_window_id", "machine_id", "tool", "n"] + [c for c in out.columns if c.endswith("_tau")]
print(anysig[show].round(3).to_string(index=False))
