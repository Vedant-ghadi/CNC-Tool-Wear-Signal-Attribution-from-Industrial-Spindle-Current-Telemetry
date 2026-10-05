"""Figures for Chapter 4. Static PNG for print; colour is never the only channel."""
from __future__ import annotations
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import theilslopes, spearmanr
from pathlib import Path

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated all-pairs safe trio
RED, GREY = "#d03b3b", "#898781"
INK, INK2, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "xtick.color": GREY, "ytick.color": GREY, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})
def tidy(ax, xlab="", ylab="", title=""):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    if title: ax.set_title(title, loc="left", pad=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    return ax

T = pd.read_csv("processed/robust_event_evidence_all_best/robust_trend_tests.csv")
F = pd.read_csv("processed/dtw_features_all_best/cnc_dtw_features.csv", low_memory=False)
B = pd.read_csv("processed/baseline_benchmark.csv")
PR = pd.read_csv("processed/phase_localisation/phase_trends_raw.csv")
PC = pd.read_csv("processed/phase_localisation/phase_trends_mean_removed.csv")
LOTO = pd.read_csv("processed/toolwear_prediction/leave_one_tool_out.csv")
REPORTED = [4, 11, 22]                 # significant AND attributable to the tool
NOT_ATTRIB = [3]                       # significant trend, fails the cohort test
lbl = {4: "m288 / tool 3", 11: "m288 / tool 7", 22: "m294 / tool 2", 3: "m288 / tool 1"}

# ---------------- Figure 4.1 : evidence attrition ----------------
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.1))
cats = ["Normal or\nuncertain", "Tool wear\nsuspected", "Unexplained\nshape anomaly",
        "Warm-up /\nthermal", "Material\nchange"]
vals = [1015, 119, 94, 21, 3]
bars = ax[0].barh(range(len(cats))[::-1], vals, color=[GREY, S1, S2, S3, "#4a3aa7"], height=.62)
ax[0].set_yticks(range(len(cats))[::-1]); ax[0].set_yticklabels(cats, fontsize=8)
for b, v in zip(bars, vals):
    ax[0].text(v + 18, b.get_y() + b.get_height()/2, f"{v:,}", va="center", fontsize=8, color=INK2)
ax[0].set_xlim(0, 1180); ax[0].xaxis.grid(True); ax[0].yaxis.grid(False)
tidy(ax[0], "Cycle–tool segments (n = 1,252)", "", "(a)  Assigned labels")

stages = ["Candidate\nstep changes", "Survive FDR\ncorrection", "Coincide with a\nrecorded boundary"]
sv = [160, 27, 3]
bb = ax[1].bar(range(3), sv, color=[GREY, S1, S2], width=.55)
for b, v in zip(bb, sv):
    ax[1].text(b.get_x() + b.get_width()/2, v + 4, str(v), ha="center", fontsize=9, color=INK)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(stages, fontsize=8)
ax[1].set_ylim(0, 185); ax[1].xaxis.grid(False)
tidy(ax[1], "", "Step changes detected", "(b)  Material-change evidence attrition")
fig.savefig(OUT / "figure_4_1_evidence_attrition.png"); plt.close(fig)

# ---------------- Figure 4.2 : the three reported windows ----------------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=False)
for a, wid in zip(axes, REPORTED):
    g = F[F.comparable_window_id == wid].sort_values("run_order")
    x = g.run_order.to_numpy(float); y = pd.to_numeric(g.residual_mean, errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    a.plot(x, y, "o", ms=3.4, color=S1, alpha=.75, mew=0)
    sl, ic, lo, hi = theilslopes(y, x, alpha=0.95)
    a.plot(x, sl*x + ic, "-", lw=2, color=RED, zorder=3)
    a.fill_between(x, lo*x + ic, hi*x + ic, color=RED, alpha=.10, lw=0)
    r = T[T.comparable_window_id == wid].iloc[0]
    tidy(a, "Cycle order within window", "Mean DTW residual (A)" if wid == 4 else "",
         f"{lbl[wid]}   τ = {r.residual_mean_kendall_tau:+.3f}")
    a.text(.03, .93, f"q = {r.residual_mean_block_q:.3f}\nlag-1 = {r.residual_mean_lag1_autocorrelation:.2f}",
           transform=a.transAxes, va="top", fontsize=7.5, color=INK2)
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=S1, label="Observed cycle"),
                        Line2D([], [], color=RED, lw=2, label="Theil–Sen fit (95% CI)")],
               loc="lower right", fontsize=7.5)
fig.suptitle("Figure 4.2  Residual trajectories of the three windows meeting every evidence requirement",
             x=0.007, ha="left", fontsize=10, weight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(OUT / "figure_4_2_trend_windows.png"); plt.close(fig)

# ---------------- Figure 4.3 : absolute vs cohort-relative tau ----------------
fig, ax = plt.subplots(figsize=(6.0, 4.4))
t = T.dropna(subset=["cohort_relative_tau"])
rep = t.comparable_window_id.isin(REPORTED)
ax.axhline(0, color=AXIS, lw=1); ax.axvline(0, color=AXIS, lw=1)
ax.axhspan(0.15, t.cohort_relative_tau.max()*1.15, color=S3, alpha=.07, lw=0)
ax.axhline(0.15, color=S3, lw=1.2, ls="--")
na = t.comparable_window_id.isin(NOT_ATTRIB)
ax.scatter(t.loc[~rep & ~na, "residual_mean_kendall_tau"], t.loc[~rep & ~na, "cohort_relative_tau"],
           s=34, facecolor="none", edgecolor=GREY, lw=1.1, label="No significant trend")
ax.scatter(t.loc[na, "residual_mean_kendall_tau"], t.loc[na, "cohort_relative_tau"],
           s=95, marker="s", color=S2, edgecolor=SURF, lw=1.4, zorder=4,
           label="Significant trend, not attributable")
ax.scatter(t.loc[rep, "residual_mean_kendall_tau"], t.loc[rep, "cohort_relative_tau"],
           s=95, marker="D", color=S1, edgecolor=SURF, lw=1.4, zorder=4,
           label="Significant and attributable")
for _, r in t[rep | na].iterrows():
    ax.annotate(lbl[int(r.comparable_window_id)],
                (r.residual_mean_kendall_tau, r.cohort_relative_tau),
                textcoords="offset points", xytext=(9, -3), fontsize=8, color=INK)
ax.text(t.residual_mean_kendall_tau.min(), 0.175, "Cohort-relative threshold (τ = 0.15)",
        fontsize=7.5, color=S3)
tidy(ax, "Kendall τ against cycle order (absolute)", "Cohort-relative Kendall τ",
     "Figure 4.3  Absolute drift against drift relative to the contemporaneous cohort")
ax.legend(loc="lower right", fontsize=7.5)
fig.savefig(OUT / "figure_4_3_cohort_relative.png"); plt.close(fig)

# ---------------- Figure 4.4 : specificity ----------------
fig, ax = plt.subplots(figsize=(6.4, 3.0))
L = LOTO.copy(); L["tool"] = L.tool.astype(str)
L = L.sort_values("naive_kendall_tau")
alarm = L.alarm.astype(str).str.lower().isin(["true", "1"]).to_numpy()
cols = [RED if a else GREY for a in alarm]
hatch = ["///" if a else "" for a in alarm]
bars = ax.barh(range(len(L)), L.naive_kendall_tau, color=cols, height=.6,
               hatch=hatch, edgecolor=SURF, lw=0)
ax.set_yticks(range(len(L)))
ax.set_yticklabels([f"Tool {t}" + ("  (target)" if a else "") for t, a in zip(L.tool, alarm)], fontsize=8.5)
for i, (v, p, r) in enumerate(zip(L.naive_kendall_tau, L.block_bootstrap_p, L.routes_significant_of_3)):
    ax.text(0.90, i, f"p = {p:.3f}    {int(r)}/3 routes", va="center", ha="left",
            fontsize=7.5, color=INK2)
ax.axvline(0, color=AXIS, lw=1); ax.set_xlim(-0.62, 1.45); ax.yaxis.grid(False)
ax.set_xticks([-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8])
tidy(ax, "Kendall τ of the health index against cycle order", "",
     "Figure 4.4  Specificity: the detector applied to every tool on machine 288, programme O005362")
ax.legend(handles=[Line2D([], [], color=RED, lw=8, label="Alarm raised"),
                   Line2D([], [], color=GREY, lw=8, label="No alarm")],
          loc="upper left", fontsize=8, bbox_to_anchor=(0.005, 0.99))
fig.savefig(OUT / "figure_4_4_specificity.png"); plt.close(fig)

# ---------------- Figure 4.5 : the three inference routes ----------------
FLOOR = 5e-4                       # q of 0 is unplottable on a log axis
d = T.copy()
for c in ["residual_mean_kendall_q", "residual_mean_hamed_rao_q", "residual_mean_block_q"]:
    d[c] = d[c].clip(lower=FLOOR)
anysig = d[(d.residual_mean_kendall_q.le(.05)) | (d.residual_mean_hamed_rao_q.le(.05))
           | (d.residual_mean_block_q.le(.05))].sort_values("residual_mean_block_q")
fig, ax = plt.subplots(figsize=(7.6, 4.6))
y = np.arange(len(anysig))[::-1]
for i, (_, r) in zip(y, anysig.iterrows()):
    qs = [r.residual_mean_kendall_q, r.residual_mean_hamed_rao_q, r.residual_mean_block_q]
    ax.plot([min(qs), max(qs)], [i, i], "-", color=GRID, lw=2, zorder=1, solid_capstyle="round")
for col, lab, c, mk, sz in [("residual_mean_kendall_q", "Naive Kendall + FDR", GREY, "o", 46),
                            ("residual_mean_hamed_rao_q", "Hamed–Rao + FDR", S2, "^", 52),
                            ("residual_mean_block_q", "Block bootstrap + FDR", S1, "D", 58)]:
    ax.scatter(anysig[col], y, s=sz, marker=mk, color=c, edgecolor=SURF, lw=.9, label=lab, zorder=3)
ax.axvline(0.05, color=RED, lw=1.4, ls="--", zorder=2)
ax.axvspan(FLOOR*0.8, 0.05, color=S3, alpha=.05, lw=0, zorder=0)
ax.set_xscale("log"); ax.set_xlim(FLOOR*0.8, 1.15)
ax.text(0.056, 0.15, "q = 0.05", fontsize=8, color=RED)
labels = [(lbl[int(w)] if int(w) in REPORTED + NOT_ATTRIB else f"Window {int(w)}") for w in anysig.comparable_window_id]
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
for t, w in zip(ax.get_yticklabels(), anysig.comparable_window_id):
    if int(w) in REPORTED: t.set_color(INK); t.set_weight("bold")
    elif int(w) in NOT_ATTRIB: t.set_color(S2); t.set_weight("bold")
ax.yaxis.grid(False)
tidy(ax, "FDR-adjusted q-value (log scale; values below %.0e are plotted at the floor)" % FLOOR, "",
     "Figure 4.5  Windows significant on at least one route: 11 naive, 10 Hamed–Rao, 4 block bootstrap")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
fig.savefig(OUT / "figure_4_5_routes.png"); plt.close(fig)

# ---------------- Figure 4.6 : autocorrelation confound ----------------
fig, ax = plt.subplots(figsize=(6.0, 4.3))
naive = T.residual_mean_kendall_q.le(0.05); block = T.residual_mean_block_q.le(0.05)
grp = [(~naive, "Not significant on either route", GREY, "o", 34, "none"),
       (naive & ~block, "Significant only before correction", S2, "s", 55, S2),
       (block, "Significant after correction", S1, "D", 85, S1)]
for m, lab, c, mk, sz, fc in grp:
    ax.scatter(T.loc[m, "residual_mean_lag1_autocorrelation"], T.loc[m, "residual_mean_kendall_tau"],
               s=sz, marker=mk, facecolor=("none" if fc == "none" else fc),
               edgecolor=c, lw=1.3, label=lab, zorder=3)
ax.axhline(0, color=AXIS, lw=1)
offsets = {4: (-10, -16, "right"), 11: (-12, 12, "right"), 22: (-10, 10, "right"),
           3: (12, -6, "left")}
for _, r in T[T.comparable_window_id.isin(REPORTED + NOT_ATTRIB)].iterrows():
    dx, dy, ha = offsets[int(r.comparable_window_id)]
    ax.annotate(lbl[int(r.comparable_window_id)],
                (r.residual_mean_lag1_autocorrelation, r.residual_mean_kendall_tau),
                textcoords="offset points", xytext=(dx, dy), fontsize=7.5, color=INK, ha=ha,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=.7, shrinkA=0, shrinkB=6))
ax.set_ylim(T.residual_mean_kendall_tau.min()-.10, T.residual_mean_kendall_tau.max()+.16)
tidy(ax, "Lag-1 autocorrelation of the residual series", "Kendall τ against cycle order",
     "Figure 4.6  Apparent trend against serial dependence")
ax.legend(loc="lower left", fontsize=7.5)
fig.savefig(OUT / "figure_4_6_autocorrelation.png"); plt.close(fig)

# ---------------- Figure 4.7 : feature benchmark ----------------
fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
feats = [("residual_mean", "DTW residual mean", True), ("dtw_shape_distance", "DTW shape distance", True),
         ("warping_amount", "Warping amount", True), ("current_mean", "Mean current", False),
         ("current_p95", "95th percentile current", False), ("current_integral_As", "Current–time integral", False)]
counts = [int(B[f"{c}_block_q"].le(.05).mul(B[f"{c}_tau"].gt(0)).sum()) for c, _, _ in feats]
cols = [S1 if al else S2 for _, _, al in feats]
hat = ["" if al else "///" for _, _, al in feats]
bars = ax[0].barh(range(len(feats))[::-1], counts, color=cols, hatch=hat, edgecolor=SURF, lw=0, height=.62)
ax[0].set_yticks(range(len(feats))[::-1]); ax[0].set_yticklabels([n for _, n, _ in feats], fontsize=8.5)
for b, v in zip(bars, counts):
    ax[0].text(v + .1, b.get_y()+b.get_height()/2, str(v), va="center", fontsize=9, color=INK)
ax[0].set_xlim(0, 7.2); ax[0].yaxis.grid(False)
tidy(ax[0], "Windows with a significant rising trend (of 40)", "", "(a)  Detection by feature")
ax[0].legend(handles=[Line2D([], [], color=S1, lw=8, label="Derived from alignment"),
                      Line2D([], [], color=S2, lw=8, label="No alignment")], loc="lower right", fontsize=8)

rho = spearmanr(B.residual_mean_tau, B.current_mean_tau, nan_policy="omit").statistic
lim = [min(B.residual_mean_tau.min(), B.current_mean_tau.min())-.06,
       max(B.residual_mean_tau.max(), B.current_mean_tau.max())+.06]
ax[1].plot(lim, lim, "-", color=AXIS, lw=1, zorder=1)
rep = B.comparable_window_id.isin(REPORTED)
ax[1].scatter(B.loc[~rep, "current_mean_tau"], B.loc[~rep, "residual_mean_tau"],
              s=34, facecolor="none", edgecolor=GREY, lw=1.1, label="Window not reported", zorder=3)
ax[1].scatter(B.loc[rep, "current_mean_tau"], B.loc[rep, "residual_mean_tau"],
              s=95, marker="D", color=S1, edgecolor=SURF, lw=1.4, label="Reported window", zorder=4)
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].text(.04, .95, f"Spearman ρ = {rho:+.3f}", transform=ax[1].transAxes, va="top", fontsize=9, color=INK)
tidy(ax[1], "Kendall τ of mean cutting current (no alignment)", "Kendall τ of DTW residual mean",
     "(b)  The two features rank windows almost identically")
ax[1].legend(loc="lower right", fontsize=8)
fig.suptitle("Figure 4.7  Alignment-based features against scalar summaries of the same signal",
             x=0.007, ha="left", fontsize=10, weight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(OUT / "figure_4_7_benchmark.png"); plt.close(fig)

# ---------------- Figure 4.8 : phase localisation, the key figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), sharey=True)
w = 0.4
for a, wid in zip(axes, REPORTED):
    r = PR[PR.comparable_window_id == wid].reset_index(drop=True)
    c = PC[PC.comparable_window_id == wid].reset_index(drop=True)
    idx = np.arange(len(r))
    a.bar(idx - w/2, r.kendall_tau, w, color=S1, edgecolor=SURF, lw=.7, label="Residual as computed")
    a.bar(idx + w/2, c.kendall_tau, w, color=S2, hatch="///", edgecolor=SURF, lw=.7,
          label="Cycle-level offset removed")
    for i, sig in enumerate(r.rising.astype(str).str.lower().isin(["true"])):
        if sig: a.plot(i - w/2, r.kendall_tau[i] + .05, marker="*", ms=8, color=RED, zorder=5)
    a.axhline(0, color=AXIS, lw=1)
    a.set_xticks(idx); a.set_xticklabels([f"{i*10}–{(i+1)*10}" for i in idx], fontsize=6.5, rotation=45)
    a.xaxis.grid(False)
    tidy(a, "Phase of the cut (%)", "Kendall τ against cycle order" if wid == 4 else "", lbl[wid])
lo = min(PR[PR.comparable_window_id.isin(REPORTED)].kendall_tau.min(),
         PC[PC.comparable_window_id.isin(REPORTED)].kendall_tau.min())
hi = max(PR[PR.comparable_window_id.isin(REPORTED)].kendall_tau.max(),
         PC[PC.comparable_window_id.isin(REPORTED)].kendall_tau.max())
axes[0].set_ylim(lo - .12, hi + .30)
fig.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color=S1, label="Residual as computed"),
    plt.Rectangle((0, 0), 1, 1, color=S2, hatch="///", label="Cycle-level offset removed"),
    Line2D([], [], marker="*", ls="", ms=8, color=RED, label="Significant after correction")],
    loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, fontsize=8)
fig.suptitle("Figure 4.8  Phase-resolved trends before and after removing each cycle's own mean residual",
             x=0.007, y=0.985, ha="left", fontsize=10, weight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, .855]); fig.savefig(OUT / "figure_4_8_phase_localisation.png"); plt.close(fig)

# ---------------- Figure 4.9 : resampling resolution ----------------
fig, ax = plt.subplots(figsize=(6.2, 3.9))
Bs = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000])
ax.plot(Bs, 1/(Bs+1), "o-", color=S1, lw=2, ms=6, label="Smallest attainable p-value,  1/(B+1)")
ax.axhline(0.05/400, color=RED, lw=1.6, ls="--", label="Required for FDR at q = 0.05 with 400 tests")
ax.axhline(0.05/40, color=S3, lw=1.6, ls=":", label="Required for FDR at q = 0.05 with 40 tests")
ax.axvspan(100, 8000, color=RED, alpha=.05, lw=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(2e-5, 3e-2)
ax.annotate("500 resamples: no phase\ntest can reach significance", xy=(500, 1/501),
            xytext=(1100, 6e-3), fontsize=7.5, color=INK2, ha="left",
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
ax.annotate("10,000 resamples used", xy=(10000, 1/10001), xytext=(160, 1.45e-4), fontsize=7.5,
            color=INK2, ha="left", arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
tidy(ax, "Bootstrap resamples per test (B)", "p-value",
     "Figure 4.9  Resampling resolution must satisfy the multiplicity correction")
ax.legend(loc="lower left", fontsize=7.5, bbox_to_anchor=(0.0, 0.0))
fig.savefig(OUT / "figure_4_9_resolution.png"); plt.close(fig)

print("figures written:")
for p in sorted(OUT.glob("*.png")): print("  ", p.name, f"{p.stat().st_size/1024:.0f} KB")
