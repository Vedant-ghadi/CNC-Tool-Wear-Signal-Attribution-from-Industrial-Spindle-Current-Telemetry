"""Schematic and flow diagrams for Chapters 1 and 3.

Coordinates are in inches so that rounded corners stay isotropic: every axes is
created with xlim = (0, width) and ylim = (0, height).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path("figures"); OUT.mkdir(exist_ok=True)

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
RED, GREY = "#d03b3b", "#898781"
INK, INK2, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# tints for box fills, all tested against INK/INK2 text
T1, T2, T3 = "#e4eefb", "#fdeae1", "#e0f5ee"
TG, TN = "#f0efea", "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.dpi": 300, "savefig.facecolor": SURF,
})


def canvas(w: float, h: float):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc=TN, ec=AXIS, fs=8.0, weight="normal",
        tc=INK, lw=1.0, ls="-", r=0.055, va="center", tx=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    yy = y + h / 2 if va == "center" else y + h - 0.13
    ax.text(tx if tx is not None else x + w / 2, yy, text,
            ha="center" if tx is None else "left", va=va, fontsize=fs,
            color=tc, weight=weight, linespacing=1.4, zorder=3)


def arrow(ax, p0, p1, color=AXIS, lw=1.15, ls="-", rad=0.0, style="-|>", ms=10):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
        linewidth=lw, linestyle=ls, shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}", zorder=1))


def band(ax, x, y, w, h, fc, label=None, lx=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=fc, edgecolor="none", zorder=0))
    if label:
        ax.text(lx, y + h / 2, label, rotation=90, ha="center", va="center",
                fontsize=8.5, weight="bold", color=INK2, zorder=1)


# ══════════════ Figure 1.1 — the attribution problem ══════════════
W, H = 9.6, 5.75
fig, ax = canvas(W, H)

sources = [
    ("Tool wear", S3), ("Machine ageing\nand spindle drift", S3),
    ("Sensor drift", S3), ("Machine warm-up\nand thermal state", S1),
    ("Material or\nbatch change", S1), ("Setup change", S1),
]
bw, gap = 1.42, 0.14
x0 = (W - (6 * bw + 5 * gap)) / 2
for i, (name, col) in enumerate(sources):
    x = x0 + i * (bw + gap)
    box(ax, x, 4.72, bw, 0.72, name, fc=T3 if col == S3 else T1, ec=col, fs=7.8)
    arrow(ax, (x + bw / 2, 4.72), (x + bw / 2, 4.30), color=col, lw=1.1)

box(ax, x0, 3.58, 6 * bw + 5 * gap, 0.72,
    "Spindle current — one aggregate channel, one value per second",
    fc=TG, ec=INK2, fs=9.2, weight="bold")
ax.text(W / 2 - 0.24, 3.42, "every load on the spindle sums into this one number",
        ha="right", va="top", fontsize=7.2, color=GREY, style="italic")

arrow(ax, (W / 2, 3.58), (W / 2, 3.22), color=INK2, lw=1.3)
box(ax, 3.42, 2.48, 2.76, 0.74, "An observed change in cutting load",
    fc=TN, ec=INK2, fs=9, weight="bold")

box(ax, 0.19, 2.48, 2.90, 0.74,
    "Which of the six produced it?\nThe signal alone cannot say.",
    fc="#fdeeee", ec=RED, fs=8.2, tc=RED, weight="bold")
arrow(ax, (3.42, 2.85), (3.09, 2.85), color=RED, lw=1.3)
ax.text(7.85, 2.85, "forward direction: physically determined\ninverse direction: "
        "underdetermined by the signal alone", ha="center", va="center",
        fontsize=7.4, color=INK2, style="italic")

lanes = [
    ("An independent record exists", S1, T1,
     "A measured idle period; a recorded\nbatch or job boundary",
     "Thermal state · Material or batch\nchange · Setup change",
     "Separable"),
    ("A contemporaneous cohort exists", S3, T3,
     "Other tools on the same machine and\nprogramme, overlapping in time",
     "Tool wear, separated from machine\nageing and sensor drift",
     "Separable only where a cohort exists"),
    ("Signal evidence alone", GREY, TG,
     "Deviation from the reference, with no\ncorroborating record of any kind",
     "Unexplained shape anomaly",
     "Not attributable to a named cause"),
]
lw_, lgap = 2.98, 0.16
lx0 = (W - (3 * lw_ + 2 * lgap)) / 2
for i, (head, col, tint, ev, res, verdict) in enumerate(lanes):
    x = lx0 + i * (lw_ + lgap)
    ax.add_patch(FancyBboxPatch((x, 0.22), lw_, 1.86,
                 boxstyle="round,pad=0,rounding_size=0.07",
                 facecolor=tint, edgecolor=col, linewidth=1.1, zorder=2))
    ax.text(x + lw_ / 2, 1.90, head, ha="center", va="center",
            fontsize=8.4, weight="bold", color=INK, zorder=3)
    ax.text(x + lw_ / 2, 1.50, ev, ha="center", va="center",
            fontsize=7.3, color=INK2, linespacing=1.4, zorder=3)
    ax.text(x + lw_ / 2, 0.98, res, ha="center", va="center",
            fontsize=7.6, color=INK, linespacing=1.4, zorder=3)
    ax.add_patch(FancyBboxPatch((x + 0.30, 0.34), lw_ - 0.60, 0.34,
                 boxstyle="round,pad=0,rounding_size=0.05",
                 facecolor=col, edgecolor="none", zorder=3))
    ax.text(x + lw_ / 2, 0.51, verdict, ha="center", va="center",
            fontsize=7.6, weight="bold", color="white", zorder=4)
    # fan out from the bottom edge of the observed-change box
    src = 3.42 + (i + 0.5) * (2.76 / 3)
    arrow(ax, (src, 2.48), (x + lw_ / 2, 2.12), color=col, lw=1.1,
          rad=0.0 if i == 1 else (0.14 if i == 0 else -0.14))

fig.savefig(OUT / "figure_1_1_attribution_problem.png", bbox_inches="tight")
plt.close(fig)


# ══════════════ Figure 3.1 — pipeline architecture ══════════════
W, H = 9.6, 8.4
fig, ax = canvas(W, H)

stages = [
    ("1", "preprocess_cnc.py", "Parse, segment into cycles, derive the cutting\n"
     "threshold, build cycle and cycle–tool features",
     "1,489 cycles · 3,816 cycle–tool segments\n20 context and quality flags"),
    ("2", "find_comparable_windows.py", "Group by machine, programme and tool; require\n"
     "≥ 12 consecutive cycles within 10% of the median",
     "1,641 candidate windows → 40 retained"),
    ("3", "create_comparable_timeseries_datasets.py",
     "Extract each tool's cutting segment; split the\nreference set from the scoring set",
     "1,252 cycle–tool traces\n1,112 reference · 140 scoring-only"),
    ("4", "build_dtw_features.py", "Leave-one-out median template; band-constrained\n"
     "alignment; shape, amplitude and residual features",
     "1,252 feature rows\n126,459 phase-resolved residuals"),
    ("5", "discover_dtw_patterns.py", "Trend statistics, changepoints and cohort\n"
     "comparisons across every window",
     "Per-window trend and changepoint tables"),
    ("6", "detect_dtw_event_candidates.py", "Assemble candidate events from the\n"
     "pattern output",
     "Candidate event set"),
    ("7", "build_robust_event_evidence.py", "Three inference routes, FDR correction,\n"
     "cohort-relative drift, common-mode checks",
     "robust_trend_tests.csv\ncommon_mode_shift_checks.csv"),
    ("8", "score_event_candidates_v2.py", "Score each candidate against its evidence\n"
     "and assign a confidence band",
     "200 scored candidates"),
    ("9", "apply_algorithmic_labels.py", "Resolve categories by priority; project run\n"
     "labels onto timestamp rows",
     "1,252 labelled segments\n21 · 3 · 94 · 119 · 1,015"),
]

bh, bg = 0.545, 0.115
top = 7.86
sx, sw = 0.92, 3.62
ox, ow = 4.92, 4.44

phases = [(0, 2, "#eef3fa", "Preparation"), (3, 3, "#fdf1ea", "Representation"),
          (4, 8, "#eaf6f1", "Inference and labelling")]
for a, b, col, name in phases:
    y_top = top - a * (bh + bg)
    y_bot = top - b * (bh + bg) - bh
    band(ax, 0.14, y_bot - 0.09, W - 0.28, (y_top - y_bot) + 0.18,
         col, name, lx=0.42)

for i, (num, script, what, out) in enumerate(stages):
    y = top - i * (bh + bg) - bh
    box(ax, sx, y, sw, bh, "", fc=TN, ec=AXIS)
    ax.add_patch(FancyBboxPatch((sx + 0.06, y + 0.06), 0.30, bh - 0.12,
                 boxstyle="round,pad=0,rounding_size=0.04",
                 facecolor=INK2, edgecolor="none", zorder=3))
    ax.text(sx + 0.21, y + bh / 2, num, ha="center", va="center",
            fontsize=8.5, weight="bold", color="white", zorder=4)
    ax.text(sx + 0.46, y + bh - 0.155, script, ha="left", va="center",
            fontsize=8.0, weight="bold", color=INK, family="monospace", zorder=3)
    ax.text(sx + 0.46, y + 0.175, what, ha="left", va="center",
            fontsize=6.9, color=INK2, linespacing=1.35, zorder=3)
    if i < len(stages) - 1:
        arrow(ax, (sx + sw / 2, y), (sx + sw / 2, y - bg), color=INK2, lw=1.2)
    arrow(ax, (sx + sw, y + bh / 2), (ox, y + bh / 2), color=GREY, lw=0.9, ls=(0, (3, 2)))
    box(ax, ox, y, ow, bh, out, fc="#ffffff", ec=GRID, fs=7.4, tc=INK2)

ax.text(sx, top + 0.16, "Pipeline stage", fontsize=8.2, weight="bold",
        color=INK2, ha="left", va="bottom")
ax.text(ox, top + 0.16, "Principal output", fontsize=8.2, weight="bold",
        color=INK2, ha="left", va="bottom")

y_qa = top - 9 * (bh + bg) - bh + 0.12
box(ax, 0.14, y_qa - 0.16, W - 0.28, 0.50,
    "Every stage writes run_manifest.json — arguments, package versions, timestamp.  "
    "Seeds fixed throughout; re-execution is byte-identical.\n"
    "review_dtw_features.py provides per-window QA plots off the main path;  "
    "verify_fixes.py asserts one condition per corrected defect — 25 of 25 pass.",
    fc="#f7f6f1", ec=AXIS, fs=7.3, tc=INK2, ls=(0, (4, 2)))

off = [("baseline_benchmark.py", "Six-feature comparison\nagainst scalar summaries"),
       ("phase_localisation_test.py", "Phase-resolved trends,\nraw and mean-removed"),
       ("feasibility_checks.py", "Alarm, reset and\ncoverage checks")]
obw = (W - 0.28 - 2 * 0.16) / 3
for i, (name, what) in enumerate(off):
    x = 0.14 + i * (obw + 0.16)
    box(ax, x, 0.22, obw, 0.72, "", fc=TN, ec=S2)
    ax.text(x + obw / 2, 0.79, name, ha="center", va="center", fontsize=7.6,
            weight="bold", color=INK, family="monospace", zorder=3)
    ax.text(x + obw / 2, 0.47, what, ha="center", va="center", fontsize=7.0,
            color=INK2, linespacing=1.35, zorder=3)
    arrow(ax, (x + obw / 2, y_qa - 0.16), (x + obw / 2, 0.94), color=S2,
          lw=1.0, ls=(0, (3, 2)))
ax.text(0.14, 1.05, "Off-pipeline analyses reported in Chapter 4",
        ha="left", va="bottom", fontsize=7.6, color=S2, weight="bold")

fig.savefig(OUT / "figure_3_1_pipeline.png", bbox_inches="tight")
plt.close(fig)


# ══════════════ Figure 3.2 — data reduction flow ══════════════
W, H = 9.6, 5.0
fig, ax = canvas(W, H)

row1 = [("1,088,518", "raw sample rows", "50 files · 6 machines\n9–23 July 2026"),
        ("1,489", "machining cycles", "split on part counter,\nprogramme, 900 s gap"),
        ("1,377", "usable cycles", "complete, untruncated,\npart count present"),
        ("1,244", "analysis-ready cycles", "at least one cutting\nsample · 220.47 h"),
        ("3,816", "cycle–tool segments", "one per tool used\nwithin a cycle")]
row2 = [("1,641", "candidate windows", "≥ 12 consecutive cycles\nwithin 10% of the median"),
        ("40", "comparable windows", "best per machine,\nprogramme and tool"),
        ("1,252", "segments scored", "1,112 reference\n+ 140 scoring-only"),
        ("126,459", "phase-resolved residuals", "10 phase bins\nper window")]


def flowrow(items, y, h, cols, reverse=False):
    """Lay a row of boxes out. reverse=True runs right-to-left (snake layout)."""
    n = len(items)
    g = 0.42
    bw_ = (W - 0.28 - (n - 1) * g) / n
    xs = []
    for i, (num, name, note) in enumerate(items):
        pos = (n - 1 - i) if reverse else i
        x = 0.14 + pos * (bw_ + g)
        xs.append(x)
        box(ax, x, y, bw_, h, "", fc=TN, ec=cols[i])
        ax.text(x + bw_ / 2, y + h - 0.30, num, ha="center", va="center",
                fontsize=13, weight="bold", color=cols[i], zorder=3)
        ax.text(x + bw_ / 2, y + h - 0.62, name, ha="center", va="center",
                fontsize=8.0, color=INK, zorder=3)
        ax.text(x + bw_ / 2, y + 0.24, note, ha="center", va="center",
                fontsize=6.9, color=GREY, linespacing=1.35, zorder=3)
        if i < n - 1:
            if reverse:
                arrow(ax, (x, y + h / 2), (x - g, y + h / 2), color=INK2, lw=1.2)
            else:
                arrow(ax, (x + bw_, y + h / 2), (x + bw_ + g, y + h / 2),
                      color=INK2, lw=1.2)
    return bw_, g, xs


c1 = [INK2, S1, S1, S1, S1]
c2 = [S3, S3, S3, S3]
bw1, g1, x1 = flowrow(row1, 3.30, 1.30, c1)
bw2, g2, x2 = flowrow(row2, 0.95, 1.30, c2, reverse=True)

ax.text(0.14, 4.82, "Segmentation and feature construction",
        fontsize=8.6, weight="bold", color=INK2, ha="left", va="center")
ax.text(W - 0.14, 2.42, "Window construction and scoring  ▸  continues right to left",
        fontsize=8.6, weight="bold", color=INK2, ha="right", va="center")

# drop-offs, centred on the transition each describes
for gi, txt in [(1, "112 cycles incomplete,\ntruncated or unnumbered"),
                (2, "133 cycles with no\ncutting samples")]:
    x = 0.14 + (gi + 1) * (bw1 + g1) - g1 / 2
    ax.text(x, 3.18, txt, ha="center", va="top", fontsize=6.6, color=RED,
            linespacing=1.35)

# snake connector: straight down the right-hand side, no crossings
arrow(ax, (x1[4] + bw1 / 2, 3.30), (x2[0] + bw2 / 2, 2.25),
      color=INK2, lw=1.2)

box(ax, 0.14, 0.14, W - 0.28, 0.52,
    "Machines 283, 479 and 481 report no tool identity — 705,493 rows, "
    "64.8% of the dataset — and cannot support tool-level analysis.\n"
    "The 40 retained windows span 3 machines, 12 programmes and 9 tools.",
    fc="#fdeeee", ec=RED, fs=7.4, tc=INK2)

fig.savefig(OUT / "figure_3_2_data_reduction.png", bbox_inches="tight")
plt.close(fig)


# ══════════════ Figure 3.3 — the DTW representation, real signal ══════════════
S = pd.read_csv("processed/comparable_timeseries_all_best/samples_long.csv.gz",
                low_memory=False)
R = pd.read_csv("processed/dtw_features_all_best/cnc_dtw_residuals_long.csv.gz",
                low_memory=False)
WID = 11
sw_ = S[S.comparable_window_id == WID]
rw = R[R.comparable_window_id == WID]
late = int(rw.run_order.max())
rl = rw[rw.run_order == late].sort_values("template_phase")

fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.05))
for a in axes:
    a.set_facecolor(SURF)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.spines["left"].set_color(AXIS); a.spines["bottom"].set_color(AXIS)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    a.tick_params(colors=GREY, labelsize=7.5)

a = axes[0]
ref_ids = sw_[sw_.in_reference_set == True].run_id.unique()
for rid in ref_ids:
    g = sw_[sw_.run_id == rid].sort_values("seconds_in_tool_cut")
    a.plot(g.seconds_in_tool_cut, g.current1, color=GREY, lw=0.55, alpha=0.30)
a.plot(rl.template_seconds, rl.template_current, color=S1, lw=2.2, zorder=5)
a.set_title("(a)  Reference set and template", loc="left", fontsize=8.6,
            weight="bold", color=INK, pad=6)
a.set_xlabel("Seconds within the cutting segment", fontsize=7.8, color=INK2)
a.set_ylabel("Spindle current (A)", fontsize=7.8, color=INK2)
a.legend(handles=[Line2D([], [], color=GREY, lw=1, label=f"{len(ref_ids)} reference cycles"),
                  Line2D([], [], color=S1, lw=2.2, label="Leave-one-out median")],
         fontsize=6.8, frameon=False, loc="upper right")

a = axes[1]
a.plot(rl.template_phase, rl.template_current, color=S1, lw=1.9, zorder=4)
a.plot(rl.template_phase, rl.aligned_current, color=S2, lw=1.4, zorder=4)
a.fill_between(rl.template_phase, rl.template_current, rl.aligned_current,
               where=rl.aligned_current >= rl.template_current,
               color=S2, alpha=0.22, lw=0, zorder=2)
a.fill_between(rl.template_phase, rl.template_current, rl.aligned_current,
               where=rl.aligned_current < rl.template_current,
               color=S1, alpha=0.16, lw=0, zorder=2)
a.set_title("(b)  One cycle after alignment", loc="left", fontsize=8.6,
            weight="bold", color=INK, pad=6)
a.set_xlabel("Phase within the cutting segment", fontsize=7.8, color=INK2)
a.set_ylabel("Spindle current (A)", fontsize=7.8, color=INK2)
a.legend(handles=[Line2D([], [], color=S1, lw=1.9, label="Template"),
                  Line2D([], [], color=S2, lw=1.4, label=f"Cycle {late} (aligned)"),
                  Line2D([], [], color=S2, lw=5, alpha=.3, label="Residual")],
         fontsize=6.8, frameon=False, loc="upper right")

a = axes[2]
bins = np.linspace(0, 1, 11)
rw_ = rl.dropna(subset=["template_phase", "residual"])   # the same cycle as (b)
idx = np.clip(np.digitize(rw_.template_phase, bins) - 1, 0, 9)
med = [rw_.residual[idx == k].median() for k in range(10)]
a.bar(np.arange(10) + 0.5, med, width=0.74,
      color=[S3 if m >= 0 else S1 for m in med], edgecolor="none")
a.axhline(0, color=INK2, lw=0.9)
a.set_xticks(np.arange(0, 11, 2)); a.set_xticklabels([f"{v:.1f}" for v in bins[::2]])
a.set_title(f"(c)  Residual of cycle {late} by phase bin", loc="left",
            fontsize=8.6, weight="bold", color=INK, pad=6)
a.set_xlabel("Phase within the cutting segment", fontsize=7.8, color=INK2)
a.set_ylabel("Median residual (A)", fontsize=7.8, color=INK2)
a.set_xlim(0, 10)

fig.tight_layout(pad=0.9)
fig.savefig(OUT / "figure_3_3_dtw_representation.png", bbox_inches="tight")
plt.close(fig)


# ══════════════ Figure 3.5 — evidence and label resolution ══════════════
W, H = 9.6, 5.1
fig, ax = canvas(W, H)

cats = [
    ("Machine warm-up\nor thermal state", S1, T1,
     "A measured idle period or an\nidentified warm-up programme,\n"
     "AND a deviation from the reference"),
    ("Material or\nsetup change", S1, T1,
     "A step change located by PELT\nsegmentation coinciding with a\n"
     "recorded batch or job boundary"),
    ("Unexplained\nshape anomaly", S2, T2,
     "Outlier in both shape and\namplitude against every\n"
     "reference construction at once"),
    ("Tool wear\nsuspected", S3, T3,
     "A corrected, cohort-relative\nrising trend that survives the\n"
     "four conditions of Figure 3.4"),
]
cw, cg = 2.22, 0.20
cx0 = (W - (4 * cw + 3 * cg)) / 2
for i, (name, col, tint, ev) in enumerate(cats):
    x = cx0 + i * (cw + cg)
    box(ax, x, 3.42, cw, 1.40, "", fc=tint, ec=col, lw=1.15)
    ax.text(x + cw / 2, 4.55, name, ha="center", va="center", fontsize=8.4,
            weight="bold", color=INK, linespacing=1.35, zorder=3)
    ax.plot([x + 0.34, x + cw - 0.34], [4.24, 4.24], color=col, lw=0.8,
            alpha=0.55, zorder=3)
    ax.text(x + cw / 2, 3.84, ev, ha="center", va="center", fontsize=6.9,
            color=INK2, linespacing=1.45, zorder=3)
    arrow(ax, (x + cw / 2, 3.42), (x + cw / 2, 2.86), color=col, lw=1.15)

ax.text(cx0, 4.94, "Four explanatory categories, each requiring evidence of a different kind",
        fontsize=8.4, weight="bold", color=INK2, ha="left", va="center")

box(ax, cx0, 2.10, 4 * cw + 3 * cg, 0.76, "", fc=TG, ec=INK2, lw=1.15)
ax.text(cx0 + 0.18, 2.70, "Priority resolution", fontsize=8.2, weight="bold",
        color=INK, ha="left", va="center", zorder=3)
ax.text(cx0 + 0.18, 2.34,
        "thermal state  →  material change  →  shape anomaly  →  tool wear",
        fontsize=8.6, color=INK, ha="left", va="center", zorder=3, family="monospace")
ax.text(W - cx0 - 0.18, 2.48, "most externally corroborated\n→ most inferential",
        fontsize=7.2, color=GREY, ha="right", va="center", style="italic",
        linespacing=1.4, zorder=3)

arrow(ax, (W / 2, 2.10), (W / 2, 1.68), color=INK2, lw=1.3)

box(ax, cx0, 0.86, 4 * cw + 3 * cg, 0.82, "", fc=TN, ec=INK2, lw=1.3)
ax.text(cx0 + 0.18, 1.44, "One resolved label per cycle and tool",
        fontsize=8.6, weight="bold", color=INK, ha="left", va="center", zorder=3)
ax.text(cx0 + 0.18, 1.10,
        "every category that fired is retained alongside it, with a confidence band; "
        "run labels are projected onto the constituent timestamp rows",
        fontsize=7.3, color=INK2, ha="left", va="center", zorder=3)

box(ax, cx0, 0.16, 4 * cw + 3 * cg, 0.52,
    "No category meets its evidence threshold  →  normal_or_uncertain   "
    "(1,015 of 1,252 segments)",
    fc="#f7f6f1", ec=GRID, fs=7.6, tc=INK2)

fig.savefig(OUT / "figure_3_5_label_resolution.png", bbox_inches="tight")
plt.close(fig)


# ══════════════ Figure 3.4 — the cohort test ══════════════
W, H = 9.6, 3.55
fig, ax = canvas(W, H)

box(ax, 0.14, 0.16, 2.72, 3.24, "", fc=TG, ec=INK2, lw=1.15)
ax.text(1.50, 3.14, "Four conditions, all required", ha="center", va="center",
        fontsize=8.5, weight="bold", color=INK, zorder=3)
conds = [
    ("1", "A monotone rising trend in the\nresidual across cycles"),
    ("2", "Significance under a null preserving\nthe observed serial dependence,\nafter FDR correction"),
    ("3", "Drift exceeding the contemporaneous\ncohort by the stated margin"),
    ("4", "No common-mode pattern in which\nthe cohort as a whole moves together"),
]
yc = 2.62
for num, txt in conds:
    ax.add_patch(FancyBboxPatch((0.30, yc - 0.13), 0.26, 0.26,
                 boxstyle="round,pad=0,rounding_size=0.04",
                 facecolor=S3, edgecolor="none", zorder=3))
    ax.text(0.43, yc, num, ha="center", va="center", fontsize=7.6,
            weight="bold", color="white", zorder=4)
    ax.text(0.66, yc, txt, ha="left", va="center", fontsize=6.9, color=INK2,
            linespacing=1.4, zorder=3)
    yc -= 0.72

panels = [
    ("Tool-specific drift", "attributed", S3,
     [0.0, 0.28, 0.55, 0.82, 1.10], [0.0, 0.04, 0.06, 0.05, 0.08], "conditions 1–4 met"),
    ("Common-mode drift", "not attributed", RED,
     [0.0, 0.30, 0.58, 0.86, 1.12], [0.0, 0.34, 0.62, 0.90, 1.16], "fails condition 3"),
    ("Cohort moves together", "disqualified", RED,
     [0.0, 0.26, 0.52, 0.80, 1.05], None, "fails condition 4"),
]
px0, pg = 3.06, 0.20
pw = (W - 0.14 - px0 - 2 * pg) / 3
x = np.linspace(0, 1, 5)
for i, (title, verdict, col, tgt, coh, note) in enumerate(panels):
    x0 = px0 + i * (pw + pg)
    box(ax, x0, 0.16, pw, 3.24, "", fc=TN, ec=AXIS)
    ax.text(x0 + pw / 2, 3.14, title, ha="center", va="center", fontsize=8.3,
            weight="bold", color=INK, zorder=3)
    sub = fig.add_axes([(x0 + 0.26) / W, 1.02 / H, (pw - 0.48) / W, 1.72 / H])
    sub.set_facecolor(TN)
    for s in ("top", "right"):
        sub.spines[s].set_visible(False)
    sub.spines["left"].set_color(AXIS); sub.spines["bottom"].set_color(AXIS)
    sub.set_xticks([]); sub.set_yticks([])
    sub.set_xlim(-0.06, 1.06); sub.set_ylim(-0.14, 1.42)
    if coh is None:
        for k, off in enumerate([0.12, -0.10]):
            sub.plot(x, np.array(tgt) + off, color=GREY, lw=1.5,
                     alpha=0.9, zorder=2)
        sub.plot(x, tgt, color=S3, lw=2.1, zorder=3)
        sub.text(0.04, 1.34, "every tool rises together", fontsize=6.2,
                 color=GREY, style="italic", va="top")
    else:
        sub.plot(x, coh, color=GREY, lw=1.6, ls=(0, (4, 2)), zorder=2)
        sub.plot(x, tgt, color=S3, lw=2.1, zorder=3)
    if i == 0:                       # direct labels replace a floating legend
        sub.text(0.99, tgt[-1] + 0.08, "target tool", fontsize=6.2, color=S3,
                 weight="bold", va="bottom", ha="right")
        sub.text(0.99, coh[-1] + 0.12, "cohort median", fontsize=6.2,
                 color=GREY, va="bottom", ha="right")
    sub.set_xlabel("cycle order", fontsize=6.6, color=GREY, labelpad=2)
    sub.set_ylabel("residual", fontsize=6.6, color=GREY, labelpad=2)
    ax.text(x0 + pw / 2, 0.80, note, ha="center", va="center", fontsize=6.9,
            color=INK2, style="italic", zorder=3)
    ax.add_patch(FancyBboxPatch((x0 + 0.34, 0.32), pw - 0.68, 0.34,
                 boxstyle="round,pad=0,rounding_size=0.05",
                 facecolor=col, edgecolor="none", zorder=3))
    ax.text(x0 + pw / 2, 0.49, verdict, ha="center", va="center", fontsize=7.8,
            weight="bold", color="white", zorder=4)

fig.savefig(OUT / "figure_3_4_cohort_test.png", bbox_inches="tight")
plt.close(fig)

print("diagrams written")
