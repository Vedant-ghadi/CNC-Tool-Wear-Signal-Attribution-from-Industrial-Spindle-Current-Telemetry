"""Review tables and plots for the DTW feature outputs.

Fixes applied relative to the original version
----------------------------------------------
D3   Reference self-comparisons (distance 0 by construction) no longer exist,
     but the review stage now also excludes any non-finite rows from the
     robust z baseline rather than letting them set the median and MAD.
A12  First/last comparison blocks are non-overlapping (they previously shared
     elements whenever a window had fewer than six valid runs).
C1   Scoring-only runs -- alarm, post-idle and boundary runs that the old
     pipeline filtered out before detection -- are marked in every ranked
     table so a reviewer can see which candidates come from them.
"""

from __future__ import annotations

import argparse
import os
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import first_last_medians, robust_z_by_group, to_numeric, write_manifest

try:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


DEFAULT_INPUT_DIR = Path("processed/dtw_features_all_best")
DEFAULT_OUTPUT_DIR = Path("processed/dtw_review_all_best")

TREND_METRICS = [
    "dtw_shape_distance", "dtw_amplitude_distance", "residual_mean",
    "residual_p95", "residual_abs_mean", "current_integral_As",
]


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(input_dir / "cnc_dtw_features.csv", low_memory=False)
    residuals = pd.read_csv(input_dir / "cnc_dtw_residuals_long.csv.gz", low_memory=False)
    references = pd.read_csv(input_dir / "cnc_dtw_references.csv", low_memory=False)

    to_numeric(features, TREND_METRICS + [
        "comparable_window_id", "machine_id", "run_id", "run_order", "runtime_s",
        "warping_amount", "batch_or_job_boundary", "in_first_runs_after_long_idle",
        "thermal_state_score",
    ])
    to_numeric(residuals, ["comparable_window_id", "run_id", "run_order",
                           "template_phase", "residual", "abs_residual"])
    for frame in (features, residuals):
        for col in ["program", "tool"]:
            if col in frame.columns:
                frame[col] = frame[col].astype(str)

    if "in_reference_set" in features.columns:
        features["in_reference_set"] = (
            features["in_reference_set"].astype(str).str.lower().isin(["true", "1"])
        )
    else:
        features["in_reference_set"] = True
    return features, residuals, references


def add_review_scores(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    for col in TREND_METRICS:
        if col in features.columns:
            features[f"{col}_robust_z"] = robust_z_by_group(features, col)
    features["positive_residual_score"] = features["residual_mean_robust_z"].clip(lower=0)
    features["shape_outlier_score"] = features["dtw_shape_distance_robust_z"].clip(lower=0)
    features["amplitude_outlier_score"] = features["dtw_amplitude_distance_robust_z"].clip(lower=0)
    return features


def trend_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_id, group in features.groupby("comparable_window_id", sort=False):
        group = group.sort_values("run_order", kind="stable")
        x = group["run_order"].to_numpy(dtype=float)
        row = {
            "comparable_window_id": window_id,
            "machine_id": int(group["machine_id"].iloc[0]),
            "program": group["program"].iloc[0],
            "tool": group["tool"].iloc[0],
            "num_runs": len(group),
            "scoring_only_runs": int((~group["in_reference_set"]).sum()),
        }
        for col in TREND_METRICS:
            if col not in group.columns:
                continue
            y = group[col].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3:
                row[f"{col}_corr_run_order"] = float(pd.Series(x[valid]).corr(pd.Series(y[valid])))
                row[f"{col}_slope"] = float(np.polyfit(x[valid], y[valid], 1)[0])
                first, last, block = first_last_medians(y[valid])
                row[f"{col}_first_median"] = first
                row[f"{col}_last_median"] = last
                row[f"{col}_delta"] = last - first
                row[f"{col}_block"] = block
            else:
                for suffix in ["corr_run_order", "slope", "first_median", "last_median", "delta", "block"]:
                    row[f"{col}_{suffix}"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["residual_mean_corr_run_order", "residual_mean_delta"],
                           ascending=[False, False], kind="stable")


def window_summary(features: pd.DataFrame) -> pd.DataFrame:
    grouped = features.groupby(["comparable_window_id", "machine_id", "program", "tool"], sort=False)
    out = grouped.agg(
        num_runs=("run_id", "nunique"),
        scoring_only_runs=("in_reference_set", lambda s: int((~s).sum())),
        runtime_s_median=("runtime_s", "median"),
        runtime_s_cv=("runtime_s", lambda s: s.std(ddof=1) / s.mean() if s.mean() else np.nan),
        shape_distance_median=("dtw_shape_distance", "median"),
        shape_distance_p95=("dtw_shape_distance", lambda s: s.quantile(0.95)),
        amplitude_distance_median=("dtw_amplitude_distance", "median"),
        amplitude_distance_p95=("dtw_amplitude_distance", lambda s: s.quantile(0.95)),
        residual_mean_median=("residual_mean", "median"),
        residual_mean_p95=("residual_mean", lambda s: s.quantile(0.95)),
        residual_abs_mean_median=("residual_abs_mean", "median"),
        residual_abs_mean_p95=("residual_abs_mean", lambda s: s.quantile(0.95)),
    ).reset_index()
    return out.sort_values(["num_runs", "runtime_s_cv"], ascending=[False, True], kind="stable")


RANKED_COLUMNS = [
    "comparable_window_id", "machine_id", "program", "tool", "run_id", "run_order",
    "in_reference_set", "runtime_s", "dtw_shape_distance", "dtw_shape_distance_robust_z",
    "dtw_amplitude_distance", "dtw_amplitude_distance_robust_z", "residual_mean",
    "residual_mean_robust_z", "residual_p95", "residual_abs_mean", "current_integral_As",
    "dtw_shape_z_mean_across_references", "dtw_shape_z_min_across_references",
    "batch_or_job_boundary", "long_idle_before_run", "in_first_runs_after_long_idle",
    "alarm_active", "thermal_state_score",
]


def write_ranked_tables(features: pd.DataFrame, output_dir: Path) -> None:
    cols = [c for c in RANKED_COLUMNS if c in features.columns]
    for score, filename in [
        ("shape_outlier_score", "top_shape_distance_runs.csv"),
        ("amplitude_outlier_score", "top_amplitude_shift_runs.csv"),
        ("positive_residual_score", "top_positive_residual_runs.csv"),
    ]:
        features.sort_values(score, ascending=False)[cols].head(100).to_csv(
            output_dir / filename, index=False)


def plot_window_features(features: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for window_id, group in features.groupby("comparable_window_id", sort=False):
        group = group.sort_values("run_order", kind="stable")
        title = (f"Window {window_id} | machine {group['machine_id'].iloc[0]} | "
                 f"{group['program'].iloc[0]} | tool {group['tool'].iloc[0]}")
        series = [
            ("shape DTW", group["dtw_shape_distance"].to_numpy(dtype=float), "#2563eb"),
            ("amplitude DTW", group["dtw_amplitude_distance"].to_numpy(dtype=float), "#b35c00"),
            ("residual mean", group["residual_mean"].to_numpy(dtype=float), "#1f7a5a"),
            ("current integral", group["current_integral_As"].to_numpy(dtype=float), "#5d4aa3"),
        ]
        if not HAS_MATPLOTLIB:
            svg = line_panels_svg(group["run_order"].to_numpy(dtype=float), series, title)
            (plot_dir / f"window_{int(window_id):03d}_dtw_features.svg").write_text(svg, encoding="utf-8")
            continue

        scoring_only = group.loc[~group["in_reference_set"], "run_order"].to_numpy(dtype=float)
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
        for ax, (label, values, colour) in zip(axes, series):
            ax.plot(group["run_order"], values, marker="o", lw=1.5, color=colour)
            ax.set_ylabel(label)
            for x in scoring_only:
                ax.axvline(x, color="#cc3333", lw=0.8, alpha=0.35)
        axes[2].axhline(0, color="#999999", lw=0.8)
        axes[3].set_xlabel("run order in comparable window (red = scoring-only run)")
        fig.suptitle(title, x=0.01, ha="left", fontsize=12)
        fig.savefig(plot_dir / f"window_{int(window_id):03d}_dtw_features.png", dpi=150)
        plt.close(fig)


def plot_residual_heatmaps(residuals: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for window_id, group in residuals.groupby("comparable_window_id", sort=False):
        pivot = group.pivot_table(index="run_order", columns="template_phase",
                                  values="residual", aggfunc="mean")
        if pivot.empty:
            continue
        if not HAS_MATPLOTLIB:
            svg = heatmap_svg(pivot, f"Window {int(window_id)} residual heatmap")
            (plot_dir / f"window_{int(window_id):03d}_residual_heatmap.svg").write_text(svg, encoding="utf-8")
            continue

        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        vmax = np.nanquantile(np.abs(pivot.to_numpy(dtype=float)), 0.98)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower",
                       cmap="coolwarm", vmin=-vmax, vmax=vmax,
                       extent=[0, 1, pivot.index.min(), pivot.index.max()])
        ax.set_xlabel("template phase")
        ax.set_ylabel("run order")
        ax.set_title(f"Window {int(window_id)} residual heatmap", loc="left")
        fig.colorbar(im, ax=ax, label="aligned current - template")
        fig.savefig(plot_dir / f"window_{int(window_id):03d}_residual_heatmap.png", dpi=150)
        plt.close(fig)


def line_panels_svg(x: np.ndarray, series: list[tuple[str, np.ndarray, str]], title: str) -> str:
    width, panel_h, left, right, top = 1000, 145, 70, 25, 45
    height = top + panel_h * len(series) + 25
    plot_w = width - left - right
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(x_min) or x_max == x_min:
        x_min, x_max = 0.0, 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="14" y="24" font-family="Arial" font-size="16" fill="#111">{escape(title)}</text>',
    ]
    for panel_i, (label, y, colour) in enumerate(series):
        y = y.astype(float)
        y_valid = y[np.isfinite(y)]
        y_min, y_max = (float(np.min(y_valid)), float(np.max(y_valid))) if y_valid.size else (0.0, 1.0)
        if y_max == y_min:
            y_min, y_max = y_min - 0.5, y_max + 0.5
        pad = (y_max - y_min) * 0.08
        y_min, y_max = y_min - pad, y_max + pad
        y0 = top + panel_i * panel_h
        base = y0 + panel_h - 28
        span_h = panel_h - 45

        points = []
        for xi, yi in zip(x, y):
            if not np.isfinite(xi) or not np.isfinite(yi):
                continue
            px = left + (xi - x_min) / (x_max - x_min) * plot_w
            py = base - (yi - y_min) / (y_max - y_min) * span_h
            points.append(f"{px:.1f},{py:.1f}")

        parts.extend([
            f'<text x="14" y="{y0 + 18}" font-family="Arial" font-size="12" fill="#333">{escape(label)}</text>',
            f'<line x1="{left}" y1="{base}" x2="{width - right}" y2="{base}" stroke="#ddd"/>',
            f'<line x1="{left}" y1="{y0 + 12}" x2="{left}" y2="{base}" stroke="#ddd"/>',
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colour}" stroke-width="2"/>',
        ])
        for point in points:
            px, py = point.split(",")
            parts.append(f'<circle cx="{px}" cy="{py}" r="2.4" fill="{colour}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def heatmap_svg(pivot: pd.DataFrame, title: str) -> str:
    data = pivot.to_numpy(dtype=float)
    if data.size == 0:
        data = np.zeros((1, 1))
    max_cols = 220
    if data.shape[1] > max_cols:
        bins = np.array_split(np.arange(data.shape[1]), max_cols)
        data = np.column_stack([np.nanmean(data[:, b], axis=1) for b in bins])

    rows, cols = data.shape
    cell_w = max(2.0, min(6.0, 880 / max(cols, 1)))
    cell_h = max(4.0, min(12.0, 420 / max(rows, 1)))
    left, top = 70, 42
    width = int(left + cols * cell_w + 30)
    height = int(top + rows * cell_h + 45)
    vmax = float(np.nanquantile(np.abs(data), 0.98)) if np.isfinite(data).any() else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="14" y="24" font-family="Arial" font-size="16" fill="#111">{escape(title)}</text>',
    ]
    for r in range(rows):
        for c in range(cols):
            value = data[r, c]
            colour = residual_colour(0.0 if not np.isfinite(value) else float(value), vmax)
            x = left + c * cell_w
            y = top + (rows - 1 - r) * cell_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w + 0.2:.1f}" '
                         f'height="{cell_h + 0.2:.1f}" fill="{colour}"/>')
    parts.extend([
        f'<text x="14" y="{top + rows * cell_h / 2:.1f}" font-family="Arial" font-size="12" fill="#333">run order</text>',
        f'<text x="{left + cols * cell_w / 2:.1f}" y="{height - 10}" font-family="Arial" font-size="12" fill="#333">template phase</text>',
        "</svg>",
    ])
    return "\n".join(parts)


def residual_colour(value: float, vmax: float) -> str:
    t = max(-1.0, min(1.0, value / vmax))
    if t >= 0:
        r, g, b = 255, int(245 - 105 * t), int(245 - 160 * t)
    else:
        t = -t
        r, g, b = int(245 - 155 * t), int(245 - 85 * t), 255
    return f"rgb({r},{g},{b})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create review tables and plots for DTW feature outputs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, residuals, references = load_inputs(args.input_dir)
    features = add_review_scores(features)

    window_summary(features).to_csv(args.output_dir / "window_summary.csv", index=False)
    trend_summary(features).to_csv(args.output_dir / "trend_summary.csv", index=False)
    references.to_csv(args.output_dir / "references.csv", index=False)
    write_ranked_tables(features, args.output_dir)

    plot_count = 0
    if not args.skip_plots:
        plot_window_features(features, args.output_dir)
        plot_residual_heatmaps(residuals, args.output_dir)
        plot_dir = args.output_dir / "plots"
        plot_count = len(list(plot_dir.glob("*.png"))) + len(list(plot_dir.glob("*.svg")))

    summary = (
        f"windows reviewed: {features['comparable_window_id'].nunique():,}\n"
        f"feature rows: {len(features):,}\n"
        f"  scoring-only rows: {int((~features['in_reference_set']).sum()):,}\n"
        f"residual rows: {len(residuals):,}\n"
        f"plots: {plot_count:,}\n"
    )
    (args.output_dir / "review_summary.txt").write_text(summary, encoding="utf-8")
    write_manifest(args.output_dir, "review_dtw_features", args)
    print(summary)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
