"""Discover DTW behaviour patterns and candidate event rules.

Fixes applied relative to the original version
----------------------------------------------
F1  **Every split position was emitted as a changepoint.**  The old
    ``detect_changepoints`` looped over all valid splits x 4 metrics x every
    window and wrote a row for each, producing a 400 KB table that downstream
    stages then thresholded at ``abs_effect_z >= 2.5``.  One real step change
    therefore generated a cluster of adjacent "candidates" and inflated every
    support count.  Only the best split per (window, metric) is now kept.

F2  ``effect_z`` was ``(median difference) / MAD``: an effect size with no
    sqrt(n) term and no null distribution, named and thresholded as though it
    were a test statistic.  Each retained split now carries a **permutation
    p-value** (the null being "no dependence on run order") and a
    Mann-Whitney U p-value, both FDR-corrected across the whole family.

F5  Clustering used a fixed ``k = 5``, farthest-point initialisation seeded
    from ``arr[0]`` -- maximally outlier-sensitive on features that *are*
    outlier scores -- and standardised the already per-window robust-z inputs
    a second time.  ``k`` is now chosen by silhouette score over a range,
    initialisation is k-means++ with a fixed seed, and the double
    normalisation is gone.

A6  Removed the dead ``if ...: pass`` block in ``load_inputs`` and the
    ``DataFrame.get(col, 0).fillna(...)`` call in ``rule_suggestions`` that
    would raise ``AttributeError`` on an int whenever the column was absent.

A12 Non-overlapping first/last comparison blocks.

Drift scores are now scale-free: the old ``positive_drift_score`` multiplied a
correlation by a raw delta in amperes, so windows on high-current machines
dominated the cross-window ranking regardless of how strong their trend was.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu

from pipeline_common import (
    benjamini_hochberg,
    block_bootstrap_trend,
    first_last_medians,
    permutation_p_value,
    robust_scale,
    robust_z_by_group,
    safe_float,
    to_numeric,
    write_manifest,
)

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


DEFAULT_DTW_DIR = Path("processed/dtw_features_all_best")
DEFAULT_OUTPUT_DIR = Path("processed/dtw_patterns_all_best")

FEATURE_COLS = [
    "dtw_shape_distance", "dtw_amplitude_distance", "residual_mean", "residual_p95",
    "residual_abs_mean", "residual_abs_p95", "current_integral_As", "warping_amount",
    "runtime_s",
]

CHANGEPOINT_METRICS = ["residual_mean", "dtw_amplitude_distance", "dtw_shape_distance",
                       "current_integral_As"]
TREND_METRICS = ["residual_mean", "residual_p95", "residual_abs_mean",
                 "current_integral_As", "dtw_amplitude_distance"]

CLUSTER_COLS = [
    "dtw_shape_distance_robust_z", "dtw_amplitude_distance_robust_z",
    "residual_mean_robust_z", "residual_abs_mean_robust_z",
    "current_integral_As_robust_z", "warping_amount_robust_z", "runtime_s_robust_z",
]


def load_inputs(dtw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(dtw_dir / "cnc_dtw_features.csv", low_memory=False)
    residuals = pd.read_csv(dtw_dir / "cnc_dtw_residuals_long.csv.gz", low_memory=False)

    to_numeric(features, FEATURE_COLS + [
        "comparable_window_id", "machine_id", "run_id", "run_order",
        "batch_or_job_boundary", "long_idle_before_run",
        "in_first_runs_after_long_idle", "thermal_state_score", "current_mean",
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
    return add_derived_features(features), residuals


def add_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    for col in FEATURE_COLS:
        if col in features.columns:
            features[f"{col}_robust_z"] = robust_z_by_group(features, col)

    features["shape_z_pos"] = features["dtw_shape_distance_robust_z"].clip(lower=0)
    features["amplitude_z_pos"] = features["dtw_amplitude_distance_robust_z"].clip(lower=0)
    features["residual_mean_z_pos"] = features["residual_mean_robust_z"].clip(lower=0)
    features["residual_abs_z_pos"] = features["residual_abs_mean_robust_z"].clip(lower=0)

    features = features.sort_values(["comparable_window_id", "run_order"], kind="stable")
    by_window = features.groupby("comparable_window_id", sort=False)
    for col in ["dtw_shape_distance", "dtw_amplitude_distance", "residual_mean",
                "current_integral_As"]:
        features[f"{col}_delta_prev"] = by_window[col].diff()
        features[f"{col}_rolling3"] = by_window[col].transform(
            lambda s: s.rolling(3, min_periods=1, center=True).median()
        )
    return features


# --------------------------------------------------------------------------- #
#  F1/F2: one changepoint per (window, metric), with a null distribution
# --------------------------------------------------------------------------- #

def split_effect(values: np.ndarray, split: int) -> float:
    left, right = values[:split], values[split:]
    if left.size == 0 or right.size == 0:
        return np.nan
    left_med, right_med = np.median(left), np.median(right)
    scale = robust_scale(np.concatenate([left - left_med, right - right_med]))
    if not np.isfinite(scale) or scale == 0:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float((right_med - left_med) / scale)


def detect_changepoints(
    features: pd.DataFrame, min_side: int = 5, n_permutations: int = 2000,
) -> pd.DataFrame:
    rows = []
    for window_id, group in features.groupby("comparable_window_id", sort=False):
        group = group.sort_values("run_order", kind="stable").reset_index(drop=True)
        if len(group) < min_side * 2:
            continue

        for metric in CHANGEPOINT_METRICS:
            if metric not in group.columns:
                continue
            values = group[metric].to_numpy(dtype=float)
            if np.isfinite(values).sum() < min_side * 2:
                continue

            candidates = range(min_side, len(group) - min_side + 1)
            effects = {split: split_effect(values, split) for split in candidates}
            effects = {k: v for k, v in effects.items() if np.isfinite(v)}
            if not effects:
                continue

            # F1: keep only the strongest split for this (window, metric).
            best_split = max(effects, key=lambda s: abs(effects[s]))
            effect = effects[best_split]

            # F2: a null distribution for the statistic actually maximised.
            p_permutation = permutation_p_value(
                effect, values, best_split, n_permutations=n_permutations,
                seed=int(window_id), min_side=min_side,
            )
            left, right = values[:best_split], values[best_split:]
            left, right = left[np.isfinite(left)], right[np.isfinite(right)]
            try:
                p_mannwhitney = float(mannwhitneyu(left, right, alternative="two-sided").pvalue)
            except ValueError:
                p_mannwhitney = np.nan

            near = group.loc[max(0, best_split - 2):min(len(group) - 1, best_split + 2)]
            rows.append({
                "comparable_window_id": int(window_id),
                "machine_id": int(group["machine_id"].iloc[0]),
                "program": group["program"].iloc[0],
                "tool": group["tool"].iloc[0],
                "metric": metric,
                "splits_scanned": len(effects),
                "split_after_run_order": int(group.loc[best_split - 1, "run_order"]),
                "first_run_after_split": int(group.loc[best_split, "run_id"]),
                "left_runs": int(best_split),
                "right_runs": int(len(group) - best_split),
                "left_median": float(np.median(left)) if left.size else np.nan,
                "right_median": float(np.median(right)) if right.size else np.nan,
                "effect_size": effect,
                "abs_effect_size": abs(effect),
                "p_permutation": p_permutation,
                "p_mannwhitney": p_mannwhitney,
                "boundary_runs_near_split": int(
                    near["batch_or_job_boundary"].fillna(0).sum()
                    if "batch_or_job_boundary" in near.columns else 0
                ),
                "thermal_runs_near_split": int(
                    near["in_first_runs_after_long_idle"].fillna(0).sum()
                    if "in_first_runs_after_long_idle" in near.columns else 0
                ),
                "scoring_only_runs_near_split": int((~near["in_reference_set"]).sum()),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # F2/F3: control the family-wise discovery rate across every split retained.
    out["q_permutation"] = benjamini_hochberg(out["p_permutation"])
    out["q_mannwhitney"] = benjamini_hochberg(out["p_mannwhitney"])
    out["significant_fdr05"] = out["q_permutation"].le(0.05)
    return out.sort_values("abs_effect_size", ascending=False, kind="stable")


# --------------------------------------------------------------------------- #
#  Trends
# --------------------------------------------------------------------------- #

def detect_trends(features: pd.DataFrame, block_resamples: int = 1000) -> pd.DataFrame:
    """Window-level trend descriptors, with autocorrelation-aware p-values.

    ``*_kendall_p`` assumes independent runs and is kept only for comparison;
    ``*_block_q`` is the circular-block-bootstrap p-value after FDR and is what
    downstream stages should threshold on.
    """
    rows = []
    for window_id, group in features.groupby("comparable_window_id", sort=False):
        group = group.sort_values("run_order", kind="stable")
        x = group["run_order"].to_numpy(dtype=float)
        current_level = float(np.nanmedian(group["current_mean"].to_numpy(dtype=float)))
        row = {
            "comparable_window_id": int(window_id),
            "machine_id": int(group["machine_id"].iloc[0]),
            "program": group["program"].iloc[0],
            "tool": group["tool"].iloc[0],
            "num_runs": len(group),
            "scoring_only_runs": int((~group["in_reference_set"]).sum()),
            "current_level_a": current_level,
            "batch_or_job_boundary_count": int(
                group["batch_or_job_boundary"].fillna(0).sum()
                if "batch_or_job_boundary" in group.columns else 0
            ),
            "thermal_context_count": int(
                group["in_first_runs_after_long_idle"].fillna(0).sum()
                if "in_first_runs_after_long_idle" in group.columns else 0
            ),
        }
        for metric in TREND_METRICS:
            if metric not in group.columns:
                continue
            y = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 5:
                xv, yv = x[valid], y[valid]
                tau, p = kendalltau(xv, yv)
                first, last, _ = first_last_medians(yv)
                block = block_bootstrap_trend(yv, n_resamples=block_resamples, seed=0)
                row[f"{metric}_corr"] = float(pd.Series(xv).corr(pd.Series(yv)))
                row[f"{metric}_kendall_tau"] = float(tau)
                row[f"{metric}_kendall_p"] = float(p)
                row[f"{metric}_block_p"] = block["p"]
                row[f"{metric}_lag1_autocorrelation"] = block["lag1"]
                row[f"{metric}_slope"] = float(np.polyfit(xv, yv, 1)[0])
                row[f"{metric}_delta"] = float(last - first)
            else:
                for suffix in ["corr", "kendall_tau", "kendall_p", "block_p",
                               "lag1_autocorrelation", "slope", "delta"]:
                    row[f"{metric}_{suffix}"] = np.nan

        # Scale-free: express the drift as a fraction of the window's own
        # operating current, so windows on different machines are comparable.
        delta = safe_float(row.get("residual_mean_delta"), np.nan)
        # Renamed: ``build_robust_event_evidence`` publishes a column of the
        # old name computed as slope x n / level, which is a different quantity.
        row["residual_delta_fraction_first_to_last"] = (
            delta / current_level if np.isfinite(delta) and current_level > 0 else np.nan
        )
        tau = safe_float(row.get("residual_mean_kendall_tau"))
        fraction = safe_float(row.get("residual_delta_fraction_first_to_last"))
        row["positive_drift_score"] = max(tau, 0.0) * min(max(fraction, 0.0) / 0.10, 1.0)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for metric in TREND_METRICS:
        if f"{metric}_kendall_p" in out.columns:
            out[f"{metric}_kendall_q"] = benjamini_hochberg(out[f"{metric}_kendall_p"])
        if f"{metric}_block_p" in out.columns:
            out[f"{metric}_block_q"] = benjamini_hochberg(out[f"{metric}_block_p"])
    return out.sort_values("positive_drift_score", ascending=False, kind="stable")


# --------------------------------------------------------------------------- #
#  F5: clustering with a chosen k
# --------------------------------------------------------------------------- #

def build_cluster_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Inputs are already per-window robust z-scores, i.e. already comparable.

    The original code standardised them a second time across all windows,
    which reintroduced between-window scale differences that the per-window
    z-scoring existed to remove.
    """
    cols = [c for c in CLUSTER_COLS if c in features.columns]
    data = features[cols].copy()
    for col in cols:
        data[col] = data[col].fillna(0.0)
    return data.to_numpy(dtype=float), cols


def choose_k(arr: np.ndarray, k_min: int, k_max: int, seed: int) -> tuple[int, pd.DataFrame]:
    """Pick k by silhouette score instead of hard-coding it."""
    rows = []
    best_k, best_score = k_min, -np.inf
    upper = min(k_max, max(k_min, len(arr) - 1))
    for k in range(k_min, upper + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(arr)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(arr, labels))
        rows.append({"k": k, "silhouette": score, "inertia": float(model.inertia_)})
        if score > best_score:
            best_k, best_score = k, score
    return best_k, pd.DataFrame(rows)


def cluster_patterns(
    features: pd.DataFrame, k_min: int, k_max: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arr, cols = build_cluster_matrix(features)
    out = features.copy()
    if not HAS_SKLEARN or len(arr) < k_min + 1:
        out["pattern_cluster"] = 0
        return out, pd.DataFrame(), pd.DataFrame()

    best_k, selection = choose_k(arr, k_min, k_max, seed)
    model = KMeans(n_clusters=best_k, n_init=10, random_state=seed)
    out["pattern_cluster"] = model.fit_predict(arr)

    summaries = []
    for cluster_id, group in out.groupby("pattern_cluster", sort=True):
        row = {
            "pattern_cluster": int(cluster_id),
            "cluster_profile": classify_cluster_profile(group),
            "rows": len(group),
            "windows": group["comparable_window_id"].nunique(),
            "scoring_only_rows": int((~group["in_reference_set"]).sum()),
        }
        for col in cols:
            row[f"{col}_median"] = float(group[col].median())
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values("rows", ascending=False, kind="stable")
    return out, summary, selection


def classify_cluster_profile(group: pd.DataFrame) -> str:
    shape = group["dtw_shape_distance_robust_z"].median()
    amplitude = group["dtw_amplitude_distance_robust_z"].median()
    residual = group["residual_mean_robust_z"].median()
    magnitude = group["residual_abs_mean_robust_z"].median()
    if shape > 2.0 and amplitude > 2.0:
        return "shape_and_amplitude_outlier"
    if amplitude > 2.0 and abs(shape) < 1.5:
        return "amplitude_shift_pattern"
    if residual > 2.0 and magnitude > 1.0:
        return "positive_residual_pattern"
    if magnitude > 2.0:
        return "high_residual_magnitude_pattern"
    if residual < -2.0:
        return "negative_residual_pattern"
    return "baseline_like_pattern"


def phase_pattern_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_id, group in residuals.groupby("comparable_window_id", sort=False):
        bins = pd.cut(group["template_phase"], bins=np.linspace(0, 1, 11), include_lowest=True)
        phase = group.groupby(bins, observed=False).agg(
            residual_mean=("residual", "mean"),
            residual_abs_mean=("abs_residual", "mean"),
            residual_p95=("residual", lambda s: s.quantile(0.95)),
            points=("residual", "size"),
        )
        if phase.empty or not np.isfinite(phase["residual_abs_mean"]).any():
            continue
        max_abs_bin = phase["residual_abs_mean"].idxmax()
        max_pos_bin = phase["residual_mean"].idxmax()
        first = group.iloc[0]
        rows.append({
            "comparable_window_id": int(window_id),
            "machine_id": int(first["machine_id"]),
            "program": first["program"], "tool": first["tool"],
            "max_abs_residual_phase_bin": str(max_abs_bin),
            "max_abs_residual_mean": float(phase.loc[max_abs_bin, "residual_abs_mean"]),
            "max_positive_residual_phase_bin": str(max_pos_bin),
            "max_positive_residual_mean": float(phase.loc[max_pos_bin, "residual_mean"]),
            "phase_bins": len(phase),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("max_abs_residual_mean", ascending=False, kind="stable")


def rule_suggestions(
    clustered: pd.DataFrame, changepoints: pd.DataFrame, trends: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    strong_shape = clustered[
        clustered["shape_z_pos"].ge(3.0) & clustered["amplitude_z_pos"].ge(2.0)
    ]
    if not strong_shape.empty:
        rows.append({
            "candidate_rule": "shape_and_amplitude_spike",
            "event_hypothesis": "unexplained_shape_anomaly",
            "evidence": "run has high robust z-score in both shape DTW and amplitude DTW",
            "support_rows": len(strong_shape),
            "support_windows": strong_shape["comparable_window_id"].nunique(),
            "review_priority": "high",
        })

    if not changepoints.empty:
        material_like = changepoints[
            changepoints["metric"].isin(["residual_mean", "dtw_amplitude_distance", "current_integral_As"])
            & changepoints["significant_fdr05"]
            & changepoints["boundary_runs_near_split"].gt(0)
        ]
        if not material_like.empty:
            rows.append({
                "candidate_rule": "step_change_near_batch_or_job_boundary",
                "event_hypothesis": "material_change_candidate",
                "evidence": "FDR-significant changepoint near a batch/job boundary",
                "support_rows": len(material_like),
                "support_windows": material_like["comparable_window_id"].nunique(),
                "review_priority": "high",
            })

    # A6: guard on column presence instead of calling .fillna on an int default.
    if "in_first_runs_after_long_idle" in clustered.columns:
        thermal = clustered["in_first_runs_after_long_idle"].fillna(0).astype(float).gt(0)
        warmup_like = clustered[
            thermal & (clustered["residual_abs_z_pos"].ge(2.0) | clustered["amplitude_z_pos"].ge(2.0))
        ]
        if not warmup_like.empty:
            rows.append({
                "candidate_rule": "early_after_idle_high_residual",
                "event_hypothesis": "machine_warmup_or_thermal_state_candidate",
                "evidence": "post-idle runs show elevated DTW residual or amplitude distance",
                "support_rows": len(warmup_like),
                "support_windows": warmup_like["comparable_window_id"].nunique(),
                "review_priority": "medium",
            })

    if not trends.empty and "residual_mean_block_q" in trends.columns:
        toolwear_like = trends[
            trends["residual_mean_kendall_tau"].ge(0.35)
            & trends["residual_mean_block_q"].le(0.05)
            & trends["residual_delta_fraction_first_to_last"].gt(0)
            & trends["batch_or_job_boundary_count"].le(1)
            & trends["thermal_context_count"].eq(0)
        ]
        if not toolwear_like.empty:
            rows.append({
                "candidate_rule": "gradual_positive_residual_trend",
                "event_hypothesis": "toolwear_signal_candidate",
                "evidence": "FDR-significant increasing residual trend without boundary or thermal context",
                "support_rows": len(toolwear_like),
                "support_windows": toolwear_like["comparable_window_id"].nunique(),
                "review_priority": "high",
            })

    return pd.DataFrame(rows)


def plot_cluster_svg(clustered: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    width, height = 900, 620
    left, top, right, bottom = 80, 50, 30, 70
    x = clustered["dtw_shape_distance_robust_z"].to_numpy(dtype=float)
    y = clustered["residual_mean_robust_z"].to_numpy(dtype=float)
    c = clustered["pattern_cluster"].to_numpy(dtype=int)
    x_min, x_max = np.nanquantile(x, [0.02, 0.98])
    y_min, y_max = np.nanquantile(y, [0.02, 0.98])
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1

    colours = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="16" y="26" font-family="Arial" font-size="17" fill="#111">DTW Pattern Clusters</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#ddd"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#ddd"/>',
        f'<text x="{width / 2 - 80:.1f}" y="{height - 22}" font-family="Arial" font-size="13" fill="#333">shape DTW robust z</text>',
        f'<text x="12" y="{height / 2:.1f}" font-family="Arial" font-size="13" fill="#333">residual mean z</text>',
    ]
    for xi, yi, ci, run_id in zip(x, y, c, clustered["run_id"]):
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        px = left + (np.clip(xi, x_min, x_max) - x_min) / (x_max - x_min) * (width - left - right)
        py = height - bottom - (np.clip(yi, y_min, y_max) - y_min) / (y_max - y_min) * (height - top - bottom)
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colours[int(ci) % len(colours)]}">'
            f'<title>run {escape(str(run_id))}, cluster {int(ci)}</title></circle>'
        )
    parts.append("</svg>")
    (plot_dir / "cluster_scatter.svg").write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover DTW behaviour patterns and candidate event rules.")
    parser.add_argument("--dtw-dir", type=Path, default=DEFAULT_DTW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--block-resamples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, residuals = load_inputs(args.dtw_dir)
    clustered, cluster_summary, k_selection = cluster_patterns(
        features, k_min=args.k_min, k_max=args.k_max, seed=args.seed
    )
    changepoints = detect_changepoints(clustered, n_permutations=args.permutations)
    trends = detect_trends(clustered, block_resamples=args.block_resamples)
    phase_summary = phase_pattern_summary(residuals)
    suggestions = rule_suggestions(clustered, changepoints, trends)

    clustered.to_csv(args.output_dir / "run_pattern_clusters.csv", index=False)
    cluster_summary.to_csv(args.output_dir / "window_pattern_summary.csv", index=False)
    k_selection.to_csv(args.output_dir / "cluster_k_selection.csv", index=False)
    changepoints.to_csv(args.output_dir / "changepoint_candidates.csv", index=False)
    trends.to_csv(args.output_dir / "trend_pattern_candidates.csv", index=False)
    phase_summary.to_csv(args.output_dir / "phase_pattern_summary.csv", index=False)
    suggestions.to_csv(args.output_dir / "event_rule_suggestions.csv", index=False)
    plot_cluster_svg(clustered, args.output_dir)

    significant = int(changepoints["significant_fdr05"].sum()) if not changepoints.empty else 0
    chosen_k = int(clustered["pattern_cluster"].nunique())
    summary = (
        f"feature rows: {len(clustered):,}\n"
        f"clusters (k chosen by silhouette): {chosen_k:,}\n"
        f"changepoint candidates (best split per window/metric): {len(changepoints):,}\n"
        f"  FDR-significant at q<=0.05: {significant:,}\n"
        f"trend candidates: {len(trends):,}\n"
        f"rule suggestions: {len(suggestions):,}\n"
    )
    (args.output_dir / "pattern_discovery_summary.txt").write_text(summary, encoding="utf-8")
    write_manifest(args.output_dir, "discover_dtw_patterns", args, {"chosen_k": chosen_k})
    print(summary)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
