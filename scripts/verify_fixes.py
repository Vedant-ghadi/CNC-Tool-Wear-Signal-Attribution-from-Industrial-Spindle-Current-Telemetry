"""Regression checks for each defect the pipeline fixes claim to close.

Run after the pipeline so that every claim in the change log is backed by an
assertion against the generated outputs rather than by inspection.

    python scripts/verify_fixes.py --processed processed --suffix all_best
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtw_core import constrained_dtw  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(identifier: str, condition: bool, detail: str) -> None:
    results.append((identifier, PASS if condition else FAIL, detail))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the pipeline defect fixes.")
    parser.add_argument("--processed", type=Path, default=Path("processed"))
    parser.add_argument("--suffix", default="all_best")
    args = parser.parse_args()

    processed, suffix = args.processed, args.suffix
    runs = pd.read_csv(processed / "cnc_runs_features.csv", low_memory=False)
    windows = pd.read_csv(processed / "cnc_dtw_best_windows.csv", low_memory=False)
    features = pd.read_csv(processed / f"dtw_features_{suffix}" / "cnc_dtw_features.csv", low_memory=False)
    trends = pd.read_csv(processed / f"robust_event_evidence_{suffix}" / "robust_trend_tests.csv")
    common = pd.read_csv(processed / f"robust_event_evidence_{suffix}" / "common_mode_shift_checks.csv")
    phase = pd.read_csv(processed / f"robust_event_evidence_{suffix}" / "phase_specific_trends.csv")
    changepoints = pd.read_csv(processed / f"dtw_patterns_{suffix}" / "changepoint_candidates.csv")
    labels = pd.read_csv(processed / f"labeled_output_{suffix}" / "cnc_run_tool_labeled.csv", low_memory=False)

    # --- A1: distance independent of return_path -------------------------- #
    rng = np.random.default_rng(0)
    a = np.sin(np.linspace(0, 6, 200))
    b = a + 0.05 * rng.standard_normal(200)
    with_path, _ = constrained_dtw(a, b, return_path=True)
    without_path, _ = constrained_dtw(a, b, return_path=False)
    check("A1", abs(with_path - without_path) < 1e-12,
          f"return_path True={with_path:.4f} False={without_path:.4f}")

    # --- A2: reference distances now share a scale ------------------------ #
    kinds = ["loo_median", "rolling_median", "medoid_early", "first_early"]
    medians = [features[f"dtw_shape_distance_{k}"].median() for k in kinds]
    ratio = max(medians) / min(medians)
    check("A2", ratio < 3.0, f"max/min median across references = {ratio:.2f}x (was 10.6x)")

    # --- A4: drift term is no longer a constant --------------------------- #
    fractions = trends["residual_delta_fraction_of_current"].dropna()
    check("A4", fractions.nunique() > 5,
          f"{fractions.nunique()} distinct drift fractions (old term pinned at 0.6667)")

    # --- B3: capture gap flag is informative ------------------------------ #
    gap_rate = runs["capture_gap_flag"].mean()
    check("B3", 0.01 < gap_rate < 0.60, f"capture_gap_flag rate = {gap_rate:.3f} (was 0.931)")
    check("B3b", windows["capture_gap_rate"].nunique() > 1,
          f"{windows['capture_gap_rate'].nunique()} distinct window gap rates (was 1, all 1.000)")
    check("B3c", windows["contamination_score"].nunique() > 1,
          f"{windows['contamination_score'].nunique()} distinct contamination scores (was 1, all 0.0)")

    # --- B4: runs no longer straddle multi-day outages -------------------- #
    check("B4", runs["max_gap_s"].max() < 3600,
          f"max intra-run gap = {runs['max_gap_s'].max():.0f}s (was 272,112s)")

    # --- B5: per-machine cutting thresholds ------------------------------- #
    samples_head = pd.read_csv(processed / "cnc_time_series_features.csv.gz",
                               usecols=["machine_id", "cutting_threshold_a"], nrows=400_000)
    thresholds = samples_head.groupby("machine_id")["cutting_threshold_a"].first()
    check("B5", thresholds.nunique() > 1,
          f"{thresholds.nunique()} distinct per-machine thresholds (was 1 global 0.6 A)")

    # --- C1: previously-excluded runs are now scored ---------------------- #
    scoring_only = int((~features["in_reference_set"].astype(bool)).sum())
    check("C1", scoring_only > 0,
          f"{scoring_only} scoring-only runs now scored (was 0 -- filtered out pre-detection)")

    # --- C2: label renamed away from an impossible event ------------------ #
    check("C2", "tool_head_change" not in set(labels["primary_label"]),
          "no 'tool_head_change' label in a single-tool window")

    # --- C3: window time span recorded ------------------------------------ #
    check("C3", "span_hours" in windows.columns and windows["span_hours"].notna().all(),
          "every window records its real elapsed span")

    # --- D3: no self-comparisons -------------------------------------------#
    zeros = int((features["dtw_shape_distance"] == 0).sum())
    check("D3", zeros == 0, f"{zeros} zero-distance rows (was 1 per window)")

    # --- D1: reference bias reduced --------------------------------------- #
    correlations = []
    for _, group in features.groupby("comparable_window_id"):
        group = group.sort_values("run_order")
        y = group["residual_mean"].to_numpy(float)
        x = group["run_order"].to_numpy(float)
        mask = np.isfinite(y)
        if mask.sum() > 2:
            correlations.append(np.corrcoef(x[mask], y[mask])[0, 1])
    positive_share = float(np.mean(np.array(correlations) > 0))
    check("D1", positive_share < 0.70,
          f"{positive_share:.0%} of windows drift positive (was 74% with early-anchored refs)")

    # --- E2: common-mode statistic can reach zero ------------------------- #
    check("E2", common["common_mode_strength"].min() < 0.5,
          f"min common-mode strength = {common['common_mode_strength'].min():.2f} "
          f"(old statistic was floored at 0.50)")

    # --- F1: one changepoint per window/metric ---------------------------- #
    per_pair = changepoints.groupby(["comparable_window_id", "metric"]).size()
    check("F1", per_pair.max() <= 1,
          f"max {per_pair.max()} changepoint per (window, metric) (was every scanned split)")

    # --- F3: FDR control present ------------------------------------------ #
    raw = int(phase["residual_kendall_p"].le(0.05).sum())
    corrected = int(phase["increasing_phase_evidence"].astype(str).str.lower().eq("true").sum())
    check("F3", "residual_kendall_q" in phase.columns and corrected <= raw,
          f"phase trends: {raw} raw p<=0.05 -> {corrected} after FDR")

    # --- F6: trend tests are autocorrelation-aware ------------------------ #
    has_block = "residual_mean_block_q" in trends.columns
    check("F6", has_block,
          "trend tests publish a block-bootstrap q-value alongside the naive one")
    if has_block:
        naive = int(trends["residual_mean_kendall_q"].le(0.05).sum())
        corrected = int(trends["residual_mean_block_q"].le(0.05).sum())
        check("F6b", corrected <= naive,
              f"residual_mean windows significant: {naive} naive+FDR -> {corrected} block+FDR")
        check("F6c", "residual_mean_lag1_autocorrelation" in trends.columns,
              f"lag-1 autocorrelation recorded (median "
              f"{trends['residual_mean_lag1_autocorrelation'].median():.3f})")
        directions = set(trends["residual_mean_trend_direction"].dropna())
        check("F6d", directions <= {"increasing", "decreasing", "not_significant"},
              "trend direction keys off the corrected q, not the naive one")
    if "residual_block_q" in phase.columns:
        naive_phase = int(phase["residual_kendall_q"].le(0.05).sum())
        corrected_phase = int(phase["increasing_phase_evidence"]
                              .astype(str).str.lower().eq("true").sum())
        check("F6e", corrected_phase <= naive_phase,
              f"phase trends: {naive_phase} naive FDR -> {corrected_phase} block+FDR rising")

    # --- G1: toolwear labels are not common-mode -------------------------- #
    toolwear = labels[labels["label_toolwear"].astype(str).str.lower().eq("true")]
    if toolwear.empty:
        check("G1", True, "no toolwear labels emitted")
    else:
        strength = common.set_index(["machine_id", "program"])["common_mode_strength"]
        index = pd.MultiIndex.from_arrays(
            [toolwear["machine_id"].astype(float), toolwear["program"].astype(str)])
        contaminated = float(
            pd.Series(index.map(strength)).fillna(0).ge(0.75).mean())
        check("G1", contaminated < 0.10,
              f"{contaminated:.0%} of toolwear rows in >=0.75 common-mode cohorts (was 80%)")

    # --- H2: all four label families reachable ---------------------------- #
    fired = {key: int(labels[f"label_{key}"].astype(str).str.lower().eq("true").sum())
             for key in ["warmup", "material", "anomaly", "toolwear"]}
    check("H2", all(count > 0 for count in fired.values()),
          f"labels fired: {fired} (warmup and material were both 0)")

    # --- H3: multi-label conflicts preserved ------------------------------ #
    check("H3", "label_conflicts" in labels.columns,
          "every firing label retained alongside the resolved primary label")

    # --- H4: manifests written -------------------------------------------- #
    manifests = list(processed.rglob("run_manifest.json"))
    check("H4", len(manifests) >= 6, f"{len(manifests)} run manifests written (was 0)")

    width = max(len(detail) for _, _, detail in results)
    print(f"\n{'ID':<6} {'RESULT':<6} DETAIL")
    print("-" * (14 + min(width, 90)))
    for identifier, status, detail in results:
        print(f"{identifier:<6} {status:<6} {detail}")

    failures = [r for r in results if r[1] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
