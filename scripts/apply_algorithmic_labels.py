"""Assign labels to comparable-window runs from the DTW evidence.

Fixes applied relative to the original version
----------------------------------------------
D4  **The third toolwear rule was close to a tautology.**
    ``residual_mean > 0 AND in_late_window AND tau >= 0.3`` was the pipeline's
    main label-generating mechanism.  Because all three references were
    anchored in the first ten runs, late runs had positive residuals as a
    structural consequence of the reference construction, so the rule largely
    encoded "this run is late in the window".  It is gone.  Toolwear now
    requires a cohort-relative, FDR-controlled trend from the robust evidence
    stage.

G1  Common mode is a **disqualifier**.  A window whose cohort drifts together
    cannot be labelled toolwear at all, rather than being labelled with a 20%
    smaller score.

H3  **Priority resolution was undocumented and silently discarded a label.**
    Masks were applied toolwear -> warmup -> material -> tool head, so tool
    head overwrote everything including the only material-change hit, and the
    comment ("reverse priority order so highest wins") never said what the
    priority was or why.  The order is now an explicit, documented,
    configurable constant, and ``label_conflicts`` records every label that
    fired so nothing is hidden by the choice.

    The default order runs from *externally corroborated* to *inferential*:
    thermal state and material change both have independent metadata evidence
    (a measured idle gap, a batch/job boundary), a shape anomaly has only DTW
    evidence, and toolwear is last because it is the residual explanation --
    it should only be claimed when nothing else accounts for the change.

H4  A run manifest is written next to the outputs.

A5  ``float(x or 0)`` NaN traps removed throughout.

A6  ``resolve_labels`` looped over every row with ``.loc[i]`` (an object-dtype
    Series copy per row) and created score columns by setting-with-enlargement
    inside that loop.  It is now vectorised.  ``project_to_timestamps`` also
    asserts that its merge key is unique instead of silently multiplying rows.

C2  The ``tool_head_change`` label is renamed ``unexplained_shape_anomaly``:
    windows contain a single tool, so a tool-head change cannot occur inside
    one.  ``--legacy-label-names`` restores the old vocabulary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import parse_bool, safe_float, to_numeric, write_manifest


DEFAULT_V2_CANDIDATES = Path("processed/dtw_event_candidates_v2_all_best/all_event_candidates_v2.csv")
DEFAULT_DTW_FEATURES = Path("processed/dtw_features_all_best/cnc_dtw_features.csv")
DEFAULT_TRENDS = Path("processed/robust_event_evidence_all_best/robust_trend_tests.csv")
DEFAULT_MULTI_REF = Path("processed/robust_event_evidence_all_best/multi_reference_consensus.csv")
DEFAULT_COMMON_MODE = Path("processed/robust_event_evidence_all_best/common_mode_shift_checks.csv")
DEFAULT_SEGMENTS = Path("processed/comparable_timeseries_all_best/run_tool_segments.csv")
DEFAULT_SAMPLES = Path("processed/cnc_time_series_features.csv.gz")
DEFAULT_OUTPUT_DIR = Path("processed/labeled_output")

# --- Thresholds (all reported in the summary so they can be audited) --------- #
V2_SHAPE_ANOMALY_THRESHOLD = 0.50
V2_MATERIAL_THRESHOLD = 0.40
V2_TOOLWEAR_THRESHOLD = 0.40
V2_WARMUP_THRESHOLD = 0.40
CURRENT_SHIFT_Z_THRESHOLD = 3.0
COHORT_RELATIVE_TAU_THRESHOLD = 0.15
TREND_Q_THRESHOLD = 0.10

LABELS = {
    "warmup": "machine_warmup_or_thermal_state",
    "material": "material_change_suspected",
    "anomaly": "unexplained_shape_anomaly",
    "toolwear": "toolwear_signal_suspected",
}
LEGACY_LABELS = {**LABELS, "anomaly": "tool_head_change"}

# H3: explicit priority, most externally corroborated first.
PRIORITY = ["warmup", "material", "anomaly", "toolwear"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


# --------------------------------------------------------------------------- #
#  Evidence assembly
# --------------------------------------------------------------------------- #

BOOL_COLUMNS = [
    "alarm_active", "alarm_tool_life_over", "alarm_maintenance", "is_mdi",
    "is_warmup_program", "capture_gap_flag", "long_idle_before_run",
    "in_first_runs_after_long_idle", "batch_boundary_run", "batch_or_job_boundary",
    "job_change_between_runs", "tool_head_change_sample", "multi_tool_run",
    "in_reference_set", "analysis_ready_run",
]


def build_run_evidence(dtw: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    dtw = dtw.copy()
    for col in ["comparable_window_id", "run_id", "machine_id"]:
        dtw[col] = pd.to_numeric(dtw[col], errors="coerce").astype("Int64")
    for col in ["program", "tool"]:
        dtw[col] = dtw[col].astype(str)

    if not segments.empty:
        for col in ["comparable_window_id", "run_id"]:
            segments[col] = pd.to_numeric(segments[col], errors="coerce").astype("Int64")
        segments["tool"] = segments["tool"].astype(str)
        keys = ["comparable_window_id", "run_id", "tool"]
        wanted = [
            "batch_id", "part_count", "job_event_id", "alarm_active", "alarm_tool_life_over",
            "alarm_maintenance", "is_mdi", "is_warmup_program", "capture_gap_flag",
            "batch_boundary_run", "batch_or_job_boundary", "job_change_between_runs",
            "minutes_since_previous_run_on_machine", "long_idle_before_run",
            "in_first_runs_after_long_idle", "thermal_state_score",
            "current_shift_z", "current_shift_abs_z", "tool_head_change_sample",
            "multi_tool_run", "analysis_ready_run",
        ]
        available = keys + [c for c in wanted if c in segments.columns and c not in dtw.columns]
        meta = segments[available].drop_duplicates(subset=keys)
        if meta.duplicated(subset=keys).any():
            raise ValueError("segment metadata is not unique per (window, run, tool)")
        before = len(dtw)
        dtw = dtw.merge(meta, on=keys, how="left")
        if len(dtw) != before:
            raise ValueError(f"metadata merge changed row count: {before} -> {len(dtw)}")

    for col in BOOL_COLUMNS:
        if col in dtw.columns:
            dtw[col] = parse_bool(dtw[col])
    to_numeric(dtw, ["run_order", "residual_mean", "dtw_shape_distance",
                     "dtw_amplitude_distance", "thermal_state_score",
                     "current_shift_z", "current_shift_abs_z"])

    window_sizes = dtw.groupby("comparable_window_id")["run_order"].transform("max")
    dtw["window_size"] = window_sizes
    dtw["in_late_window"] = dtw["run_order"] > (window_sizes * 0.5)
    return dtw


def attach_v2_candidates(dtw: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    """Attach the best v2 score per hypothesis to each run."""
    for key in LABELS:
        dtw[f"v2_{key}_score"] = 0.0
    dtw["v2_disqualified_reason"] = ""
    if v2.empty:
        return dtw

    v2 = v2.copy()
    for col in ["comparable_window_id", "run_id", "v2_evidence_score"]:
        v2[col] = pd.to_numeric(v2[col], errors="coerce")
    hypothesis_to_key = {
        "machine_warmup_or_thermal_state_candidate": "warmup",
        "material_change_candidate": "material",
        "unexplained_shape_anomaly": "anomaly",
        "tool_head_or_setup_change_candidate": "anomaly",
        "tool_replacement_proxy_candidate": "anomaly",
        "toolwear_signal_candidate": "toolwear",
    }

    for hypothesis, group in v2.groupby("event_hypothesis", sort=False):
        key = hypothesis_to_key.get(str(hypothesis))
        if key is None:
            continue
        column = f"v2_{key}_score"

        run_level = group[group["run_id"].notna()]
        if not run_level.empty:
            best = run_level.groupby(["comparable_window_id", "run_id"])[
                "v2_evidence_score"].max()
            index = pd.MultiIndex.from_arrays(
                [dtw["comparable_window_id"].astype("float"), dtw["run_id"].astype("float")])
            mapped = pd.Series(index.map(best), index=dtw.index).astype(float)
            dtw[column] = np.fmax(dtw[column], mapped.fillna(0.0))

        window_level = group[group["run_id"].isna()]
        if not window_level.empty:
            best = window_level.groupby("comparable_window_id")["v2_evidence_score"].max()
            mapped = dtw["comparable_window_id"].astype(float).map(best).astype(float)
            dtw[column] = np.fmax(dtw[column], mapped.fillna(0.0))

    disqualified = v2[v2["disqualified_reason"].astype(str).str.len() > 0] \
        if "disqualified_reason" in v2.columns else pd.DataFrame()
    if not disqualified.empty:
        reasons = disqualified.groupby("comparable_window_id")["disqualified_reason"].first()
        dtw["v2_disqualified_reason"] = (
            dtw["comparable_window_id"].astype(float).map(reasons).fillna(""))
    return dtw


def attach_trend_evidence(dtw: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "residual_mean_kendall_tau": "trend_kendall_tau",
        "residual_mean_kendall_q": "trend_kendall_q",
        "residual_mean_block_q": "trend_block_q",
        "residual_mean_theil_sen_slope": "trend_theil_sen_slope",
        "toolwear_evidence_score": "trend_toolwear_evidence_score",
        "cohort_relative_tau": "cohort_relative_tau",
        "cohort_median_tau": "cohort_median_tau",
        "cohort_tools": "cohort_tools",
        "drifts_with_cohort": "drifts_with_cohort",
    }
    for target in columns.values():
        dtw[target] = np.nan
    dtw["drifts_with_cohort"] = False
    if trends.empty:
        return dtw

    trends = trends.copy()
    trends["comparable_window_id"] = pd.to_numeric(
        trends["comparable_window_id"], errors="coerce")
    indexed = trends.set_index("comparable_window_id")
    window = dtw["comparable_window_id"].astype(float)
    for source, target in columns.items():
        if source not in indexed.columns:
            continue
        mapped = window.map(indexed[source])
        dtw[target] = parse_bool(mapped.fillna(False)) if target == "drifts_with_cohort" \
            else pd.to_numeric(mapped, errors="coerce")
    return dtw


def attach_common_mode(dtw: pd.DataFrame, common_mode: pd.DataFrame) -> pd.DataFrame:
    dtw["common_mode_confound"] = False
    dtw["common_mode_strength"] = np.nan
    if common_mode.empty:
        return dtw
    common_mode = common_mode.copy()
    common_mode["machine_id"] = pd.to_numeric(common_mode["machine_id"], errors="coerce")
    common_mode["program"] = common_mode["program"].astype(str)
    keyed = common_mode.set_index(["machine_id", "program"])
    index = pd.MultiIndex.from_arrays([dtw["machine_id"].astype(float), dtw["program"].astype(str)])
    dtw["common_mode_confound"] = parse_bool(
        pd.Series(index.map(keyed["common_mode_confound"]), index=dtw.index).fillna(False))
    dtw["common_mode_strength"] = pd.to_numeric(
        pd.Series(index.map(keyed["common_mode_strength"]), index=dtw.index), errors="coerce")
    return dtw


def attach_multi_ref(dtw: pd.DataFrame, multi_ref: pd.DataFrame) -> pd.DataFrame:
    dtw["multi_ref_outlier"] = False
    if multi_ref.empty or "multi_reference_outlier_evidence" not in multi_ref.columns:
        return dtw
    multi_ref = multi_ref.copy()
    for col in ["comparable_window_id", "run_id"]:
        multi_ref[col] = pd.to_numeric(multi_ref[col], errors="coerce")
    keyed = multi_ref.set_index(["comparable_window_id", "run_id"])["multi_reference_outlier_evidence"]
    index = pd.MultiIndex.from_arrays(
        [dtw["comparable_window_id"].astype(float), dtw["run_id"].astype(float)])
    dtw["multi_ref_outlier"] = parse_bool(
        pd.Series(index.map(keyed), index=dtw.index).fillna(False))
    return dtw


# --------------------------------------------------------------------------- #
#  Label resolution (vectorised)
# --------------------------------------------------------------------------- #

def resolve_labels(dtw: pd.DataFrame, label_names: dict[str, str]) -> pd.DataFrame:
    dtw = dtw.copy()
    zero = pd.Series(0.0, index=dtw.index)

    def column(name: str) -> pd.Series:
        return pd.to_numeric(dtw[name], errors="coerce").fillna(0.0) if name in dtw.columns else zero

    def flag(name: str) -> pd.Series:
        return dtw[name].fillna(False).astype(bool) if name in dtw.columns else pd.Series(False, index=dtw.index)

    # --- Warm-up / thermal: metadata context AND a measured deviation -------- #
    warmup_hit = column("v2_warmup_score").ge(V2_WARMUP_THRESHOLD)
    warmup_program = flag("is_warmup_program")
    dtw["label_warmup"] = warmup_hit | warmup_program
    dtw["score_warmup"] = np.where(
        warmup_program, 1.0, np.maximum(column("v2_warmup_score"), 0.0))

    # --- Material change: a step change with independent boundary evidence --- #
    material_metadata = flag("batch_or_job_boundary") & column("current_shift_abs_z").ge(
        CURRENT_SHIFT_Z_THRESHOLD)
    dtw["label_material"] = column("v2_material_score").ge(V2_MATERIAL_THRESHOLD) | material_metadata
    dtw["score_material"] = np.maximum(
        column("v2_material_score"), np.where(material_metadata, 0.60, 0.0))

    # --- Shape anomaly: DTW-only, explicitly not a tool-head observation ----- #
    anomaly_multi = flag("multi_ref_outlier") & column("dtw_shape_distance").gt(0)
    dtw["label_anomaly"] = column("v2_anomaly_score").ge(V2_SHAPE_ANOMALY_THRESHOLD) | anomaly_multi
    dtw["score_anomaly"] = np.maximum(
        column("v2_anomaly_score"), np.where(anomaly_multi, 0.55, 0.0))

    # --- Toolwear: cohort-relative, FDR-controlled, common-mode disqualified - #
    # D4: the old `residual_mean > 0 & in_late_window & tau >= 0.3` rule is gone.
    cohort_relative = column("cohort_relative_tau").ge(COHORT_RELATIVE_TAU_THRESHOLD)
    # F6d: the block-bootstrap q-value is the decision rule.  The naive Kendall
    # q assumes independent runs and is retained for reporting only.
    trend_q_column = "trend_block_q" if "trend_block_q" in dtw.columns else "trend_kendall_q"
    trend_significant = (
        pd.to_numeric(dtw[trend_q_column], errors="coerce").le(TREND_Q_THRESHOLD)
        if trend_q_column in dtw.columns else pd.Series(False, index=dtw.index)
    ).fillna(False)
    confounded = flag("common_mode_confound") | flag("drifts_with_cohort")

    dtw["label_toolwear"] = (
        column("v2_toolwear_score").ge(V2_TOOLWEAR_THRESHOLD)
        & cohort_relative
        & trend_significant
        & ~confounded          # G1: a disqualifier, not a discount
    )
    dtw["score_toolwear"] = np.where(
        dtw["label_toolwear"],
        np.maximum(column("v2_toolwear_score"), column("trend_toolwear_evidence_score")),
        0.0,
    )

    # --- H3: documented priority, with every firing label preserved ---------- #
    dtw["primary_label"] = "normal_or_uncertain"
    dtw["label_source"] = "default"
    dtw["label_evidence_score"] = 0.0
    for key in reversed(PRIORITY):
        mask = dtw[f"label_{key}"].fillna(False).astype(bool)
        dtw.loc[mask, "primary_label"] = label_names[key]
        dtw.loc[mask, "label_source"] = key
        dtw.loc[mask, "label_evidence_score"] = dtw.loc[mask, f"score_{key}"]

    conflict_frame = pd.DataFrame(
        {label_names[key]: dtw[f"label_{key}"].fillna(False).astype(bool) for key in PRIORITY}
    )
    dtw["labels_fired"] = conflict_frame.sum(axis=1)
    dtw["label_conflicts"] = conflict_frame.apply(
        lambda row: ";".join(sorted(conflict_frame.columns[row.to_numpy()])), axis=1)

    dtw["label_confidence"] = np.select(
        [dtw["label_evidence_score"].ge(0.75), dtw["label_evidence_score"].ge(0.50)],
        ["high", "medium"], default="low")
    dtw.loc[dtw["primary_label"].eq("normal_or_uncertain"), "label_confidence"] = "baseline"
    return dtw


# --------------------------------------------------------------------------- #
#  Timestamp projection
# --------------------------------------------------------------------------- #

def project_to_timestamps(labeled_runs: pd.DataFrame, samples_path: Path) -> pd.DataFrame:
    label_cols = ["run_id", "tool", "comparable_window_id", "primary_label",
                  "label_source", "label_confidence", "label_evidence_score", "label_conflicts"]
    lookup = labeled_runs[label_cols].copy()
    lookup["run_id"] = lookup["run_id"].astype("Int64")
    lookup["tool"] = lookup["tool"].astype(str)

    # A6: a duplicate key here would silently multiply timestamp rows.
    if lookup.duplicated(subset=["run_id", "tool"]).any():
        raise ValueError("(run_id, tool) is not unique in the label table")

    run_ids = set(lookup["run_id"].dropna().astype(int))
    read_cols = ["created_at", "machine_id", "program", "tool", "run_id",
                 "current1", "is_cutting", "sample_in_run", "seconds_in_run"]

    chunks = []
    for chunk in pd.read_csv(samples_path, usecols=lambda c: c in read_cols,
                             chunksize=200_000, low_memory=False):
        chunk["run_id"] = pd.to_numeric(chunk["run_id"], errors="coerce").astype("Int64")
        chunk = chunk[chunk["run_id"].isin(run_ids)]
        if chunk.empty:
            continue
        chunk["tool"] = chunk["tool"].astype(str)
        before = len(chunk)
        merged = chunk.merge(lookup, on=["run_id", "tool"], how="left")
        if len(merged) != before:
            raise ValueError("timestamp merge duplicated rows")
        merged["primary_label"] = merged["primary_label"].fillna("normal_or_uncertain")
        merged["label_source"] = merged["label_source"].fillna("default")
        merged["label_confidence"] = merged["label_confidence"].fillna("baseline")
        merged["label_evidence_score"] = merged["label_evidence_score"].fillna(0.0)
        merged["label_conflicts"] = merged["label_conflicts"].fillna("")
        chunks.append(merged)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def write_outputs(labeled_runs, labeled_samples, output_dir, args, label_names) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_runs.to_csv(output_dir / "cnc_run_tool_labeled.csv", index=False)
    if not labeled_samples.empty:
        labeled_samples.to_csv(output_dir / "cnc_time_series_labeled.csv.gz", index=False)

    counts = labeled_runs["primary_label"].value_counts()
    lines = [
        f"labeled run-tool rows: {len(labeled_runs):,}",
        f"labeled timestamp rows: {len(labeled_samples):,}",
        f"comparable windows: {labeled_runs['comparable_window_id'].nunique():,}",
        f"scoring-only rows (excluded from detection by the original pipeline): "
        f"{int((~labeled_runs['in_reference_set']).sum()):,}",
        "",
        "--- Label distribution (run-tool level) ---",
    ]
    for key in PRIORITY:
        label = label_names[key]
        count = int(counts.get(label, 0))
        lines.append(f"  {label}: {count:,} ({100.0 * count / max(len(labeled_runs), 1):.1f}%)")
    normal = int(counts.get("normal_or_uncertain", 0))
    lines.append(f"  normal_or_uncertain: {normal:,} "
                 f"({100.0 * normal / max(len(labeled_runs), 1):.1f}%)")

    lines += ["", "--- Multi-label overlap (before priority resolution) ---"]
    for key in PRIORITY:
        lines.append(f"  {label_names[key]}: {int(labeled_runs[f'label_{key}'].sum()):,}")
    lines.append(f"  rows with >1 label firing: {int((labeled_runs['labels_fired'] > 1).sum()):,}")

    lines += ["", "--- Confidence distribution ---"]
    for band, count in labeled_runs["label_confidence"].value_counts().items():
        lines.append(f"  {band}: {count:,}")

    lines += ["", "--- Priority order applied (most corroborated first) ---",
              "  " + " > ".join(label_names[k] for k in PRIORITY) + " > normal_or_uncertain"]
    lines += ["", "--- Thresholds ---",
              f"  v2 shape anomaly >= {V2_SHAPE_ANOMALY_THRESHOLD}",
              f"  v2 material      >= {V2_MATERIAL_THRESHOLD}",
              f"  v2 warmup        >= {V2_WARMUP_THRESHOLD}",
              f"  v2 toolwear      >= {V2_TOOLWEAR_THRESHOLD}",
              f"  cohort-relative tau >= {COHORT_RELATIVE_TAU_THRESHOLD}",
              f"  trend FDR q      <= {TREND_Q_THRESHOLD}",
              f"  current shift |z| >= {CURRENT_SHIFT_Z_THRESHOLD}",
              "",
              "NOTE: these thresholds are not calibrated against ground truth. "
              "Fill in label_review_decisions_template.csv to measure precision."]

    summary = "\n".join(lines) + "\n"
    (output_dir / "label_summary.txt").write_text(summary, encoding="utf-8")
    write_manifest(output_dir, "apply_algorithmic_labels", args, {
        "priority": PRIORITY,
        "thresholds": {
            "shape_anomaly": V2_SHAPE_ANOMALY_THRESHOLD, "material": V2_MATERIAL_THRESHOLD,
            "warmup": V2_WARMUP_THRESHOLD, "toolwear": V2_TOOLWEAR_THRESHOLD,
            "cohort_relative_tau": COHORT_RELATIVE_TAU_THRESHOLD,
            "trend_q": TREND_Q_THRESHOLD,
        },
    })
    print(summary)
    print(f"Wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign labels to comparable-window runs from DTW evidence.")
    parser.add_argument("--v2-candidates", type=Path, default=DEFAULT_V2_CANDIDATES)
    parser.add_argument("--dtw-features", type=Path, default=DEFAULT_DTW_FEATURES)
    parser.add_argument("--trends", type=Path, default=DEFAULT_TRENDS)
    parser.add_argument("--multi-ref", type=Path, default=DEFAULT_MULTI_REF)
    parser.add_argument("--common-mode", type=Path, default=DEFAULT_COMMON_MODE)
    parser.add_argument("--comparable-segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-timestamps", action="store_true")
    parser.add_argument("--legacy-label-names", action="store_true",
                        help="Use the original label vocabulary, including 'tool_head_change'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_names = LEGACY_LABELS if args.legacy_label_names else LABELS

    print("Loading inputs...")
    dtw = read_csv(args.dtw_features)
    if dtw.empty:
        raise SystemExit(f"No DTW features at {args.dtw_features}")

    print("Building run evidence table...")
    evidence = build_run_evidence(dtw, read_csv(args.comparable_segments))
    evidence = attach_v2_candidates(evidence, read_csv(args.v2_candidates))
    evidence = attach_trend_evidence(evidence, read_csv(args.trends))
    evidence = attach_common_mode(evidence, read_csv(args.common_mode))
    evidence = attach_multi_ref(evidence, read_csv(args.multi_ref))

    print("Resolving labels...")
    labeled_runs = resolve_labels(evidence, label_names)

    labeled_samples = pd.DataFrame()
    if not args.skip_timestamps:
        print("Projecting labels to timestamp level...")
        labeled_samples = project_to_timestamps(labeled_runs, args.samples)

    print("Writing outputs...")
    write_outputs(labeled_runs, labeled_samples, args.output_dir, args, label_names)


if __name__ == "__main__":
    main()
