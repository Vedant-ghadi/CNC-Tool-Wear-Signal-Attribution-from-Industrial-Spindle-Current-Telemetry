"""Create reviewable DTW event candidate tables.

Fixes applied relative to the original version
----------------------------------------------
A4  ``norm_score(delta, 0.0, max(abs(delta) * 1.5, 0.2))`` used an upper bound
    that was a function of the value being normalised, so for any
    ``delta > 0.1333`` it returned exactly ``delta / (1.5 * delta) = 0.6667``.
    That term carried 35% of the toolwear score and was constant for 5 of the
    10 candidates in the original output.  Drift is now normalised against the
    window's own operating current, so it means "percent of load" and is
    comparable across machines.

A5  ``float(x or 0.0)`` and ``max(nan, 0.0)`` do not trap NaN -- NaN is truthy
    and NaN comparisons are all False.  Every score path now uses the NaN-safe
    helpers in ``pipeline_common``.

A6  ``load_tables`` read ``cnc_dtw_features.csv`` and never returned it.
    Removed.

C2  **A tool-head change cannot occur inside a window.**  Windows are grouped
    on ``(machine, program, tool)``, so every window contains exactly one tool
    -- the event is definitionally impossible within the unit of analysis, yet
    95 rows were labelled with it.  What the rule actually detects is "this
    run's shape and amplitude are outliers against every reference", so it is
    now named ``unexplained_shape_anomaly``.  ``--tool-head-label`` restores
    the old name for anyone reproducing the original output.

    A genuine tool-replacement proxy is also emitted separately: a *downward*
    step in current for the same tool (a fresh edge cuts more easily), which
    is more specific than a generic shape outlier.

F1/F3 Material-change candidates now require an FDR-significant changepoint
    rather than a raw effect threshold applied to every scanned split.

Warm-up candidates previously fired on the preprocessing flag alone, with no
DTW evidence required -- re-emitting an input as a "detection".  They now
require thermal context **and** a measurable deviation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import safe_float, safe_max, to_numeric, write_manifest


DEFAULT_DTW_DIR = Path("processed/dtw_features_all_best")
DEFAULT_PATTERN_DIR = Path("processed/dtw_patterns_all_best")
DEFAULT_OUTPUT_DIR = Path("processed/dtw_event_candidates_all_best")

EVENT_COLUMNS = [
    "event_hypothesis", "candidate_scope", "comparable_window_id", "machine_id",
    "program", "tool", "run_id", "run_order", "start_run_order", "end_run_order",
    "in_reference_set", "evidence_score", "confidence_band", "primary_evidence",
    "supporting_evidence", "review_recommendation",
]

# Shared confidence bands -- the original used 0.80/0.55 here and 0.75/0.50 in
# the v2 scorer, so the same number meant different things at different stages.
HIGH_BAND = 0.75
MEDIUM_BAND = 0.50


def confidence(score: float) -> str:
    if score >= HIGH_BAND:
        return "high"
    if score >= MEDIUM_BAND:
        return "medium"
    return "low"


def norm_score(value: float, low: float, high: float) -> float:
    value = safe_float(value, np.nan)
    if not np.isfinite(value) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def make_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def load_tables(pattern_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clusters = pd.read_csv(pattern_dir / "run_pattern_clusters.csv", low_memory=False)
    changepoints = pd.read_csv(pattern_dir / "changepoint_candidates.csv", low_memory=False)
    trends = pd.read_csv(pattern_dir / "trend_pattern_candidates.csv", low_memory=False)

    to_numeric(clusters, [
        "comparable_window_id", "machine_id", "run_id", "run_order",
        "dtw_shape_distance", "dtw_amplitude_distance", "residual_mean", "residual_p95",
        "residual_abs_mean", "current_integral_As", "current_mean", "batch_or_job_boundary",
        "long_idle_before_run", "in_first_runs_after_long_idle", "thermal_state_score",
        "shape_z_pos", "amplitude_z_pos", "residual_abs_z_pos", "current_shift_z",
    ])
    to_numeric(changepoints, [
        "comparable_window_id", "machine_id", "split_after_run_order",
        "first_run_after_split", "left_runs", "right_runs", "effect_size",
        "abs_effect_size", "p_permutation", "q_permutation",
        "boundary_runs_near_split", "thermal_runs_near_split",
    ])
    to_numeric(trends, [c for c in trends.columns if c not in ("program", "tool")])

    for table in (clusters, changepoints, trends):
        for col in ["program", "tool"]:
            if col in table.columns:
                table[col] = table[col].astype(str)

    if "in_reference_set" in clusters.columns:
        clusters["in_reference_set"] = (
            clusters["in_reference_set"].astype(str).str.lower().isin(["true", "1", "1.0"])
        )
    else:
        clusters["in_reference_set"] = True
    return clusters, changepoints, trends


def base_fields(row: pd.Series) -> dict[str, object]:
    return {
        "comparable_window_id": int(row["comparable_window_id"]),
        "machine_id": int(row["machine_id"]),
        "program": row["program"],
        "tool": row["tool"],
        "in_reference_set": bool(row.get("in_reference_set", True)),
    }


# --------------------------------------------------------------------------- #
#  Candidate families
# --------------------------------------------------------------------------- #

def shape_anomaly_candidates(clusters: pd.DataFrame, label: str) -> pd.DataFrame:
    if not {"shape_z_pos", "amplitude_z_pos"}.issubset(clusters.columns):
        return make_empty()

    selected = clusters[
        clusters["shape_z_pos"].ge(2.5) & clusters["amplitude_z_pos"].ge(1.5)
    ]
    rows = []
    for _, row in selected.iterrows():
        shape_z = safe_float(row["shape_z_pos"])
        amplitude_z = safe_float(row["amplitude_z_pos"])
        score = 0.55 * norm_score(shape_z, 2.5, 6.0) + 0.45 * norm_score(amplitude_z, 1.5, 5.0)
        rows.append({
            **base_fields(row),
            "event_hypothesis": label,
            "candidate_scope": "run",
            "run_id": int(row["run_id"]),
            "run_order": int(row["run_order"]),
            "start_run_order": int(row["run_order"]),
            "end_run_order": int(row["run_order"]),
            "evidence_score": score,
            "confidence_band": confidence(score),
            "primary_evidence": f"shape_z={shape_z:.2f}, amplitude_z={amplitude_z:.2f}",
            "supporting_evidence": (
                f"shape_dtw={safe_float(row.get('dtw_shape_distance'), np.nan):.4g}, "
                f"amplitude_dtw={safe_float(row.get('dtw_amplitude_distance'), np.nan):.4g}"
            ),
            "review_recommendation": (
                "Inspect the run trace and its neighbours. NOTE: the window contains a "
                "single tool, so this is an unexplained shape/amplitude outlier, not an "
                "observed tool-head change."
            ),
        })
    if not rows:
        return make_empty()
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("evidence_score", ascending=False)


def tool_replacement_proxy_candidates(clusters: pd.DataFrame) -> pd.DataFrame:
    """A downward current step for the same tool.

    More specific than a generic shape outlier: a freshly replaced edge cuts
    with less load, so a sustained drop is the observable signature of a tool
    change that the ``tool`` column cannot show (the number does not change
    when the physical cutter is swapped).
    """
    if "current_shift_z" not in clusters.columns:
        return make_empty()

    selected = clusters[clusters["current_shift_z"].le(-2.5)]
    rows = []
    for _, row in selected.iterrows():
        shift = safe_float(row["current_shift_z"], np.nan)
        if not np.isfinite(shift):
            continue
        score = 0.70 * norm_score(abs(shift), 2.5, 6.0) + 0.30 * norm_score(
            safe_float(row.get("shape_z_pos")), 1.0, 4.0)
        rows.append({
            **base_fields(row),
            "event_hypothesis": "tool_replacement_proxy_candidate",
            "candidate_scope": "run",
            "run_id": int(row["run_id"]),
            "run_order": int(row["run_order"]),
            "start_run_order": int(row["run_order"]),
            "end_run_order": int(row["run_order"]),
            "evidence_score": score,
            "confidence_band": confidence(score),
            "primary_evidence": f"downward current step, current_shift_z={shift:.2f}",
            "supporting_evidence": f"shape_z={safe_float(row.get('shape_z_pos')):.2f}",
            "review_recommendation": (
                "Check whether load drops and stays down for subsequent runs, which "
                "would indicate a fresh cutting edge rather than a transient."
            ),
        })
    if not rows:
        return make_empty()
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("evidence_score", ascending=False)


def material_change_candidates(changepoints: pd.DataFrame) -> pd.DataFrame:
    if changepoints.empty:
        return make_empty()

    metrics = ["residual_mean", "dtw_amplitude_distance", "current_integral_As"]
    significant = (
        changepoints["q_permutation"].le(0.05)
        if "q_permutation" in changepoints.columns
        else changepoints["abs_effect_size"].ge(2.5)
    )
    selected = changepoints[
        changepoints["metric"].isin(metrics)
        & significant
        & changepoints["boundary_runs_near_split"].gt(0)
    ]

    rows = []
    for _, row in selected.iterrows():
        effect = safe_float(row["effect_size"])
        q_value = safe_float(row.get("q_permutation"), 1.0)
        score = (
            0.55 * norm_score(abs(effect), 2.0, 8.0)
            + 0.25 * (1.0 - min(q_value / 0.05, 1.0))
            + 0.20 * norm_score(safe_float(row["boundary_runs_near_split"]), 1.0, 3.0)
        )
        rows.append({
            "event_hypothesis": "material_change_candidate",
            "candidate_scope": "window_split",
            "comparable_window_id": int(row["comparable_window_id"]),
            "machine_id": int(row["machine_id"]),
            "program": row["program"], "tool": row["tool"],
            "run_id": int(row["first_run_after_split"]),
            "run_order": int(row["split_after_run_order"]) + 1,
            "start_run_order": int(row["split_after_run_order"]) + 1,
            "end_run_order": np.nan,
            "in_reference_set": True,
            "evidence_score": score,
            "confidence_band": confidence(score),
            "primary_evidence": (
                f"{row['metric']} step "
                f"{'increase' if effect > 0 else 'decrease'}, "
                f"effect={effect:.2f}, q={q_value:.3g}"
            ),
            "supporting_evidence": (
                f"batch/job boundary runs near split="
                f"{int(safe_float(row['boundary_runs_near_split']))}"
            ),
            "review_recommendation": (
                "Check adjacent tools on the same machine/program at the same time for a "
                "common-mode shift, which would point to material or setup rather than tooling."
            ),
        })
    if not rows:
        return make_empty()
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("evidence_score", ascending=False)


def warmup_thermal_candidates(clusters: pd.DataFrame) -> pd.DataFrame:
    """Thermal context **plus** a measurable deviation.

    The original version emitted a candidate for every run carrying the
    ``long_idle_before_run`` or ``thermal_state_score`` flag regardless of any
    DTW evidence, so it simply re-published a preprocessing input as a
    detection.  Requiring deviation makes it a test.
    """
    required = {"in_first_runs_after_long_idle", "residual_abs_z_pos", "amplitude_z_pos"}
    if not required.issubset(clusters.columns):
        return make_empty()

    thermal_context = (
        clusters["in_first_runs_after_long_idle"].fillna(0).astype(float).gt(0)
        | clusters["thermal_state_score"].fillna(0).astype(float).gt(0.5)
    )
    deviation = (
        clusters["residual_abs_z_pos"].fillna(0).ge(1.5)
        | clusters["amplitude_z_pos"].fillna(0).ge(1.5)
    )
    selected = clusters[thermal_context & deviation]

    rows = []
    for _, row in selected.iterrows():
        deviation_z = safe_max(row.get("residual_abs_z_pos"), row.get("amplitude_z_pos"))
        thermal = safe_max(
            row.get("thermal_state_score"),
            0.5 if safe_float(row.get("long_idle_before_run")) > 0 else 0.0,
        )
        score = 0.45 * norm_score(thermal, 0.2, 1.0) + 0.55 * norm_score(deviation_z, 1.5, 5.0)
        rows.append({
            **base_fields(row),
            "event_hypothesis": "machine_warmup_or_thermal_state_candidate",
            "candidate_scope": "run",
            "run_id": int(row["run_id"]),
            "run_order": int(row["run_order"]),
            "start_run_order": int(row["run_order"]),
            "end_run_order": int(row["run_order"]),
            "evidence_score": score,
            "confidence_band": confidence(score),
            "primary_evidence": (
                f"thermal_score={safe_float(row.get('thermal_state_score')):.2f}, "
                f"deviation_z={deviation_z:.2f}"
            ),
            "supporting_evidence": (
                f"long_idle_before_run={bool(safe_float(row.get('long_idle_before_run')))}, "
                f"post_idle_run={bool(safe_float(row.get('in_first_runs_after_long_idle')))}"
            ),
            "review_recommendation": (
                "Confirm the deviation decays over the following runs; a persistent shift "
                "is not thermal."
            ),
        })
    if not rows:
        return make_empty()
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("evidence_score", ascending=False)


def toolwear_signal_candidates(trends: pd.DataFrame) -> pd.DataFrame:
    if trends.empty or "residual_mean_kendall_tau" not in trends.columns:
        return make_empty()

    # Prefer the autocorrelation-corrected q-value; fall back only if absent.
    q_column = ("residual_mean_block_q" if "residual_mean_block_q" in trends.columns
                else "residual_mean_kendall_q")
    significant = (
        trends[q_column].le(0.10) if q_column in trends.columns
        else trends["residual_mean_kendall_p"].le(0.05)
    )
    selected = trends[
        trends["residual_mean_kendall_tau"].ge(0.30)
        & significant
        & trends["residual_delta_fraction_first_to_last"].gt(0)
    ]

    rows = []
    for _, row in selected.iterrows():
        tau = safe_float(row["residual_mean_kendall_tau"])
        q_value = safe_float(row.get(q_column), 1.0)
        # A4: drift as a fraction of the window's own operating current, so the
        # term varies with the evidence instead of pinning at 0.667.
        fraction = safe_float(row.get("residual_delta_fraction_first_to_last"))
        boundary_penalty = min(safe_float(row.get("batch_or_job_boundary_count")) / 3.0, 1.0)
        thermal_penalty = min(safe_float(row.get("thermal_context_count")) / 3.0, 1.0)

        raw = (
            0.45 * norm_score(tau, 0.30, 0.80)
            + 0.30 * norm_score(fraction, 0.0, 0.10)
            + 0.25 * (1.0 - min(q_value / 0.10, 1.0))
        )
        score = raw * (1.0 - 0.35 * boundary_penalty) * (1.0 - 0.35 * thermal_penalty)
        rows.append({
            "event_hypothesis": "toolwear_signal_candidate",
            "candidate_scope": "window_trend",
            "comparable_window_id": int(row["comparable_window_id"]),
            "machine_id": int(row["machine_id"]),
            "program": row["program"], "tool": row["tool"],
            "run_id": np.nan, "run_order": np.nan,
            "start_run_order": 1,
            "end_run_order": int(safe_float(row["num_runs"])),
            "in_reference_set": True,
            "evidence_score": score,
            "confidence_band": confidence(score),
            "primary_evidence": (
                f"kendall_tau={tau:.2f}, q={q_value:.3g}, "
                f"drift={100 * fraction:.1f}% of operating current"
            ),
            "supporting_evidence": (
                f"boundary_count={int(safe_float(row.get('batch_or_job_boundary_count')))}, "
                f"thermal_count={int(safe_float(row.get('thermal_context_count')))}"
            ),
            "review_recommendation": (
                "Check the cohort: if the other tools on this machine/program drift the same "
                "way, this is common mode, not wear."
            ),
        })
    if not rows:
        return make_empty()
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("evidence_score", ascending=False)


# --------------------------------------------------------------------------- #
#  Output
# --------------------------------------------------------------------------- #

FILE_MAP = {
    "shape_anomaly": "tool_head_or_setup_change_candidates.csv",
    "tool_replacement_proxy": "tool_replacement_proxy_candidates.csv",
    "material_change": "material_change_candidates.csv",
    "machine_warmup_thermal": "machine_warmup_thermal_candidates.csv",
    "toolwear_signal": "toolwear_signal_candidates.csv",
}


def write_outputs(output_dir: Path, tables: dict[str, pd.DataFrame], args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    non_empty = [table for table in tables.values() if not table.empty]
    all_events = (
        pd.concat(non_empty, ignore_index=True).sort_values(
            "evidence_score", ascending=False, kind="stable")
        if non_empty else make_empty()
    )

    for key, filename in FILE_MAP.items():
        tables[key].to_csv(output_dir / filename, index=False)
    all_events.to_csv(output_dir / "all_event_candidates.csv", index=False)

    lines = [f"all event candidates: {len(all_events):,}"]
    lines += [f"{key}: {len(table):,}" for key, table in tables.items()]
    if not all_events.empty:
        bands = all_events["confidence_band"].value_counts()
        lines += [f"confidence {band}: {count:,}" for band, count in bands.items()]
    summary = "\n".join(lines) + "\n"
    (output_dir / "event_candidate_summary.txt").write_text(summary, encoding="utf-8")
    write_manifest(output_dir, "detect_dtw_event_candidates", args,
                   {"candidates": int(len(all_events))})
    print(summary)
    print(f"Wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reviewable DTW event candidate tables.")
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tool-head-label", default="unexplained_shape_anomaly",
        help="Name for the shape/amplitude outlier family. The original pipeline called "
             "this 'tool_head_or_setup_change_candidate', which the single-tool window "
             "design cannot support.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clusters, changepoints, trends = load_tables(args.pattern_dir)
    tables = {
        "shape_anomaly": shape_anomaly_candidates(clusters, args.tool_head_label),
        "tool_replacement_proxy": tool_replacement_proxy_candidates(clusters),
        "material_change": material_change_candidates(changepoints),
        "machine_warmup_thermal": warmup_thermal_candidates(clusters),
        "toolwear_signal": toolwear_signal_candidates(trends),
    }
    write_outputs(args.output_dir, tables, args)


if __name__ == "__main__":
    main()
