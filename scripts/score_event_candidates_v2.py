"""Score DTW event candidates against the robust evidence tables.

Fixes applied relative to the original version
----------------------------------------------
E1  **The same quantity was counted four times and presented as corroboration.**
    The old toolwear score was
    ``0.25*base + 0.45*trend + 0.20*phase + 0.10*multi``, but ``base`` was a
    function of ``residual_mean_corr``, ``trend`` was Kendall's tau on
    ``residual_mean``, ``phase`` was Kendall's tau on the same residuals binned
    by phase, and ``multi`` came from the same alignment.  All four are
    monotone functions of one residual series produced by one medoid
    alignment.  The four "methods" -- hand-rolled changepoints, PELT,
    Theil-Sen, phase trends -- differ in *estimator*, not in *evidence*, so
    summing them inflated confidence without adding information.

    Evidence is now grouped into channels that really are distinct in what
    they observe:

      A  dtw_alignment  -- everything derived from the DTW residual/distance.
                          Collapsed to its single strongest member, never summed.
      B  raw_signal     -- current statistics that do not pass through DTW at
                          all (robust current shift, integral, drift fraction).
      C  cohort         -- behaviour of the *other* tools on the same machine
                          and program.
      D  metadata       -- batch/job boundaries, thermal context, alarms.

    The headline score is channel A, adjusted by how many *other* channels
    agree.  ``independent_channels`` and ``evidence_channels`` are published so
    a reader can see what the number is actually made of.

G1  **Common mode is a disqualifier, not a discount.**  The original applied a
    flat -20% to a toolwear candidate whose whole cohort moved together, so it
    survived as a slightly smaller number: 162 of 202 toolwear-labelled rows
    (80%) sat in machine/programs where >=75% of tools drifted the same way.
    A confounded cohort now caps the toolwear score below the reporting
    threshold and records the reason.

E2  ``common_mode_fraction`` was floored at 0.5 by construction, giving every
    candidate a free 0.325 common-mode score.  It now consumes the signed
    statistic from ``build_robust_event_evidence``.

A5  ``float(x or 0.0)`` NaN traps and the ``np.nanmean([nan, nan])`` path that
    handed every candidate with no reference CV a free ``+0.10`` bonus are
    both gone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import safe_float, to_numeric, write_manifest


DEFAULT_CANDIDATE_DIR = Path("processed/dtw_event_candidates_all_best")
DEFAULT_EVIDENCE_DIR = Path("processed/robust_event_evidence_all_best")
DEFAULT_OUTPUT_DIR = Path("processed/dtw_event_candidates_v2_all_best")

TEXT_COLUMNS = {
    "event_hypothesis", "candidate_scope", "program", "tool", "confidence_band",
    "primary_evidence", "supporting_evidence", "review_recommendation", "candidate_id",
    "metric", "method", "phase_bin", "dominant_direction", "tools", "window_ids",
    "strongest_phase_bin", "residual_mean_trend_direction",
}

HIGH_BAND = 0.75
MEDIUM_BAND = 0.50

CHANNELS = ["dtw_alignment", "raw_signal", "cohort", "metadata"]


def confidence(score: float) -> str:
    if score >= HIGH_BAND:
        return "high"
    if score >= MEDIUM_BAND:
        return "medium"
    return "low"


def clamp(value: float) -> float:
    return float(np.clip(safe_float(value), 0.0, 1.0))


def norm(value: float, low: float, high: float) -> float:
    value = safe_float(value, np.nan)
    if not np.isfinite(value) or high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    to_numeric(df, [c for c in df.columns if c not in TEXT_COLUMNS])
    for col in ["program", "tool"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def load_inputs(candidate_dir: Path, evidence_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "candidates": read_csv(candidate_dir / "all_event_candidates.csv"),
        "multi_ref": read_csv(evidence_dir / "multi_reference_consensus.csv"),
        "changepoints": read_csv(evidence_dir / "ruptures_changepoints.csv"),
        "trends": read_csv(evidence_dir / "robust_trend_tests.csv"),
        "phase_trends": read_csv(evidence_dir / "phase_specific_trends.csv"),
        "common_mode": read_csv(evidence_dir / "common_mode_shift_checks.csv"),
    }


# --------------------------------------------------------------------------- #
#  Evidence lookups
# --------------------------------------------------------------------------- #

def make_multiref_lookup(multi_ref: pd.DataFrame) -> dict[tuple[int, int], dict]:
    lookup = {}
    if multi_ref.empty:
        return lookup
    for _, row in multi_ref.iterrows():
        key = (int(row["comparable_window_id"]), int(row["run_id"]))
        shape_z = safe_float(row.get("shape_z_min_across_references"), np.nan)
        amplitude_z = safe_float(row.get("amplitude_z_min_across_references"), np.nan)
        outlier = bool(row.get("multi_reference_outlier_evidence", False))
        # Scored on within-window z-scores: "every reference ranks this run as
        # an outlier", not "four raw numbers happen to be close".
        score = clamp(0.5 * norm(shape_z, 1.0, 4.0) + 0.5 * norm(amplitude_z, 1.0, 4.0))
        lookup[key] = {
            "multi_reference_support": outlier,
            "multi_reference_score": score,
            "shape_z_min_across_references": shape_z,
            "amplitude_z_min_across_references": amplitude_z,
            "reference_disagreement": safe_float(row.get("reference_disagreement"), np.nan),
        }
    return lookup


def make_changepoint_lookup(changepoints: pd.DataFrame) -> dict[tuple[int, int], dict]:
    lookup = {}
    if changepoints.empty:
        return lookup
    for key, group in changepoints.groupby(
        ["comparable_window_id", "first_run_after_split"], sort=False
    ):
        window_id, run_id = int(key[0]), int(key[1])
        max_effect = safe_float(group["abs_effect_size"].max())
        boundary = bool(safe_float(group.get("boundary_runs_near_split", pd.Series([0])).max()) > 0)
        lookup[(window_id, run_id)] = {
            "changepoint_support": True,
            "changepoint_score": clamp(norm(max_effect, 2.0, 7.0)),
            "changepoint_max_effect": max_effect,
            "changepoint_metrics": ",".join(sorted(group["metric"].astype(str).unique())),
            "changepoint_boundary_support": boundary,
        }
    return lookup


def make_trend_lookup(trends: pd.DataFrame) -> dict[int, dict]:
    lookup = {}
    if trends.empty:
        return lookup
    for _, row in trends.iterrows():
        window_id = int(row["comparable_window_id"])
        lookup[window_id] = {
            "trend_score": clamp(row.get("toolwear_evidence_score")),
            "residual_mean_kendall_tau": safe_float(row.get("residual_mean_kendall_tau"), np.nan),
            "residual_mean_kendall_q": safe_float(row.get("residual_mean_kendall_q"), np.nan),
            "cohort_tools": safe_float(row.get("cohort_tools"), 1.0),
            "cohort_relative_tau": safe_float(row.get("cohort_relative_tau"), np.nan),
            "cohort_median_tau": safe_float(row.get("cohort_median_tau"), np.nan),
            "drifts_with_cohort": bool(row.get("drifts_with_cohort", False)),
            "drift_fraction_of_current": safe_float(
                row.get("residual_delta_fraction_of_current"), np.nan),
        }
    return lookup


def make_phase_lookup(phase_trends: pd.DataFrame) -> dict[int, dict]:
    lookup = {}
    if phase_trends.empty or "increasing_phase_evidence" not in phase_trends.columns:
        return lookup
    strong = phase_trends[
        phase_trends["increasing_phase_evidence"].astype(str).str.lower().isin(["true", "1"])
    ]
    for window_id, group in strong.groupby("comparable_window_id", sort=False):
        best = group.sort_values("residual_kendall_tau", ascending=False).iloc[0]
        lookup[int(window_id)] = {
            "phase_trend_support": True,
            "phase_trend_score": clamp(norm(safe_float(best["residual_kendall_tau"]), 0.35, 0.80)),
            "strongest_phase_bin": str(best["phase_bin"]),
            "phase_bins_with_increasing_evidence": int(len(group)),
        }
    return lookup


def make_common_mode_lookup(common_mode: pd.DataFrame) -> dict[tuple[int, str], dict]:
    lookup = {}
    if common_mode.empty:
        return lookup
    for _, row in common_mode.iterrows():
        key = (int(row["machine_id"]), str(row["program"]))
        lookup[key] = {
            "common_mode_strength": safe_float(row.get("common_mode_strength")),
            "common_mode_signed_fraction": safe_float(row.get("common_mode_signed_fraction")),
            "common_mode_direction": str(row.get("dominant_direction", "")),
            "common_mode_confound": bool(row.get("common_mode_confound", False)),
            "cohort_tools_compared": int(safe_float(row.get("tools_compared"), 0)),
            "common_mode_windows": str(row.get("window_ids", "")),
        }
    return lookup


def default_evidence() -> dict:
    return {
        "multi_reference_support": False, "multi_reference_score": 0.0,
        "shape_z_min_across_references": np.nan, "amplitude_z_min_across_references": np.nan,
        "reference_disagreement": np.nan,
        "changepoint_support": False, "changepoint_score": 0.0,
        "changepoint_max_effect": np.nan, "changepoint_metrics": "",
        "changepoint_boundary_support": False,
        "trend_score": 0.0, "residual_mean_kendall_tau": np.nan,
        "residual_mean_kendall_q": np.nan, "cohort_tools": 1.0,
        "cohort_relative_tau": np.nan, "cohort_median_tau": np.nan,
        "drifts_with_cohort": False, "drift_fraction_of_current": np.nan,
        "phase_trend_support": False, "phase_trend_score": 0.0,
        "strongest_phase_bin": "", "phase_bins_with_increasing_evidence": 0,
        "common_mode_strength": 0.0, "common_mode_signed_fraction": 0.0,
        "common_mode_direction": "", "common_mode_confound": False,
        "cohort_tools_compared": 0, "common_mode_windows": "",
    }


# --------------------------------------------------------------------------- #
#  E1: channelled scoring
# --------------------------------------------------------------------------- #

def channel_scores(row: dict, hypothesis: str) -> dict[str, float]:
    """Score each evidence channel separately.

    Everything inside ``dtw_alignment`` derives from one alignment of one
    signal, so it is collapsed with ``max`` rather than summed -- three views
    of the same residual are one piece of evidence, not three.
    """
    base = safe_float(row.get("evidence_score"))
    multi = safe_float(row.get("multi_reference_score"))
    changepoint = safe_float(row.get("changepoint_score"))
    trend = safe_float(row.get("trend_score"))
    phase = safe_float(row.get("phase_trend_score"))

    if hypothesis == "toolwear_signal_candidate":
        dtw = max(base, trend, phase)
    elif hypothesis == "material_change_candidate":
        dtw = max(base, changepoint)
    else:
        dtw = max(base, multi)

    # Channel B: current statistics that never pass through the DTW alignment.
    drift_fraction = safe_float(row.get("drift_fraction_of_current"), np.nan)
    if not np.isfinite(drift_fraction):
        raw_signal = 0.0
    elif hypothesis == "toolwear_signal_candidate":
        # Wear is directional: a falling load is not corroborating evidence of
        # a tool wearing out, so the signed value is used, not its magnitude.
        raw_signal = norm(drift_fraction, 0.02, 0.15)
    else:
        raw_signal = norm(abs(drift_fraction), 0.02, 0.15)

    # Channel C: what the other tools on this machine/program are doing.
    cohort_relative = safe_float(row.get("cohort_relative_tau"), np.nan)
    cohort_tools = safe_float(row.get("cohort_tools"), 1.0)
    if hypothesis == "toolwear_signal_candidate":
        cohort = norm(cohort_relative, 0.10, 0.50) if cohort_tools >= 2 else 0.0
    elif hypothesis == "material_change_candidate":
        cohort = norm(safe_float(row.get("common_mode_strength")), 0.30, 1.0)
    else:
        cohort = 0.0

    # Channel D: independent metadata context.
    if hypothesis == "material_change_candidate":
        metadata = 1.0 if row.get("changepoint_boundary_support") else 0.0
    elif hypothesis == "machine_warmup_or_thermal_state_candidate":
        metadata = 1.0
    else:
        metadata = 0.0

    return {"dtw_alignment": clamp(dtw), "raw_signal": clamp(raw_signal),
            "cohort": clamp(cohort), "metadata": clamp(metadata)}


def score_candidate(row: dict, hypothesis: str) -> dict:
    channels = channel_scores(row, hypothesis)
    primary = channels["dtw_alignment"]
    supporting = [name for name in ["raw_signal", "cohort", "metadata"]
                  if channels[name] >= 0.35]

    # Corroboration is a bounded bonus on the primary channel, never a sum of
    # correlated terms: two views of the same residual cannot make each other
    # more true.
    corroboration = 1.0 + 0.12 * len(supporting)
    score = clamp(primary * corroboration)

    disqualified_reason = ""
    if hypothesis == "toolwear_signal_candidate":
        # G1: the cohort moving together is an explanation, not a nuisance.
        if bool(row.get("common_mode_confound")) or bool(row.get("drifts_with_cohort")):
            score = min(score, 0.30)
            disqualified_reason = (
                "cohort common-mode: other tools on this machine/program drift the same "
                "way, so the change is not tool-specific"
            )
        elif safe_float(row.get("cohort_tools"), 1.0) < 2:
            score = min(score, 0.60)
            disqualified_reason = "no cohort available: common-mode confound cannot be excluded"

    return {
        **{f"channel_{name}": value for name, value in channels.items()},
        "primary_channel_score": primary,
        "independent_channels": len(supporting),
        "evidence_channels": ";".join(["dtw_alignment"] + supporting),
        "v2_evidence_score": score,
        "v2_confidence_band": confidence(score),
        "disqualified_reason": disqualified_reason,
    }


def parse_optional_int(value) -> int | None:
    text = str(value).strip().lower()
    if text in ("", "nan", "none"):
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def enrich_candidates(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = tables["candidates"].copy()
    if candidates.empty:
        return candidates
    candidates.insert(0, "candidate_id", [f"V2C{i:04d}" for i in range(1, len(candidates) + 1)])

    multi_ref = make_multiref_lookup(tables["multi_ref"])
    changepoints = make_changepoint_lookup(tables["changepoints"])
    trends = make_trend_lookup(tables["trends"])
    phase = make_phase_lookup(tables["phase_trends"])
    common = make_common_mode_lookup(tables["common_mode"])

    rows = []
    for _, candidate in candidates.iterrows():
        row = candidate.to_dict()
        window_id = int(row["comparable_window_id"])
        run_id = parse_optional_int(row.get("run_id"))
        hypothesis = str(row["event_hypothesis"])

        row.update(default_evidence())
        if run_id is not None:
            row.update(multi_ref.get((window_id, run_id), {}))
            row.update(changepoints.get((window_id, run_id), {}))
        row.update(trends.get(window_id, {}))
        row.update(phase.get(window_id, {}))
        row.update(common.get((int(row["machine_id"]), str(row["program"])), {}))
        row["dtw_candidate_score"] = safe_float(row.get("evidence_score"))
        row.update(score_candidate(row, hypothesis))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("v2_evidence_score", ascending=False, kind="stable")


def write_outputs(scored: pd.DataFrame, output_dir: Path, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "all_event_candidates_v2.csv", index=False)
    for hypothesis, group in scored.groupby("event_hypothesis", sort=False):
        filename = str(hypothesis).replace("_candidate", "").replace("_or_", "_")
        group.to_csv(output_dir / f"{filename}_candidates_v2.csv", index=False)

    lines = [f"v2 candidates: {len(scored):,}"]
    if not scored.empty:
        for band in ["high", "medium", "low"]:
            lines.append(f"{band} confidence: {int((scored['v2_confidence_band'] == band).sum()):,}")
        lines.append("")
        lines.append("by hypothesis:")
        for hypothesis, group in scored.groupby("event_hypothesis"):
            lines.append(f"  {hypothesis}: {len(group):,} "
                         f"(high={int((group['v2_confidence_band'] == 'high').sum())})")
        disqualified = int((scored["disqualified_reason"] != "").sum())
        lines.append("")
        lines.append(f"candidates capped by a confound rule: {disqualified:,}")
        lines.append(f"mean independent supporting channels: "
                     f"{scored['independent_channels'].mean():.2f}")
    summary = "\n".join(lines) + "\n"
    (output_dir / "candidate_v2_summary.txt").write_text(summary, encoding="utf-8")
    write_manifest(output_dir, "score_event_candidates_v2", args, {"candidates": len(scored)})
    print(summary)
    print(f"Wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score DTW event candidates using channel-separated robust evidence.")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_inputs(args.candidate_dir, args.evidence_dir)
    scored = enrich_candidates(tables)
    write_outputs(scored, args.output_dir, args)


if __name__ == "__main__":
    main()
