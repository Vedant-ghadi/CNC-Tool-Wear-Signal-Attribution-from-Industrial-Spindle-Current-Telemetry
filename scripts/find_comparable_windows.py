"""Find comparable run/tool windows for DTW analysis.

Fixes applied relative to the original version
----------------------------------------------
C1  **The events were filtered out before detection began.**  ``clean_candidates``
    dropped alarm runs, MDI, warm-up programs and post-idle runs, and
    ``quality_score`` then further rewarded windows containing the fewest
    batch/job boundaries, tool changes, thermal runs and alarms.  The pipeline
    was therefore selecting for the absence of the four events it went on to
    look for, and two of them (``material_change``, ``machine_warmup``) could
    never be found: all 46 selected windows had ``alarm_count = 0`` and
    ``thermal_context_count = 0`` by construction.

    Each window now carries **two** run sets:

    * ``reference_run_ids`` -- the clean runs, used to learn what normal looks
      like (this is the old ``run_ids``, and it is still what defines the
      window's runtime homogeneity).
    * ``scoring_run_ids``   -- *every* run/tool row for this machine/program/
      tool inside the window's time span, including the alarm, post-idle,
      boundary and MDI runs that were excluded from the reference.

    Downstream stages build references from the first set and score the
    second, which is what makes the excluded events detectable at all.

C3  Windows were contiguous slices of the *filtered* sequence, so a 12-run
    window could silently straddle removed runs and hours of real time while
    every downstream test treated ``run_order`` as an evenly spaced axis.
    Windows now record ``span_hours``, ``excluded_runs_inside_window`` and
    ``max_inter_run_hours`` so the discontinuity is visible and testable.

B3  ``capture_gap_rate`` was 1.000 for every window (see ``preprocess_cnc``),
    so ``contamination_rate`` always exceeded 1.0, was clipped, and left
    ``contamination_score`` at exactly 0 for all 46 windows -- 20% of
    ``quality_score`` was a constant.  Contamination terms are now each
    normalised and averaged rather than summed-then-clipped, and the
    uninformative tool-change term has been removed (in a single-tool window
    ``tool_head_change_sample`` fires on normal program sequencing: it was true
    for a mean of 24 of 26 runs).

Also: window ids are globally unique (they previously restarted per group).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import parse_bool, to_numeric, write_manifest


DEFAULT_INPUT = Path("processed/cnc_run_tool_features.csv")
DEFAULT_OUTPUT = Path("processed/cnc_dtw_candidate_windows.csv")
DEFAULT_BEST_OUTPUT = Path("processed/cnc_dtw_best_windows.csv")

NUMERIC_COLUMNS = [
    "run_id", "machine_id", "cutting_samples", "cutting_s", "current_mean",
    "current_p95", "current_max", "current_std", "current_integral_As", "batch_id",
    "run_no_in_program", "minutes_since_previous_run_on_machine", "thermal_state_score",
    "current_shift_abs_z", "group_run_count_for_trend",
    "group_current_corr_with_run_order", "group_current_delta_fraction_first_to_last",
    "wall_clock_s", "run_time_integrity_ratio",
]

BOOL_COLUMNS = [
    "alarm_active", "alarm_tool_life_over", "alarm_maintenance", "is_mdi",
    "is_warmup_program", "capture_gap_flag", "completed_run", "truncated_start_run",
    "usable_run", "analysis_ready_run", "batch_boundary_run", "job_change_between_runs",
    "long_idle_before_run", "in_first_runs_after_long_idle", "multi_tool_run",
    "tool_head_change_sample", "batch_or_job_boundary", "part_count_missing",
]


def load_run_tools(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], utc=True, errors="coerce")
    to_numeric(df, NUMERIC_COLUMNS)
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = parse_bool(df[col])
    df["tool"] = df["tool"].astype(str)
    df["program"] = df["program"].astype(str)
    return df.sort_values(["machine_id", "program", "tool", "started_at"], kind="stable")


# --------------------------------------------------------------------------- #
#  C1: the reference set and the scoring set
# --------------------------------------------------------------------------- #

EXCLUSION_REASONS = {
    "not_analysis_ready": lambda d: ~d["analysis_ready_run"],
    "mdi": lambda d: d["is_mdi"],
    "warmup_program": lambda d: d["is_warmup_program"],
    "alarm": lambda d: d["alarm_active"],
    "post_idle_thermal": lambda d: d["in_first_runs_after_long_idle"],
    "too_few_samples": None,   # depends on min_samples, applied separately
}


def reference_candidates(df: pd.DataFrame, min_samples: int, include_warmup: bool) -> pd.DataFrame:
    """Runs clean enough to define what 'normal' looks like.

    This is deliberately strict.  It is *not* the set that gets scored -- see
    ``scoring_pool`` -- so excluding an event here no longer hides it.
    """
    mask = (
        df["analysis_ready_run"]
        & ~df["is_mdi"]
        & ~df["is_warmup_program"]
        & ~df["alarm_active"]
        & df["cutting_samples"].ge(min_samples)
        & df["cutting_s"].gt(0)
        & df["current_mean"].notna()
    )
    if not include_warmup:
        mask &= ~df["in_first_runs_after_long_idle"]
    return df[mask].copy()


def scoring_pool(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Every run/tool segment with enough signal to align at all.

    Only the physical minimum is required here: some cutting samples and a
    finite mean current.  Alarm, MDI, warm-up, post-idle and boundary runs are
    all kept, because those are precisely the events the pipeline exists to
    label.
    """
    mask = (
        df["cutting_samples"].ge(max(min_samples // 2, 5))
        & df["cutting_s"].gt(0)
        & df["current_mean"].notna()
        & ~df["part_count_missing"]
    )
    return df[mask].copy()


def exclusion_breakdown(scored: pd.DataFrame, reference_ids: set[int]) -> dict[str, int]:
    """Why each scored run is not in the reference set -- reported per window."""
    extra = scored[~scored["run_id"].isin(reference_ids)]
    if extra.empty:
        return {}
    out = {}
    for name, predicate in EXCLUSION_REASONS.items():
        if predicate is None:
            continue
        try:
            count = int(predicate(extra).fillna(False).sum())
        except KeyError:
            continue
        if count:
            out[name] = count
    return out


# --------------------------------------------------------------------------- #
#  Window scoring
# --------------------------------------------------------------------------- #

def score_window(
    window: pd.DataFrame,
    scored: pd.DataFrame,
    window_id: int,
    target_runtime_cv: float,
) -> dict[str, object]:
    cutting_s = window["cutting_s"]
    current_mean = window["current_mean"]
    median_runtime = float(cutting_s.median())
    mean_runtime = float(cutting_s.mean())
    runtime_std = float(cutting_s.std(ddof=1)) if len(window) > 1 else 0.0
    runtime_cv = runtime_std / mean_runtime if mean_runtime > 0 else np.nan
    runtime_iqr = float(cutting_s.quantile(0.75) - cutting_s.quantile(0.25))
    runtime_rel_iqr = runtime_iqr / median_runtime if median_runtime > 0 else np.nan

    current_cv = (
        float(current_mean.std(ddof=1)) / float(current_mean.mean())
        if len(window) > 1 and float(current_mean.mean()) != 0 else np.nan
    )
    current_range_fraction = (
        float(current_mean.max() - current_mean.min()) / float(current_mean.median())
        if float(current_mean.median()) != 0 else np.nan
    )

    n = len(window)
    gap_rate = float(window["capture_gap_flag"].mean())
    boundary_count = int(window["batch_or_job_boundary"].sum())
    thermal_count = int(window["in_first_runs_after_long_idle"].sum())
    alarm_count = int(window["alarm_active"].sum())

    # C3: real elapsed time, and how much of the sequence was skipped.
    start_time = window["started_at"].iloc[0]
    end_time = window["ended_at"].iloc[-1]
    span_hours = float((end_time - start_time).total_seconds() / 3600.0)
    inter_run = window["started_at"].diff().dt.total_seconds().dropna()
    max_inter_run_hours = float(inter_run.max() / 3600.0) if not inter_run.empty else 0.0

    reference_ids = set(window["run_id"].astype(int))
    in_span = scored[
        scored["started_at"].ge(start_time) & scored["started_at"].le(end_time)
    ]
    scoring_ids = list(dict.fromkeys(in_span["run_id"].astype(int).tolist()))
    excluded_inside = len(scoring_ids) - len(reference_ids & set(scoring_ids))
    reasons = exclusion_breakdown(in_span, reference_ids)

    # B3: each contamination source is normalised on its own and averaged, so
    # one saturated term can no longer zero out the whole component.  The
    # tool-change term is gone: in a single-tool window it fires on ordinary
    # program sequencing and carried no information.
    contamination_terms = [
        min(gap_rate, 1.0),
        min(boundary_count / n, 1.0),
        min(thermal_count / n, 1.0),
        min(alarm_count / n, 1.0),
    ]
    contamination_score = float(1.0 - np.mean(contamination_terms))

    runtime_score = max(0.0, 1.0 - min(runtime_cv / target_runtime_cv, 2.0) / 2.0)
    count_score = min(n / 30.0, 1.0)
    sample_score = min(float(window["cutting_samples"].median()) / 120.0, 1.0)

    quality_score = (
        0.40 * runtime_score
        + 0.25 * count_score
        + 0.20 * contamination_score
        + 0.15 * sample_score
    )

    return {
        "machine_id": int(window["machine_id"].iloc[0]),
        "program": window["program"].iloc[0],
        "tool": window["tool"].iloc[0],
        "window_id": window_id,
        "start_run_id": int(window["run_id"].iloc[0]),
        "end_run_id": int(window["run_id"].iloc[-1]),
        "start_time": start_time,
        "end_time": end_time,
        "span_hours": span_hours,
        "max_inter_run_hours": max_inter_run_hours,
        "num_runs": n,
        "num_scoring_runs": len(scoring_ids),
        "excluded_runs_inside_window": excluded_inside,
        "exclusion_reasons": ";".join(f"{k}={v}" for k, v in sorted(reasons.items())),
        "median_cutting_s": median_runtime,
        "mean_cutting_s": mean_runtime,
        "runtime_std_s": runtime_std,
        "runtime_cv": runtime_cv,
        "runtime_iqr_s": runtime_iqr,
        "runtime_rel_iqr": runtime_rel_iqr,
        "median_cutting_samples": float(window["cutting_samples"].median()),
        "current_mean_median": float(current_mean.median()),
        "current_mean_cv": current_cv,
        "current_mean_range_fraction": current_range_fraction,
        "capture_gap_rate": gap_rate,
        "batch_or_job_boundary_count": boundary_count,
        "thermal_context_count": thermal_count,
        "alarm_count": alarm_count,
        "contamination_score": contamination_score,
        "unique_batches": int(window["batch_id"].nunique(dropna=True)),
        "unique_jobs": int(window["job_event_id"].nunique(dropna=True)),
        "quality_score": quality_score,
        "reference_run_ids": ",".join(window["run_id"].astype(int).astype(str)),
        "scoring_run_ids": ",".join(str(r) for r in scoring_ids),
        # Backwards-compatible alias for any external notebook still reading it.
        "run_ids": ",".join(window["run_id"].astype(int).astype(str)),
    }


def candidate_windows_for_group(
    group: pd.DataFrame,
    scored_group: pd.DataFrame,
    min_runs: int,
    max_runs: int,
    runtime_tolerance: float,
    target_runtime_cv: float,
    next_window_id: int,
) -> tuple[list[dict[str, object]], int]:
    group = group.sort_values("started_at", kind="stable").reset_index(drop=True)
    if len(group) < min_runs:
        return [], next_window_id

    windows: list[dict[str, object]] = []
    for start in range(0, len(group) - min_runs + 1):
        for size in range(min(max_runs, len(group) - start), min_runs - 1, -1):
            window = group.iloc[start:start + size]
            median_runtime = float(window["cutting_s"].median())
            if median_runtime <= 0:
                continue
            rel_dev = (window["cutting_s"] - median_runtime).abs() / median_runtime
            if rel_dev.max() > runtime_tolerance:
                continue
            windows.append(score_window(window, scored_group, next_window_id, target_runtime_cv))
            next_window_id += 1
            break
    return windows, next_window_id


def find_windows(
    df: pd.DataFrame,
    min_runs: int,
    max_runs: int,
    min_samples: int,
    runtime_tolerance: float,
    target_runtime_cv: float,
    include_warmup: bool,
) -> pd.DataFrame:
    reference = reference_candidates(df, min_samples=min_samples, include_warmup=include_warmup)
    scored = scoring_pool(df, min_samples=min_samples)

    rows: list[dict[str, object]] = []
    next_window_id = 1
    for key, group in reference.groupby(["machine_id", "program", "tool"], sort=False):
        machine_id, program, tool = key
        scored_group = scored[
            scored["machine_id"].eq(machine_id)
            & scored["program"].eq(program)
            & scored["tool"].eq(tool)
        ]
        group_rows, next_window_id = candidate_windows_for_group(
            group, scored_group,
            min_runs=min_runs, max_runs=max_runs,
            runtime_tolerance=runtime_tolerance,
            target_runtime_cv=target_runtime_cv,
            next_window_id=next_window_id,
        )
        rows.extend(group_rows)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["quality_score", "num_runs", "runtime_cv"],
        ascending=[False, False, True], kind="stable",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find comparable run/tool windows for DTW analysis.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--best-output", type=Path, default=DEFAULT_BEST_OUTPUT)
    parser.add_argument("--min-runs", type=int, default=12)
    parser.add_argument("--max-runs", type=int, default=40)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--runtime-tolerance", type=float, default=0.10)
    parser.add_argument("--target-runtime-cv", type=float, default=0.05)
    parser.add_argument(
        "--include-warmup", action="store_true",
        help="Also allow post-idle runs into the reference set (they are scored either way).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tools = load_run_tools(args.input)
    windows = find_windows(
        run_tools,
        min_runs=args.min_runs, max_runs=args.max_runs, min_samples=args.min_samples,
        runtime_tolerance=args.runtime_tolerance, target_runtime_cv=args.target_runtime_cv,
        include_warmup=args.include_warmup,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output, index=False)
    best = (
        windows if windows.empty
        else windows.groupby(["machine_id", "program", "tool"], as_index=False, sort=False).head(1)
    )
    args.best_output.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(args.best_output, index=False)

    print(f"candidate windows: {len(windows):,}")
    print(f"best distinct windows: {len(best):,}")
    if not best.empty:
        extra = int(best["excluded_runs_inside_window"].sum())
        print(f"reference runs in best windows: {int(best['num_runs'].sum()):,}")
        print(f"scoring runs in best windows:   {int(best['num_scoring_runs'].sum()):,}")
        print(f"  of which excluded from reference (now still scored): {extra:,}")
        print(f"median capture_gap_rate: {best['capture_gap_rate'].median():.3f}")
        print(f"windows with >=1 alarm run in scoring span: "
              f"{int((best['exclusion_reasons'].str.contains('alarm', na=False)).sum())}")
        print(f"windows with >=1 post-idle run in scoring span: "
              f"{int((best['exclusion_reasons'].str.contains('post_idle', na=False)).sum())}")
        show = ["machine_id", "program", "tool", "num_runs", "num_scoring_runs",
                "excluded_runs_inside_window", "span_hours", "runtime_cv",
                "capture_gap_rate", "contamination_score", "quality_score"]
        print(best[show].head(15).round(4).to_string(index=False))

    write_manifest(args.output.parent, "find_comparable_windows", args,
                   {"candidate_windows": len(windows), "best_windows": len(best)})
    print(f"Wrote {args.output}")
    print(f"Wrote {args.best_output}")


if __name__ == "__main__":
    main()
