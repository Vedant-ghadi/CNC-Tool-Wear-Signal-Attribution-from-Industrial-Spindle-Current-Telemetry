"""Build feature-only CNC time-series, run-level and run/tool-level datasets.

Fixes applied relative to the original version
----------------------------------------------
B3  Capture gaps were computed within ``(machine_id, program)``, so every
    return to a program after running a different one looked like a telemetry
    outage.  That flagged 93% of all runs and 100% of every selected DTW
    window, which in turn pinned the window scorer's contamination term at a
    constant.  Gaps are now measured on the machine timeline; program
    transitions are recorded separately as ``program_change_sample``.
    (On this dataset 99.85% of machine-timeline gaps are <= 5 s.)

B4  ``elapsed_s`` summed ``dt_s`` with every gap capped at 1 s, so it was not
    wall-clock time, and runs were split on ``part_count`` alone -- letting a
    single "run" span days.  Runs are now also split on program change and on
    any gap longer than ``--split-run-gap-s``; both ``elapsed_s`` (active
    seconds) and ``wall_clock_s`` (true span) are emitted, along with
    ``run_time_integrity_ratio`` so the discrepancy is visible.

B5  ``is_cutting`` used one global 0.6 A threshold across machines whose
    current levels differ by an order of magnitude (machine 294 medians
    2.4 A, machine 283 medians 23.8 A).  The threshold is now derived per
    machine by Otsu's method on the current distribution, with the global
    value available via ``--cutting-threshold``.

A7  ``part_count`` was ``fillna(-1)``, which would silently merge a block of
    missing part counts into one run and fabricate a batch boundary on either
    side.  It stays nullable and such runs are excluded from ``usable_run``.
    ``tool`` was stringified straight from a float column, producing '6.0';
    it is now normalised so an integer-typed and float-typed source file
    cannot split one physical tool into two identities.

A12 First/last comparison blocks used ``max(3, n // 10)`` at both ends, so for
    n < 6 they overlapped and the delta was biased toward zero.  Now uses the
    non-overlapping helper in ``pipeline_common``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (
    first_last_medians,
    robust_scale,
    write_manifest,
)


RAW_COLUMNS = [
    "created_at",
    "machine_id",
    "device_id",
    "current1",
    "current2",
    "current3",
    "energy",
    "program",
    "tool",
    "feed_override",
    "speed_override",
    "part_count",
    "alarms",
    "machine_data_at",
    "job_event_id",
    "job_eventable_type",
    "job_eventable_id",
    "job_started_at",
]

NO_TOOL = "-"
NO_PROGRAM = "(none)"
DEFAULT_RUNNING_A = 0.6
MAX_GAP_S = 5.0
NOMINAL_S = 1.0
DEFAULT_SPLIT_RUN_GAP_S = 900.0
ROLLING_WINDOW_RUNS = 10
LONG_IDLE_MINUTES = 60.0
WARMUP_WINDOW_RUNS = 5
MIN_CUTTING_THRESHOLD = 0.15


# --------------------------------------------------------------------------- #
#  Loading and normalisation
# --------------------------------------------------------------------------- #

def _read_one(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, low_memory=False)
    raw["source_file"] = path.name
    return raw


def load_raw(data_dir: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {data_dir}")
    frames = [_read_one(path) for path in paths]
    return normalize(pd.concat(frames, ignore_index=True))


def clean_text(series: pd.Series, fill_value: str) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace("", fill_value)


def normalize_tool(series: pd.Series) -> pd.Series:
    """Canonical tool identity.

    A7: tool numbers arrive as ints in some files and floats in others.  A bare
    ``astype(str)`` therefore yields both '6' and '6.0' for the same physical
    tool, which silently splits every downstream group.  Numeric values are
    routed through a nullable integer so they render identically; genuinely
    non-numeric labels are preserved as trimmed text.
    """
    text = series.fillna("").astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    is_integral = numeric.notna() & np.isclose(numeric % 1, 0, equal_nan=False)

    out = text.copy()
    out.loc[is_integral] = numeric.loc[is_integral].astype("Int64").astype(str)
    # Non-integral numerics keep a stable, trimmed decimal form.
    fractional = numeric.notna() & ~is_integral
    out.loc[fractional] = numeric.loc[fractional].map(lambda v: f"{v:g}")
    return out.replace("", NO_TOOL)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["machine_data_at"] = pd.to_datetime(df["machine_data_at"], utc=True, errors="coerce")
    df["job_started_at"] = pd.to_datetime(df["job_started_at"], utc=True, errors="coerce")

    df["machine_id"] = pd.to_numeric(df["machine_id"], errors="coerce").astype("Int64")
    df["job_event_id"] = pd.to_numeric(df["job_event_id"], errors="coerce").astype("Int64")
    df["job_eventable_id"] = pd.to_numeric(df["job_eventable_id"], errors="coerce").astype("Int64")

    # A7: stays nullable -- no -1 sentinel that can masquerade as a real count.
    df["part_count"] = pd.to_numeric(df["part_count"], errors="coerce").astype("Int64")
    df["part_count_missing"] = df["part_count"].isna()

    for col in ["current1", "current2", "current3", "energy", "feed_override", "speed_override"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["program"] = clean_text(df["program"], NO_PROGRAM)
    df["tool"] = normalize_tool(df["tool"])
    df["alarms"] = df["alarms"].fillna("{}").astype(str)
    df["job_eventable_type"] = clean_text(df["job_eventable_type"], "")
    df["device_id"] = clean_text(df["device_id"], "")

    df = df.dropna(subset=["created_at", "machine_id"]).copy()
    df["machine_id"] = df["machine_id"].astype("int64")
    return df


# --------------------------------------------------------------------------- #
#  B5: per-machine cutting threshold
# --------------------------------------------------------------------------- #

def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's between-class variance threshold on the given scale."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 50:
        return np.nan
    lo, hi = float(np.nanmin(values)), float(np.nanquantile(values, 0.999))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan

    counts, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    weights = counts.astype(float) / counts.sum()
    centres = (edges[:-1] + edges[1:]) / 2.0

    w0 = np.cumsum(weights)
    w1 = 1.0 - w0
    m0 = np.cumsum(weights * centres)
    total_mean = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / w0
        mu1 = (total_mean - m0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -np.inf
    if not np.isfinite(between).any():
        return np.nan
    return float(centres[int(np.argmax(between))])


def machine_cutting_threshold(values: np.ndarray) -> float:
    """Idle-vs-cutting boundary for one machine.

    Otsu is applied in the **log** domain.  The idle cluster sits near the
    sensor noise floor (~0.1 A) and the cutting cluster is machine-specific
    (2.4 A on machine 294, 23.8 A on machine 283), so the two modes are
    separated by more than an order of magnitude.  On a linear scale the
    between-class variance is dominated by the spread *within* the cutting
    cluster and Otsu returns a threshold part-way up it -- for machine 294
    that gave 7.1 A against a cutting level of 2.4 A, which would have
    discarded 99% of its cutting samples.  In log space the two modes are
    comparable in width and the split lands in the trough between them.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 50:
        return DEFAULT_RUNNING_A

    log_threshold = otsu_threshold(np.log10(values))
    if not np.isfinite(log_threshold):
        return DEFAULT_RUNNING_A
    threshold = float(10.0 ** log_threshold)

    # Sanity rails: the boundary must sit above the noise floor and below the
    # bulk of the cutting distribution, or fall back to the legacy constant.
    floor = float(np.quantile(values, 0.01))
    ceiling = float(np.median(values[values >= threshold])) if (values >= threshold).any() else np.inf
    if not (floor < threshold < ceiling):
        return DEFAULT_RUNNING_A
    return float(max(threshold, MIN_CUTTING_THRESHOLD))


def machine_cutting_thresholds(df: pd.DataFrame, override: float | None) -> dict[int, float]:
    """One cutting threshold per machine, with a documented fallback."""
    thresholds: dict[int, float] = {}
    for machine_id, group in df.groupby("machine_id", sort=True):
        if override is not None:
            thresholds[int(machine_id)] = float(override)
        else:
            thresholds[int(machine_id)] = machine_cutting_threshold(
                group["current1"].to_numpy(dtype=float)
            )
    return thresholds


# --------------------------------------------------------------------------- #
#  Sample-level features
# --------------------------------------------------------------------------- #

def add_time_series_flags(
    df: pd.DataFrame,
    cutting_thresholds: dict[int, float],
    split_run_gap_s: float,
) -> pd.DataFrame:
    # B3/B4: the machine timeline is the physical timeline.  Ordering and gap
    # measurement both happen there; program identity is a per-sample attribute,
    # not a partition of time.
    df = df.sort_values(["machine_id", "created_at"], kind="stable").reset_index(drop=True)
    by_machine = df.groupby("machine_id", sort=False)

    forward_s = (by_machine["created_at"].shift(-1) - df["created_at"]).dt.total_seconds()
    df["dt_s"] = forward_s.where(forward_s.le(MAX_GAP_S) & forward_s.gt(0), NOMINAL_S).fillna(NOMINAL_S)

    df["gap_s"] = (df["created_at"] - by_machine["created_at"].shift()).dt.total_seconds().fillna(0.0)
    df["capture_gap_flag"] = df["gap_s"].gt(MAX_GAP_S)

    prev_program = by_machine["program"].shift()
    df["program_change_sample"] = df["program"].ne(prev_program) & prev_program.notna()

    prev_part = by_machine["part_count"].shift()
    df["batch_boundary_sample"] = (df["part_count"] < prev_part).fillna(False)
    df["batch_id"] = (
        df["batch_boundary_sample"].groupby([df["machine_id"], df["program"]], sort=False).cumsum()
    ).astype("int64")

    # B4: a run ends at a part-count change, a program change, or a telemetry
    # outage long enough that the following samples are a different activity.
    new_run = (
        df["part_count"].ne(prev_part).fillna(True)
        | prev_part.isna()
        | df["program_change_sample"]
        | df["gap_s"].gt(split_run_gap_s)
        | by_machine.cumcount().eq(0)
    )
    df["run_id"] = new_run.cumsum().astype("int64")
    df["run_split_by_gap"] = df["gap_s"].gt(split_run_gap_s)
    df["sample_in_run"] = df.groupby("run_id", sort=False).cumcount() + 1
    df["seconds_in_run"] = df.groupby("run_id", sort=False)["dt_s"].cumsum() - df["dt_s"]

    # B3: the gap carried by a run's *first* sample is the idle time since the
    # previous run -- normal between-part behaviour, not a telemetry outage.
    # Only gaps strictly inside a run indicate missing capture.
    df["intra_run_gap_s"] = df["gap_s"].where(df["sample_in_run"].gt(1), 0.0)
    df["intra_run_capture_gap"] = df["intra_run_gap_s"].gt(MAX_GAP_S)

    # B5: threshold per machine rather than one global constant.
    threshold_series = df["machine_id"].map(cutting_thresholds).astype(float)
    df["cutting_threshold_a"] = threshold_series
    df["is_cutting"] = df["current1"].ge(threshold_series)

    df["alarm_active"] = df["alarms"].ne("{}")
    df["alarm_tool_life_over"] = df["alarms"].str.contains("TOOL LIFE OVER", regex=False)
    df["alarm_maintenance"] = df["alarms"].str.contains("MAINTENANCE CHECK WARNING", regex=False)
    df["is_mdi"] = df["program"].str.upper().eq("MDI")
    df["is_warmup_program"] = df["program"].str.upper().str.contains("WARM", regex=False)

    df["prev_tool"] = by_machine["tool"].shift()
    df["tool_head_change_sample"] = (
        df["tool"].ne(df["prev_tool"])
        & df["prev_tool"].notna()
        & df["tool"].ne(NO_TOOL)
        & df["prev_tool"].ne(NO_TOOL)
    )

    df["prev_job_event_id"] = by_machine["job_event_id"].shift()
    df["job_change_sample"] = (
        df["job_event_id"].ne(df["prev_job_event_id"])
        & df["job_event_id"].notna()
        & df["prev_job_event_id"].notna()
    )
    return df


# --------------------------------------------------------------------------- #
#  Run level
# --------------------------------------------------------------------------- #

def unique_join(values: pd.Series) -> str:
    seen: list[str] = []
    for value in values.astype(str):
        if value not in seen:
            seen.append(value)
    return ",".join(seen)


def build_runs(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("run_id", sort=True)
    runs = pd.DataFrame(
        {
            "machine_id": grouped["machine_id"].first(),
            "program": grouped["program"].first(),
            "batch_id": grouped["batch_id"].first(),
            "part_count": grouped["part_count"].first(),
            "started_at": grouped["created_at"].min(),
            "ended_at": grouped["created_at"].max(),
            "samples": grouped.size(),
            "elapsed_s": grouped["dt_s"].sum(),
            "max_gap_s": grouped["intra_run_gap_s"].max(),
            "idle_gap_before_run_s": grouped["gap_s"].first(),
            "tool_count": grouped["tool"].nunique(),
            "first_tool": grouped["tool"].first(),
            "last_tool": grouped["tool"].last(),
            "tool_list": grouped["tool"].apply(unique_join),
            "job_event_id": grouped["job_event_id"].first(),
            "job_eventable_type": grouped["job_eventable_type"].first(),
            "job_eventable_id": grouped["job_eventable_id"].first(),
            "alarm_active": grouped["alarm_active"].any(),
            "alarm_tool_life_over": grouped["alarm_tool_life_over"].any(),
            "alarm_maintenance": grouped["alarm_maintenance"].any(),
            "is_mdi": grouped["is_mdi"].any(),
            "is_warmup_program": grouped["is_warmup_program"].any(),
            "capture_gap_flag": grouped["intra_run_capture_gap"].any(),
            "part_count_missing": grouped["part_count_missing"].any(),
            "tool_head_change_inside_run": grouped["tool_head_change_sample"].any(),
            "job_change_inside_run": grouped["job_change_sample"].any(),
        }
    )

    # B4: true span alongside active seconds, so the difference is auditable.
    runs["wall_clock_s"] = (runs["ended_at"] - runs["started_at"]).dt.total_seconds()
    runs["run_time_integrity_ratio"] = np.where(
        runs["elapsed_s"].gt(0), runs["wall_clock_s"] / runs["elapsed_s"], np.nan
    )

    cutting = df[df["is_cutting"]].groupby("run_id", sort=True)
    current_features = pd.DataFrame(
        {
            "cutting_s": cutting["dt_s"].sum(),
            "cutting_samples": cutting.size(),
            "current_mean": cutting["current1"].mean(),
            "current_median": cutting["current1"].median(),
            "current_p95": cutting["current1"].quantile(0.95),
            "current_max": cutting["current1"].max(),
            "current_std": cutting["current1"].std(ddof=1),
            "current_integral_As": cutting.apply(
                lambda x: float((x["current1"] * x["dt_s"]).sum()), include_groups=False
            ),
            "energy_mean": cutting["energy"].mean(),
            "feed_override_mean": cutting["feed_override"].mean(),
        }
    )
    runs = runs.join(current_features)
    runs["cutting_s"] = runs["cutting_s"].fillna(0.0)
    runs["cutting_samples"] = runs["cutting_samples"].fillna(0).astype("int64")
    runs["cutting_fraction"] = np.where(
        runs["elapsed_s"].gt(0), runs["cutting_s"] / runs["elapsed_s"], 0.0
    )

    runs = runs.sort_values(["machine_id", "program", "started_at"], kind="stable")
    in_batch = runs.groupby(["machine_id", "program", "batch_id"], sort=False)
    runs["completed_run"] = in_batch.cumcount(ascending=False).ne(0)
    runs["truncated_start_run"] = runs.groupby(["machine_id", "program"], sort=False).cumcount().eq(0)

    # A7: a run built from rows with no part count cannot be trusted as a cycle.
    runs["usable_run"] = (
        runs["completed_run"] & ~runs["truncated_start_run"] & ~runs["part_count_missing"]
    )
    runs["analysis_ready_run"] = runs["usable_run"] & runs["cutting_samples"].gt(0)
    runs["run_no_in_program"] = runs.groupby(["machine_id", "program"], sort=False).cumcount() + 1

    previous = runs.groupby(["machine_id", "program"], sort=False)
    runs["prev_run_last_tool"] = previous["last_tool"].shift()
    runs["tool_head_change_between_runs"] = (
        runs["first_tool"].ne(runs["prev_run_last_tool"])
        & runs["first_tool"].ne(NO_TOOL)
        & runs["prev_run_last_tool"].notna()
        & runs["prev_run_last_tool"].ne(NO_TOOL)
    )
    runs["multi_tool_run"] = runs["tool_count"].gt(1)

    runs["prev_job_event_id"] = previous["job_event_id"].shift()
    runs["job_change_between_runs"] = (
        runs["job_event_id"].ne(runs["prev_job_event_id"])
        & runs["job_event_id"].notna()
        & runs["prev_job_event_id"].notna()
    )
    runs["batch_boundary_run"] = (
        runs.groupby(["machine_id", "program"], sort=False)["batch_id"].diff().fillna(0).ne(0)
    )
    runs["tool_changed_in_or_before_run"] = (
        runs["tool_head_change_between_runs"] | runs["tool_head_change_inside_run"]
    )
    return runs


def add_thermal_context_features(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy().sort_values(["machine_id", "started_at"], kind="stable")
    by_machine = runs.groupby("machine_id", sort=False)

    previous_end = by_machine["ended_at"].shift()
    idle_min = (runs["started_at"] - previous_end).dt.total_seconds() / 60.0
    runs["minutes_since_previous_run_on_machine"] = idle_min
    runs["long_idle_before_run"] = idle_min.ge(LONG_IDLE_MINUTES).fillna(True)

    new_thermal_window = runs["long_idle_before_run"] | runs["is_warmup_program"]
    runs["thermal_window_id"] = (
        new_thermal_window.groupby(runs["machine_id"], sort=False).cumsum().astype("int64")
    )
    runs["warmup_run_index_after_idle"] = (
        runs.groupby(["machine_id", "thermal_window_id"], sort=False).cumcount() + 1
    )

    in_warmup_window = runs["warmup_run_index_after_idle"].le(WARMUP_WINDOW_RUNS)
    runs["in_first_runs_after_long_idle"] = runs["thermal_window_id"].gt(0) & in_warmup_window
    runs["thermal_state_score"] = np.where(
        runs["is_warmup_program"] | runs["in_first_runs_after_long_idle"],
        (WARMUP_WINDOW_RUNS - runs["warmup_run_index_after_idle"] + 1).clip(lower=1) / WARMUP_WINDOW_RUNS,
        0.0,
    )
    runs.loc[runs["is_warmup_program"], "thermal_state_score"] = 1.0
    return runs.sort_values(["machine_id", "program", "started_at"], kind="stable")


def build_run_tools(df: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    cutting = df[df["is_cutting"] & df["tool"].ne(NO_TOOL)].copy()
    grouped = cutting.groupby(["run_id", "tool"], sort=True)
    run_tools = pd.DataFrame(
        {
            "machine_id": grouped["machine_id"].first(),
            "program": grouped["program"].first(),
            "first_tool": grouped["tool"].first(),
            "started_at": grouped["created_at"].min(),
            "ended_at": grouped["created_at"].max(),
            "cutting_samples": grouped.size(),
            "cutting_s": grouped["dt_s"].sum(),
            "current_mean": grouped["current1"].mean(),
            "current_median": grouped["current1"].median(),
            "current_p95": grouped["current1"].quantile(0.95),
            "current_max": grouped["current1"].max(),
            "current_std": grouped["current1"].std(ddof=1),
            "current_integral_As": grouped.apply(
                lambda x: float((x["current1"] * x["dt_s"]).sum()), include_groups=False
            ),
            "tool_head_change_sample": grouped["tool_head_change_sample"].any(),
        }
    )

    run_context = runs[
        [
            "batch_id", "part_count", "job_event_id", "job_eventable_type", "job_eventable_id",
            "alarm_active", "alarm_tool_life_over", "alarm_maintenance", "is_mdi",
            "is_warmup_program", "capture_gap_flag", "part_count_missing",
            "wall_clock_s", "run_time_integrity_ratio",
            "completed_run", "truncated_start_run", "usable_run", "analysis_ready_run",
            "run_no_in_program", "batch_boundary_run", "job_change_between_runs",
            "minutes_since_previous_run_on_machine", "long_idle_before_run",
            "thermal_window_id", "warmup_run_index_after_idle",
            "in_first_runs_after_long_idle", "thermal_state_score", "multi_tool_run",
        ]
    ]
    run_tools = run_tools.join(run_context, on="run_id")
    return run_tools.sort_values(["machine_id", "program", "first_tool", "started_at"], kind="stable")


# --------------------------------------------------------------------------- #
#  Neutral context features (no target labels)
# --------------------------------------------------------------------------- #

def add_current_shift_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["current_shift_z"] = np.nan
    frame["current_shift_abs_z"] = 0.0

    keys = ["machine_id", "program", "first_tool"]
    for _, idx in frame.groupby(keys, sort=False).groups.items():
        ordered = frame.loc[list(idx)].sort_values("started_at")
        usable = ordered["usable_run"] & ordered["current_mean"].notna()
        current = ordered["current_mean"].where(usable)

        rolling_median = current.shift().rolling(ROLLING_WINDOW_RUNS, min_periods=5).median()
        rolling_sigma = (
            current.shift()
            .rolling(ROLLING_WINDOW_RUNS, min_periods=5)
            .apply(robust_scale, raw=False)
            .replace(0, np.nan)
        )
        z = (current - rolling_median) / rolling_sigma

        frame.loc[ordered.index, "current_shift_z"] = z
        frame.loc[ordered.index, "current_shift_abs_z"] = z.abs().fillna(0.0).clip(upper=10.0)

    frame["batch_or_job_boundary"] = frame["batch_boundary_run"] | frame["job_change_between_runs"]
    return frame


def add_repeated_run_trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Group-level drift descriptors.

    These are constant across every row of a (machine, program, tool, batch)
    group by construction.  The ``group_`` prefix makes that explicit -- the
    original unprefixed names invited them to be read as per-run features.
    """
    frame = frame.copy()
    frame["group_run_count_for_trend"] = 0
    frame["group_current_corr_with_run_order"] = np.nan
    frame["group_current_delta_fraction_first_to_last"] = np.nan

    stable = (
        frame.get("analysis_ready_run", frame["usable_run"])
        & frame["current_mean"].notna()
        & ~frame["alarm_active"]
        & ~frame["is_mdi"]
        & ~frame["is_warmup_program"]
        & frame["first_tool"].ne(NO_TOOL)
    )

    keys = ["machine_id", "program", "first_tool", "batch_id"]
    for _, idx in frame[stable].groupby(keys, sort=False).groups.items():
        ordered = frame.loc[list(idx)].sort_values("started_at")
        if len(ordered) < 3:
            continue

        x = np.arange(1, len(ordered) + 1, dtype=float)
        y = ordered["current_mean"].to_numpy(dtype=float)
        corr = float(pd.Series(x).corr(pd.Series(y)))
        # A12: non-overlapping head/tail blocks.
        first_level, last_level, _ = first_last_medians(y)
        delta_fraction = (
            (last_level - first_level) / max(abs(first_level), 1e-9)
            if np.isfinite(first_level) and np.isfinite(last_level)
            else np.nan
        )

        frame.loc[ordered.index, "group_run_count_for_trend"] = len(ordered)
        frame.loc[ordered.index, "group_current_corr_with_run_order"] = corr
        frame.loc[ordered.index, "group_current_delta_fraction_first_to_last"] = delta_fraction

    return frame


# --------------------------------------------------------------------------- #
#  Projection to sample level
# --------------------------------------------------------------------------- #

RUN_PROJECTION_COLUMNS = [
    "tool_changed_in_or_before_run",
    "current_shift_z",
    "current_shift_abs_z",
    "batch_or_job_boundary",
    "group_current_corr_with_run_order",
    "group_current_delta_fraction_first_to_last",
    "group_run_count_for_trend",
    "thermal_state_score",
    "minutes_since_previous_run_on_machine",
    "long_idle_before_run",
    "warmup_run_index_after_idle",
    "in_first_runs_after_long_idle",
    "usable_run",
    "completed_run",
    "multi_tool_run",
    "analysis_ready_run",
    "wall_clock_s",
    "run_time_integrity_ratio",
]

RUN_TOOL_PROJECTION_COLUMNS = [
    "current_shift_z",
    "current_shift_abs_z",
    "batch_or_job_boundary",
    "group_current_corr_with_run_order",
    "group_current_delta_fraction_first_to_last",
    "group_run_count_for_trend",
    "thermal_state_score",
]


def project_run_features_to_samples(df: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    features = runs[RUN_PROJECTION_COLUMNS].add_prefix("run_")
    return df.join(features, on="run_id")


def project_run_tool_features_to_samples(df: pd.DataFrame, run_tools: pd.DataFrame) -> pd.DataFrame:
    features = (
        run_tools[RUN_TOOL_PROJECTION_COLUMNS]
        .add_prefix("run_tool_")
        .reset_index()
        .set_index(["run_id", "tool"])
    )
    if features.index.has_duplicates:
        raise ValueError("run/tool index is not unique; sample projection would duplicate rows")
    return df.join(features, on=["run_id", "tool"])


# --------------------------------------------------------------------------- #
#  Output
# --------------------------------------------------------------------------- #

SAMPLE_COLUMNS = [
    "created_at", "machine_id", "device_id", "program", "tool", "part_count",
    "part_count_missing", "batch_id", "run_id", "sample_in_run", "seconds_in_run",
    "current1", "energy", "feed_override", "alarms", "is_cutting",
    "cutting_threshold_a", "dt_s", "gap_s", "capture_gap_flag", "intra_run_gap_s",
    "program_change_sample", "run_split_by_gap",
    "tool_head_change_sample", "job_change_sample", "batch_boundary_sample",
    "run_tool_changed_in_or_before_run", "run_current_shift_z", "run_current_shift_abs_z",
    "run_batch_or_job_boundary", "run_group_current_corr_with_run_order",
    "run_group_current_delta_fraction_first_to_last", "run_group_run_count_for_trend",
    "run_thermal_state_score", "run_minutes_since_previous_run_on_machine",
    "run_long_idle_before_run", "run_warmup_run_index_after_idle",
    "run_in_first_runs_after_long_idle", "run_wall_clock_s", "run_run_time_integrity_ratio",
    "run_tool_current_shift_z", "run_tool_current_shift_abs_z",
    "run_tool_batch_or_job_boundary", "run_tool_group_current_corr_with_run_order",
    "run_tool_group_current_delta_fraction_first_to_last",
    "run_tool_group_run_count_for_trend", "run_tool_thermal_state_score",
    "run_usable_run", "run_analysis_ready_run", "source_file",
]


def write_outputs(
    samples: pd.DataFrame,
    runs: pd.DataFrame,
    run_tools: pd.DataFrame,
    thresholds: dict[int, float],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    available = [c for c in SAMPLE_COLUMNS if c in samples.columns]
    samples[available].to_csv(output_dir / "cnc_time_series_features.csv.gz", index=False)
    runs.reset_index(names="run_id").to_csv(output_dir / "cnc_runs_features.csv", index=False)
    run_tools.reset_index().to_csv(output_dir / "cnc_run_tool_features.csv", index=False)

    threshold_lines = "\n".join(
        f"  machine {machine}: {value:.3f} A" for machine, value in sorted(thresholds.items())
    )
    integrity_bad = int((runs["run_time_integrity_ratio"] > 2).sum())

    summary = (
        f"rows: {len(samples):,}\n"
        f"runs: {len(runs):,}\n"
        f"run/tool rows: {len(run_tools):,}\n"
        f"machines: {samples['machine_id'].nunique():,}\n"
        f"machine/program series: {samples.groupby(['machine_id', 'program']).ngroups:,}\n"
        f"span: {samples['created_at'].min()} .. {samples['created_at'].max()}\n"
        f"usable runs: {int(runs['usable_run'].sum()):,}\n"
        f"analysis-ready runs: {int(runs['analysis_ready_run'].sum()):,}\n"
        f"cutting hours: {samples.loc[samples['is_cutting'], 'dt_s'].sum() / 3600:.2f}\n"
        f"capture-gap runs: {int(runs['capture_gap_flag'].sum()):,} "
        f"({100 * runs['capture_gap_flag'].mean():.1f}%) [machine timeline]\n"
        f"runs with wall-clock > 2x active seconds: {integrity_bad:,}\n"
        f"runs split by telemetry gap > {args.split_run_gap_s:.0f}s: "
        f"{int(samples['run_split_by_gap'].sum()):,}\n"
        f"cutting thresholds (Otsu per machine):\n{threshold_lines}\n"
        f"feature-only output: no heuristic target labels are written\n"
    )
    (output_dir / "preprocessing_summary.txt").write_text(summary, encoding="utf-8")
    write_manifest(output_dir, "preprocess_cnc", args, {"cutting_thresholds": thresholds})

    print(summary)
    print(f"Wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build feature-only CNC time-series, run and run/tool datasets."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pattern", default="gold_*.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("processed"))
    parser.add_argument(
        "--cutting-threshold", type=float, default=None,
        help="Override the per-machine Otsu threshold with one global value (legacy: 0.6).",
    )
    parser.add_argument(
        "--split-run-gap-s", type=float, default=DEFAULT_SPLIT_RUN_GAP_S,
        help="Telemetry gap above which a run is split. Use a huge value for legacy behaviour.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_raw(args.data_dir, args.pattern)

    thresholds = machine_cutting_thresholds(raw, args.cutting_threshold)
    print("Cutting thresholds per machine (A):", {k: round(v, 3) for k, v in thresholds.items()})

    samples = add_time_series_flags(raw, thresholds, args.split_run_gap_s)
    runs = build_runs(samples)
    runs = add_thermal_context_features(runs)
    runs = add_current_shift_features(runs)
    runs = add_repeated_run_trend_features(runs)

    run_tools = build_run_tools(samples, runs)
    run_tools = add_current_shift_features(run_tools)
    run_tools = add_repeated_run_trend_features(run_tools)

    samples = project_run_features_to_samples(samples, runs)
    samples = project_run_tool_features_to_samples(samples, run_tools)
    write_outputs(samples, runs, run_tools, thresholds, args.output_dir, args)


if __name__ == "__main__":
    main()
