"""Materialise DTW-ready comparable time-series datasets for each window.

Fixes applied relative to the original version
----------------------------------------------
B1  **Non-cutting intervals were excised and the survivors concatenated.**
    ``attach_window_membership`` kept only ``is_cutting`` rows and then rebuilt
    the time axis as a cumulative sum over those rows alone.  Air cuts,
    rapids, dwells and retracts were deleted and the gaps closed up, so a tool
    that cut in three separate bursts had them welded into one continuous
    pseudo-signal.  DTW was aligning a trace whose time axis no longer
    corresponded to the machining cycle, and ``template_phase`` -- which the
    phase-specific trend stage interprets as position along the tool path --
    was a position in that edited signal.

    The default is now ``--trace-mode contiguous``: the segment runs from the
    first to the last cutting sample of that tool within the run and *keeps
    everything in between*, so elapsed time is real.  ``burst_id`` and
    ``burst_count`` identify the individual cutting bursts, and
    ``cutting_fraction_in_segment`` records how much of the segment was under
    load.  ``--trace-mode cutting_only`` reproduces the old behaviour.

C1  Each window now materialises both the reference set (clean runs, used to
    learn normal behaviour) and the scoring set (every run in the window's
    time span, including alarm / post-idle / boundary runs).  Rows are tagged
    with ``in_reference_set`` so the DTW stage can build references from one
    and score the other.

Also: the phase-resampled tables were produced by every run but never read by
any downstream stage (~19 MB per full run).  They are now opt-in via
``--write-resampled``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import write_manifest


DEFAULT_WINDOWS = Path("processed/cnc_dtw_best_windows.csv")
DEFAULT_SAMPLES = Path("processed/cnc_time_series_features.csv.gz")
DEFAULT_RUN_TOOLS = Path("processed/cnc_run_tool_features.csv")
DEFAULT_OUTPUT_DIR = Path("processed/comparable_timeseries")

SAMPLE_COLUMNS = [
    "created_at", "machine_id", "device_id", "program", "tool", "part_count",
    "batch_id", "run_id", "sample_in_run", "seconds_in_run", "current1", "energy",
    "feed_override", "alarms", "is_cutting", "dt_s", "gap_s", "intra_run_gap_s",
    "capture_gap_flag", "tool_head_change_sample", "job_change_sample",
    "batch_boundary_sample", "source_file",
]


def safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("_")[:120]


def parse_run_ids(value: object) -> list[int]:
    if pd.isna(value):
        return []
    return [int(float(v)) for v in str(value).split(",") if str(v).strip()]


def load_windows(path: Path) -> pd.DataFrame:
    windows = pd.read_csv(path, low_memory=False)
    windows["tool"] = windows["tool"].astype(str)
    reference_column = "reference_run_ids" if "reference_run_ids" in windows.columns else "run_ids"
    windows["reference_run_id_list"] = windows[reference_column].apply(parse_run_ids)
    if "scoring_run_ids" in windows.columns:
        windows["scoring_run_id_list"] = windows["scoring_run_ids"].apply(parse_run_ids)
    else:
        windows["scoring_run_id_list"] = windows["reference_run_id_list"]
    return windows


def filter_windows(
    windows: pd.DataFrame, top_n: int | None,
    machine_id: int | None, program: str | None, tool: str | None,
) -> pd.DataFrame:
    out = windows.copy()
    if machine_id is not None:
        out = out[out["machine_id"].eq(machine_id)]
    if program is not None:
        out = out[out["program"].astype(str).eq(program)]
    if tool is not None:
        out = out[out["tool"].astype(str).eq(str(tool))]
    out = out.sort_values(["quality_score", "num_runs", "runtime_cv"], ascending=[False, False, True])
    if top_n is not None:
        out = out.head(top_n)
    out = out.reset_index(drop=True)
    out["comparable_window_id"] = np.arange(1, len(out) + 1)
    return out


def load_samples(path: Path, selected_run_ids: set[int]) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(
        path, usecols=lambda c: c in SAMPLE_COLUMNS, chunksize=200_000, low_memory=False
    ):
        chunk["run_id"] = pd.to_numeric(chunk["run_id"], errors="coerce").astype("Int64")
        chunk = chunk[chunk["run_id"].isin(selected_run_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)

    samples = pd.concat(chunks, ignore_index=True)
    samples["created_at"] = pd.to_datetime(samples["created_at"], utc=True, errors="coerce")
    samples["tool"] = samples["tool"].astype(str)
    samples["program"] = samples["program"].astype(str)
    samples["is_cutting"] = samples["is_cutting"].astype(str).str.lower().isin(["true", "1"])
    samples["dt_s"] = pd.to_numeric(samples["dt_s"], errors="coerce").fillna(1.0)
    samples["current1"] = pd.to_numeric(samples["current1"], errors="coerce")
    return samples


def load_run_tools(path: Path, selected_run_ids: set[int]) -> pd.DataFrame:
    run_tools = pd.read_csv(path, low_memory=False)
    run_tools["run_id"] = pd.to_numeric(run_tools["run_id"], errors="coerce").astype("Int64")
    run_tools = run_tools[run_tools["run_id"].isin(selected_run_ids)].copy()
    run_tools["tool"] = run_tools["tool"].astype(str)
    run_tools["program"] = run_tools["program"].astype(str)
    return run_tools


# --------------------------------------------------------------------------- #
#  B1: segment extraction that preserves the real time axis
# --------------------------------------------------------------------------- #

def extract_tool_segment(part: pd.DataFrame, trace_mode: str) -> pd.DataFrame:
    """One tool's segment inside one run.

    ``contiguous`` keeps everything between the first and last cutting sample,
    so the elapsed-time axis is physically real and intra-cycle structure
    (retracts, dwells, repositioning between cutting bursts) survives.
    ``cutting_only`` reproduces the original behaviour.
    """
    part = part.sort_values("created_at", kind="stable")
    cutting_positions = np.flatnonzero(part["is_cutting"].to_numpy())
    if cutting_positions.size == 0:
        return pd.DataFrame()

    if trace_mode == "cutting_only":
        segment = part.iloc[cutting_positions].copy()
    else:
        first, last = cutting_positions[0], cutting_positions[-1]
        segment = part.iloc[first:last + 1].copy()

    if len(segment) < 2:
        return pd.DataFrame()

    # Contiguous runs of cutting samples inside the segment.
    is_cut = segment["is_cutting"].to_numpy()
    burst_id = np.cumsum(np.concatenate([[True], is_cut[1:] & ~is_cut[:-1]])) * is_cut
    segment["burst_id"] = burst_id
    segment["burst_count"] = int(burst_id.max())
    segment["cutting_fraction_in_segment"] = float(is_cut.mean())

    dt = segment["dt_s"].to_numpy(dtype=float)
    segment["seconds_in_tool_segment"] = np.concatenate([[0.0], np.cumsum(dt)[:-1]])
    # Backwards-compatible alias: downstream stages read this name.
    segment["seconds_in_tool_cut"] = segment["seconds_in_tool_segment"]
    segment["sample_in_tool_cut"] = np.arange(1, len(segment) + 1)
    return segment


def attach_window_membership(
    samples: pd.DataFrame, windows: pd.DataFrame, trace_mode: str,
) -> pd.DataFrame:
    rows = []
    for _, window in windows.iterrows():
        reference_ids = set(window["reference_run_id_list"])
        scoring_ids = set(window["scoring_run_id_list"]) | reference_ids
        mask = (
            samples["run_id"].isin(scoring_ids)
            & samples["machine_id"].eq(window["machine_id"])
            & samples["program"].astype(str).eq(str(window["program"]))
            & samples["tool"].astype(str).eq(str(window["tool"]))
        )
        part = samples.loc[mask]
        if part.empty:
            continue

        for run_id, run_part in part.groupby("run_id", sort=True):
            segment = extract_tool_segment(run_part, trace_mode)
            if segment.empty:
                continue
            segment["comparable_window_id"] = int(window["comparable_window_id"])
            segment["window_quality_score"] = float(window["quality_score"])
            segment["in_reference_set"] = int(run_id) in reference_ids
            rows.append(segment)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["comparable_window_id", "run_id", "created_at"], kind="stable")
    out["segment_key"] = (
        out["comparable_window_id"].astype(str) + "_run_" + out["run_id"].astype(str)
        + "_tool_" + out["tool"].astype(str)
    )
    return out


def attach_window_membership_run_tools(run_tools: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, window in windows.iterrows():
        reference_ids = set(window["reference_run_id_list"])
        scoring_ids = set(window["scoring_run_id_list"]) | reference_ids
        mask = (
            run_tools["run_id"].isin(scoring_ids)
            & run_tools["machine_id"].eq(window["machine_id"])
            & run_tools["program"].astype(str).eq(str(window["program"]))
            & run_tools["tool"].astype(str).eq(str(window["tool"]))
        )
        part = run_tools.loc[mask].copy()
        if part.empty:
            continue
        part["comparable_window_id"] = int(window["comparable_window_id"])
        part["window_quality_score"] = float(window["quality_score"])
        part["in_reference_set"] = part["run_id"].astype(int).isin(reference_ids)
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["comparable_window_id", "started_at"], kind="stable"
    )


# --------------------------------------------------------------------------- #
#  Optional phase-resampled views (not consumed by the DTW stage)
# --------------------------------------------------------------------------- #

def resample_segment(segment: pd.DataFrame, phase_points: int) -> pd.DataFrame:
    segment = segment.sort_values("seconds_in_tool_segment", kind="stable")
    t = segment["seconds_in_tool_segment"].to_numpy(dtype=float)
    y = segment["current1"].to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    if len(t) < 2 or t.max() <= t.min():
        return pd.DataFrame()

    unique_t, unique_idx = np.unique(t, return_index=True)
    if unique_t.size < 2:
        return pd.DataFrame()
    phase = (unique_t - unique_t.min()) / (unique_t.max() - unique_t.min())
    grid = np.linspace(0.0, 1.0, phase_points)

    first = segment.iloc[0]
    return pd.DataFrame({
        "comparable_window_id": int(first["comparable_window_id"]),
        "machine_id": int(first["machine_id"]),
        "program": first["program"],
        "tool": first["tool"],
        "run_id": int(first["run_id"]),
        "phase": grid,
        "current1": np.interp(grid, phase, y[unique_idx]),
    })


def build_resampled(samples: pd.DataFrame, phase_points: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = [
        resampled
        for _, segment in samples.groupby(["comparable_window_id", "run_id", "tool"], sort=False)
        if not (resampled := resample_segment(segment, phase_points)).empty
    ]
    if not long_rows:
        return pd.DataFrame(), pd.DataFrame()

    long = pd.concat(long_rows, ignore_index=True)
    wide = long.pivot_table(
        index=["comparable_window_id", "machine_id", "program", "tool", "run_id"],
        columns="phase", values="current1",
    ).reset_index()
    phase_i = 0
    columns = []
    for col in wide.columns:
        if isinstance(col, float):
            columns.append(f"phase_{phase_i:03d}")
            phase_i += 1
        else:
            columns.append(col)
    wide.columns = columns
    return long, wide


def write_per_window_outputs(
    samples: pd.DataFrame, run_tools: pd.DataFrame,
    windows: pd.DataFrame, output_dir: Path,
) -> None:
    per_window_dir = output_dir / "windows"
    per_window_dir.mkdir(parents=True, exist_ok=True)
    drop = [c for c in ["reference_run_id_list", "scoring_run_id_list"] if c in windows.columns]
    for _, window in windows.iterrows():
        name = (
            f"window_{int(window['comparable_window_id']):03d}"
            f"_m{window['machine_id']}_{safe_name(window['program'])}"
            f"_tool{safe_name(window['tool'])}"
        )
        folder = per_window_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        wid = window["comparable_window_id"]
        window.drop(labels=drop).to_frame().T.to_csv(folder / "window_metadata.csv", index=False)
        run_tools[run_tools["comparable_window_id"].eq(wid)].to_csv(
            folder / "run_tool_segments.csv", index=False)
        samples[samples["comparable_window_id"].eq(wid)].to_csv(
            folder / "samples_long.csv.gz", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DTW-ready comparable time-series datasets.")
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--run-tools", type=Path, default=DEFAULT_RUN_TOOLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--machine-id", type=int)
    parser.add_argument("--program")
    parser.add_argument("--tool")
    parser.add_argument("--phase-points", type=int, default=200)
    parser.add_argument("--all-windows", action="store_true")
    parser.add_argument(
        "--trace-mode", choices=["contiguous", "cutting_only"], default="contiguous",
        help="'contiguous' keeps non-cutting samples between the first and last cut "
             "so the time axis is real; 'cutting_only' is the original behaviour.",
    )
    parser.add_argument(
        "--write-resampled", action="store_true",
        help="Also write the phase-normalised tables (not used by any downstream stage).",
    )
    parser.add_argument("--skip-per-window", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_n = None if args.all_windows else args.top_n
    windows = filter_windows(
        load_windows(args.windows), top_n=top_n,
        machine_id=args.machine_id, program=args.program, tool=args.tool,
    )
    if windows.empty:
        raise SystemExit("No windows matched the requested filters.")

    selected_run_ids = {
        run_id
        for column in ["reference_run_id_list", "scoring_run_id_list"]
        for ids in windows[column] for run_id in ids
    }
    samples = load_samples(args.samples, selected_run_ids=selected_run_ids)
    run_tools = load_run_tools(args.run_tools, selected_run_ids=selected_run_ids)

    comparable_samples = attach_window_membership(samples, windows, args.trace_mode)
    comparable_run_tools = attach_window_membership_run_tools(run_tools, windows)
    if comparable_samples.empty:
        raise SystemExit("No samples matched the selected windows.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = windows.drop(columns=["reference_run_id_list", "scoring_run_id_list"])
    manifest.to_csv(args.output_dir / "windows_manifest.csv", index=False)
    comparable_run_tools.to_csv(args.output_dir / "run_tool_segments.csv", index=False)
    comparable_samples.to_csv(args.output_dir / "samples_long.csv.gz", index=False)

    if args.write_resampled:
        resampled_long, resampled_wide = build_resampled(comparable_samples, args.phase_points)
        resampled_long.to_csv(args.output_dir / "resampled_phase_long.csv", index=False)
        resampled_wide.to_csv(args.output_dir / "resampled_phase_wide.csv", index=False)

    if not args.skip_per_window:
        write_per_window_outputs(comparable_samples, comparable_run_tools, windows, args.output_dir)

    reference_segments = comparable_run_tools["in_reference_set"].sum()
    print(f"windows: {len(windows):,}")
    print(f"unique runs materialised: {comparable_samples['run_id'].nunique():,}")
    print(f"samples: {len(comparable_samples):,}")
    print(f"run/tool segments: {len(comparable_run_tools):,} "
          f"({reference_segments:,} reference / "
          f"{len(comparable_run_tools) - reference_segments:,} scoring-only)")
    print(f"trace mode: {args.trace_mode}")
    print(f"median cutting fraction within segment: "
          f"{comparable_samples['cutting_fraction_in_segment'].median():.3f}")
    print(f"median bursts per segment: {comparable_samples['burst_count'].median():.1f}")

    write_manifest(args.output_dir, "create_comparable_timeseries_datasets", args, {
        "windows": len(windows),
        "run_tool_segments": int(len(comparable_run_tools)),
        "reference_segments": int(reference_segments),
    })
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
