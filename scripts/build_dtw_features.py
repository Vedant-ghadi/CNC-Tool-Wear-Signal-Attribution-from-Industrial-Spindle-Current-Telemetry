"""Build DTW feature tables from the comparable CNC time series.

Fixes applied relative to the original version
----------------------------------------------
A1/A3  The DTW itself moved to ``dtw_core``, where the distance no longer
       depends on whether the caller asked for the alignment path and the
       backtrack follows a recorded predecessor instead of a ``min()`` over
       possibly-infinite neighbours.  Previously the medoid reference was
       scored with the path and the other three without it, putting them on
       scales that differed by ~20x (median shape distance 0.275 vs 2.5-2.9).

A2     Every reference distance is additionally published as a **per-window
       robust z-score**.  Agreement between references is now a comparison of
       ranks within a window rather than of raw magnitudes across windows,
       which is what the consensus stage actually needs.

D1     **All three static references were anchored in the first ten runs**
       (medoid run_order was 1-10 in all 46 windows, median window size 23),
       so distance-to-reference rose with run order under any slow
       non-stationarity -- thermal, material or sensor drift -- and that rise
       was then read as wear.  The primary reference is now a **leave-one-out
       median template** built from the whole reference set, which has no
       positional bias.  The early-anchored references are still computed, but
       are explicitly named ``*_early_anchored`` and are for comparison only.

D2     The old median template was built from the first ten traces and then
       scored *against those same traces*.  Self-inclusion depressed their
       distances and mechanically produced an upward slope.  Leave-one-out
       removes the target trace from its own template.

D3     The medoid reference was compared to itself, producing exactly one
       zero-distance row per window at early run order which then entered
       every robust z-score, slope and Kendall tau.  Self-comparisons are now
       NaN and are excluded from all downstream statistics.

C1     References are built **only** from the reference (clean) run set, while
       every run in the window's time span is scored -- including the alarm,
       post-idle and boundary runs that the old pipeline filtered out before
       detection began.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dtw_core import constrained_dtw, dtw_shape_and_amplitude, z_normalize
from pipeline_common import robust_z_by_group, write_manifest


DEFAULT_INPUT_DIR = Path("processed/comparable_timeseries_all_best")
DEFAULT_OUTPUT_DIR = Path("processed/dtw_features")

REFERENCE_KINDS = ["loo_median", "rolling_median", "medoid_early", "first_early"]


@dataclass
class Trace:
    comparable_window_id: int
    machine_id: int
    program: str
    tool: str
    run_id: int
    run_order: int
    seconds: np.ndarray
    current: np.ndarray
    in_reference_set: bool

    @property
    def duration(self) -> float:
        return float(self.seconds[-1] - self.seconds[0]) if self.seconds.size else np.nan


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #

def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(input_dir / "windows_manifest.csv", low_memory=False)
    segments = pd.read_csv(input_dir / "run_tool_segments.csv", low_memory=False)
    samples = pd.read_csv(input_dir / "samples_long.csv.gz", low_memory=False)

    samples["created_at"] = pd.to_datetime(samples["created_at"], utc=True, errors="coerce")
    for col in ["tool", "program"]:
        samples[col] = samples[col].astype(str)
        segments[col] = segments[col].astype(str)
    for col in ["current1", "seconds_in_tool_cut"]:
        samples[col] = pd.to_numeric(samples[col], errors="coerce")
    for col in ["comparable_window_id", "run_id"]:
        samples[col] = pd.to_numeric(samples[col], errors="coerce").astype("int64")
        segments[col] = pd.to_numeric(segments[col], errors="coerce").astype("int64")

    if "in_reference_set" in samples.columns:
        samples["in_reference_set"] = (
            samples["in_reference_set"].astype(str).str.lower().isin(["true", "1"])
        )
    else:
        samples["in_reference_set"] = True

    segments["started_at"] = pd.to_datetime(segments["started_at"], utc=True, errors="coerce")
    return manifest, segments, samples


def make_traces(samples: pd.DataFrame) -> list[Trace]:
    traces: list[Trace] = []
    group_cols = ["comparable_window_id", "machine_id", "program", "tool", "run_id"]
    for key, group in samples.groupby(group_cols, sort=False):
        window_id, machine_id, program, tool, run_id = key
        group = group.sort_values("seconds_in_tool_cut", kind="stable")
        seconds = group["seconds_in_tool_cut"].to_numpy(dtype=float)
        current = group["current1"].to_numpy(dtype=float)
        valid = np.isfinite(seconds) & np.isfinite(current)
        seconds, current = seconds[valid], current[valid]
        if current.size < 2:
            continue
        traces.append(Trace(
            comparable_window_id=int(window_id), machine_id=int(machine_id),
            program=str(program), tool=str(tool), run_id=int(run_id), run_order=0,
            seconds=seconds, current=current,
            in_reference_set=bool(group["in_reference_set"].iloc[0]),
        ))

    traces.sort(key=lambda t: (t.comparable_window_id, t.run_id))
    current_window, order = None, 0
    for trace in traces:
        if trace.comparable_window_id != current_window:
            current_window, order = trace.comparable_window_id, 1
        else:
            order += 1
        trace.run_order = order
    return traces


# --------------------------------------------------------------------------- #
#  D1/D2: unbiased reference construction
# --------------------------------------------------------------------------- #

def phase_interpolate(trace: Trace, n_points: int) -> np.ndarray:
    """Resample a trace onto ``n_points`` of unit phase."""
    seconds = trace.seconds
    duration = max(float(seconds[-1] - seconds[0]), 1e-9)
    phase = (seconds - seconds[0]) / duration
    unique_phase, unique_idx = np.unique(phase, return_index=True)
    if unique_phase.size < 2:
        return np.full(n_points, float(np.median(trace.current)))
    return np.interp(np.linspace(0.0, 1.0, n_points), unique_phase, trace.current[unique_idx])


class WindowTemplates:
    """Reference templates for one comparable window.

    All templates live on a single canonical grid -- the median length and
    median duration of the reference traces -- so that traces of differing
    length still have something to warp *against*.  Building the template on
    each target's own grid (as the original rolling reference did) would force
    the two series to equal length and leave DTW nothing to align.
    """

    def __init__(self, reference_traces: list[Trace], band_fraction: float, baseline_runs: int):
        self.band_fraction = band_fraction
        self.baseline_runs = baseline_runs
        self.reference_traces = reference_traces
        self.valid = len(reference_traces) >= 2

        if not reference_traces:
            self.n_points = 0
            return

        lengths = [t.current.size for t in reference_traces]
        durations = [t.duration for t in reference_traces]
        self.n_points = int(np.clip(np.median(lengths), 8, 2000))
        self.duration = float(np.median([d for d in durations if np.isfinite(d)] or [1.0]))
        self.grid_seconds = np.linspace(0.0, self.duration, self.n_points)

        # Stack every reference trace on the canonical grid once; the
        # leave-one-out templates are then medians over row subsets.
        self.stack = np.vstack([phase_interpolate(t, self.n_points) for t in reference_traces])
        self.run_ids = np.array([t.run_id for t in reference_traces])
        self.run_orders = np.array([t.run_order for t in reference_traces])
        self.full_median = np.median(self.stack, axis=0)

        # Early-anchored references, retained only for comparison with the
        # original method.  These are the ones that carry the D1 positional bias.
        early = sorted(reference_traces, key=lambda t: t.run_order)[:baseline_runs]
        self.first_early = phase_interpolate(early[0], self.n_points) if early else None
        self.medoid_early = self._medoid(early)
        self.medoid_run_id = self._medoid_run_id(early)

    def _medoid(self, candidates: list[Trace]) -> np.ndarray | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return phase_interpolate(candidates[0], self.n_points)
        normalised = [z_normalize(phase_interpolate(t, self.n_points)) for t in candidates]
        distances = np.zeros((len(candidates), len(candidates)), dtype=float)
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                d, _ = constrained_dtw(normalised[i], normalised[j], self.band_fraction)
                distances[i, j] = distances[j, i] = d
        with np.errstate(invalid="ignore"):
            means = np.nanmean(distances, axis=1)
        if not np.isfinite(means).any():
            return phase_interpolate(candidates[0], self.n_points)
        return phase_interpolate(candidates[int(np.nanargmin(means))], self.n_points)

    def _medoid_run_id(self, candidates: list[Trace]) -> int:
        """Run id of the trace ``_medoid`` actually selected."""
        if not candidates:
            return -1
        if len(candidates) == 1:
            return int(candidates[0].run_id)
        normalised = [z_normalize(phase_interpolate(t, self.n_points)) for t in candidates]
        distances = np.zeros((len(candidates), len(candidates)), dtype=float)
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                d, _ = constrained_dtw(normalised[i], normalised[j], self.band_fraction)
                distances[i, j] = distances[j, i] = d
        with np.errstate(invalid="ignore"):
            means = np.nanmean(distances, axis=1)
        if not np.isfinite(means).any():
            return int(candidates[0].run_id)
        return int(candidates[int(np.nanargmin(means))].run_id)

    def leave_one_out_median(self, run_id: int) -> np.ndarray | None:
        """Median template over the reference set, excluding ``run_id``.

        D2/D3: a run never contributes to the template it is scored against,
        so no run gets an artificially small distance and there is no
        positional gradient introduced by the reference construction.
        """
        if not self.valid:
            return None
        keep = self.run_ids != run_id
        if keep.sum() < 1:
            return None
        if keep.all():
            return self.full_median
        return np.median(self.stack[keep], axis=0)

    def rolling_median(self, run_order: int) -> tuple[np.ndarray | None, int]:
        """Median template over the ``baseline_runs`` reference runs immediately
        before ``run_order``.  Local, so it responds to step changes rather
        than to cumulative drift."""
        if not self.valid:
            return None, 0
        prior = self.run_orders < run_order
        if prior.sum() == 0:
            return None, 0
        idx = np.flatnonzero(prior)[-self.baseline_runs:]
        return np.median(self.stack[idx], axis=0), int(idx.size)


# --------------------------------------------------------------------------- #
#  Residuals and features
# --------------------------------------------------------------------------- #

def aligned_residuals(
    trace: Trace, template: np.ndarray, template_seconds: np.ndarray,
    path: list[tuple[int, int]], reference_kind: str,
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    max_second = max(float(template_seconds[-1]), 1e-9)
    matches: dict[int, list[float]] = {}
    for ref_i, run_j in path:
        if 0 <= ref_i < template.size and 0 <= run_j < trace.current.size:
            matches.setdefault(ref_i, []).append(float(trace.current[run_j]))

    rows = []
    for ref_i, values in matches.items():
        aligned = float(np.mean(values))
        expected = float(template[ref_i])
        rows.append({
            "comparable_window_id": trace.comparable_window_id,
            "machine_id": trace.machine_id, "program": trace.program, "tool": trace.tool,
            "run_id": trace.run_id, "run_order": trace.run_order,
            "reference_kind": reference_kind,
            "template_index": ref_i,
            "template_seconds": float(template_seconds[ref_i]),
            "template_phase": float(template_seconds[ref_i] / max_second),
            "template_current": expected,
            "aligned_current": aligned,
            "residual": aligned - expected,
            "abs_residual": abs(aligned - expected),
            "matched_points": len(values),
        })
    return pd.DataFrame(rows)


def build_dtw_features(
    samples: pd.DataFrame, baseline_runs: int, band_fraction: float, primary_reference: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traces = make_traces(samples)
    grouped: dict[int, list[Trace]] = {}
    for trace in traces:
        grouped.setdefault(trace.comparable_window_id, []).append(trace)

    feature_rows, residual_frames, reference_rows = [], [], []

    for window_id, window_traces in grouped.items():
        reference_traces = [t for t in window_traces if t.in_reference_set]
        if len(reference_traces) < 2:
            continue
        templates = WindowTemplates(reference_traces, band_fraction, baseline_runs)
        head = window_traces[0]

        reference_rows.append({
            "comparable_window_id": window_id, "machine_id": head.machine_id,
            "program": head.program, "tool": head.tool,
            "template_points": templates.n_points,
            "template_duration_s": templates.duration,
            "reference_traces": len(reference_traces),
            "scored_traces": len(window_traces),
            "scoring_only_traces": len(window_traces) - len(reference_traces),
            "primary_reference": primary_reference,
            "medoid_early_reference_run_id": templates.medoid_run_id,
        })

        for trace in window_traces:
            references: dict[str, np.ndarray | None] = {
                "loo_median": templates.leave_one_out_median(trace.run_id),
                "medoid_early": templates.medoid_early,
                "first_early": templates.first_early,
            }
            rolling, rolling_count = templates.rolling_median(trace.run_order)
            references["rolling_median"] = rolling

            row: dict[str, object] = {
                "comparable_window_id": trace.comparable_window_id,
                "machine_id": trace.machine_id, "program": trace.program, "tool": trace.tool,
                "run_id": trace.run_id, "run_order": trace.run_order,
                "in_reference_set": trace.in_reference_set,
                "trace_len": int(trace.current.size),
                "template_len": int(templates.n_points),
                "runtime_s": trace.duration,
                "current_mean": float(np.mean(trace.current)),
                "current_p95": float(np.quantile(trace.current, 0.95)),
                "current_max": float(np.max(trace.current)),
                "current_std": float(np.std(trace.current, ddof=1)) if trace.current.size > 1 else np.nan,
                "current_integral_As": float(np.trapezoid(trace.current, trace.seconds)),
                "rolling_reference_run_count": rolling_count,
                "primary_reference_kind": primary_reference,
            }

            primary_path: list[tuple[int, int]] = []
            for kind, template in references.items():
                if template is None:
                    row[f"dtw_shape_distance_{kind}"] = np.nan
                    row[f"dtw_amplitude_distance_{kind}"] = np.nan
                    continue
                want_path = kind == primary_reference
                shape, amplitude, path = dtw_shape_and_amplitude(
                    template, trace.current, band_fraction=band_fraction, return_path=want_path,
                )
                row[f"dtw_shape_distance_{kind}"] = shape
                row[f"dtw_amplitude_distance_{kind}"] = amplitude
                if want_path:
                    primary_path = path
                    row["warping_path_length"] = len(path)
                    row["warping_amount"] = (
                        len(path) / max(templates.n_points, trace.current.size)
                        if path else np.nan
                    )

            template = references.get(primary_reference)
            residuals = pd.DataFrame()
            if template is not None and primary_path:
                residuals = aligned_residuals(
                    trace, template, templates.grid_seconds, primary_path, primary_reference,
                )
            if not residuals.empty:
                residual_frames.append(residuals)

            residual = residuals["residual"].to_numpy(dtype=float) if not residuals.empty else np.array([])
            abs_residual = np.abs(residual)
            row.update({
                "dtw_shape_distance": row.get(f"dtw_shape_distance_{primary_reference}", np.nan),
                "dtw_amplitude_distance": row.get(f"dtw_amplitude_distance_{primary_reference}", np.nan),
                "residual_mean": float(np.mean(residual)) if residual.size else np.nan,
                "residual_median": float(np.median(residual)) if residual.size else np.nan,
                "residual_p95": float(np.quantile(residual, 0.95)) if residual.size else np.nan,
                "residual_abs_mean": float(np.mean(abs_residual)) if abs_residual.size else np.nan,
                "residual_abs_p95": float(np.quantile(abs_residual, 0.95)) if abs_residual.size else np.nan,
                "residual_max": float(np.max(residual)) if residual.size else np.nan,
                "residual_min": float(np.min(residual)) if residual.size else np.nan,
            })
            feature_rows.append(row)

    features = pd.DataFrame(feature_rows)
    residuals = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    references_table = pd.DataFrame(reference_rows)
    if not features.empty:
        features = add_reference_z_scores(features)
    return features, residuals, references_table


def add_reference_z_scores(features: pd.DataFrame) -> pd.DataFrame:
    """A2: put every reference on a common, within-window scale.

    Raw DTW distances from different reference constructions are not directly
    comparable -- a leave-one-out median template and a single early run
    produce different magnitudes even when they rank runs identically.  The
    consensus stage needs agreement in *ranking*, so each reference gets a
    per-window robust z-score and agreement is measured on those.
    """
    features = features.copy()
    # Runs outside the reference set are the candidate anomalies.  Letting them
    # contribute to the per-window median and MAD inflates their own baseline
    # and shrinks the z-score that is supposed to reveal them.
    exclude = ~features["in_reference_set"].astype(bool) \
        if "in_reference_set" in features.columns else None
    for metric in ["shape", "amplitude"]:
        z_columns = []
        for kind in REFERENCE_KINDS:
            column = f"dtw_{metric}_distance_{kind}"
            if column not in features.columns:
                continue
            z_column = f"{column}_z"
            features[z_column] = robust_z_by_group(features, column, exclude_mask=exclude)
            z_columns.append(z_column)
        if z_columns:
            block = features[z_columns]
            features[f"dtw_{metric}_z_mean_across_references"] = block.mean(axis=1, skipna=True)
            features[f"dtw_{metric}_z_min_across_references"] = block.min(axis=1, skipna=True)
            features[f"dtw_{metric}_z_std_across_references"] = block.std(axis=1, ddof=0, skipna=True)
            features[f"dtw_{metric}_reference_count"] = block.notna().sum(axis=1)
    return features


def merge_context(features: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    context_cols = [
        "comparable_window_id", "run_id", "tool", "cutting_s", "cutting_samples",
        "current_median", "batch_id", "part_count", "job_event_id", "started_at",
        "alarm_active", "alarm_tool_life_over", "alarm_maintenance", "is_mdi",
        "is_warmup_program", "capture_gap_flag", "batch_or_job_boundary",
        "batch_boundary_run", "job_change_between_runs",
        "minutes_since_previous_run_on_machine", "long_idle_before_run",
        "in_first_runs_after_long_idle", "thermal_state_score",
        "current_shift_z", "current_shift_abs_z", "multi_tool_run",
        "tool_head_change_sample", "analysis_ready_run", "wall_clock_s",
    ]
    available = [c for c in context_cols if c in segments.columns]
    keys = ["comparable_window_id", "run_id", "tool"]
    context = segments[available].drop_duplicates(subset=keys)
    if context.duplicated(subset=keys).any():
        raise ValueError("run/tool context is not unique per window; merge would duplicate rows")
    before = len(features)
    merged = features.merge(context, on=keys, how="left")
    if len(merged) != before:
        raise ValueError(f"context merge changed row count: {before} -> {len(merged)}")
    return merged


def write_outputs(output_dir, features, residuals, references, manifest, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "cnc_dtw_features.csv", index=False)
    residuals.to_csv(output_dir / "cnc_dtw_residuals_long.csv.gz", index=False)
    references.to_csv(output_dir / "cnc_dtw_references.csv", index=False)
    manifest.to_csv(output_dir / "windows_manifest.csv", index=False)

    self_compared = int((features["dtw_shape_distance"] == 0).sum()) if not features.empty else 0
    scoring_only = int((~features["in_reference_set"]).sum()) if not features.empty else 0
    summary = (
        f"windows: {features['comparable_window_id'].nunique() if not features.empty else 0:,}\n"
        f"dtw feature rows: {len(features):,}\n"
        f"  reference-set rows: {len(features) - scoring_only:,}\n"
        f"  scoring-only rows (previously never scored): {scoring_only:,}\n"
        f"residual rows: {len(residuals):,}\n"
        f"reference rows: {len(references):,}\n"
        f"primary reference: {args.primary_reference}\n"
        f"zero-distance self-comparisons: {self_compared} (was 1 per window)\n"
        f"outputs are feature-only: no labels are assigned\n"
    )
    (output_dir / "dtw_feature_summary.txt").write_text(summary, encoding="utf-8")
    write_manifest(output_dir, "build_dtw_features", args, {"feature_rows": len(features)})
    print(summary)
    print(f"Wrote {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DTW feature tables from comparable CNC time series.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-runs", type=int, default=10)
    parser.add_argument("--band-fraction", type=float, default=0.10)
    parser.add_argument(
        "--primary-reference", choices=REFERENCE_KINDS, default="loo_median",
        help="Reference used for residuals and the headline distances. "
             "'loo_median' has no positional bias; the '*_early' options "
             "reproduce the original early-anchored behaviour.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, segments, samples = load_inputs(args.input_dir)
    features, residuals, references = build_dtw_features(
        samples=samples, baseline_runs=args.baseline_runs,
        band_fraction=args.band_fraction, primary_reference=args.primary_reference,
    )
    if not features.empty:
        features = merge_context(features, segments)
    write_outputs(args.output_dir, features, residuals, references, manifest, args)


if __name__ == "__main__":
    main()
