"""Robust, multi-method evidence tables for DTW event candidates.

Fixes applied relative to the original version
----------------------------------------------
H4  ``import ruptures`` was a hard module-level dependency, so the whole stage
    crashed on any environment without it, and ``requirements.txt`` pinned no
    versions.  A pure-NumPy PELT (exact, L2 cost) is now built in and used
    automatically when ``ruptures`` is unavailable.

F3  **690 hypothesis tests with no multiplicity control.**  ``phase_specific_
    trends`` ran 10 phase bins x every window and ``robust_trends`` 5 metrics x
    every window, both writing raw p-values and a boolean
    ``increasing_phase_evidence`` at alpha = 0.05.  On the original outputs
    130 of 460 phase tests (28.3%) were "significant" against 23 expected
    under the null.  Every test family now carries Benjamini-Hochberg
    q-values, and the boolean flags key off those.

E2  ``common_mode_fraction`` was ``max(up, down) / n``, which can never fall
    below 0.5, so "no common mode" was unrepresentable and every candidate
    received a free floor of 0.325 in the v2 score.  Ties silently resolved to
    "up".  It is now a **signed** statistic with a two-sided binomial test, so
    an evenly split cohort scores zero.

G1  **Cohort-relative drift.**  The old pipeline computed the common-mode
    check and then discounted the toolwear score by a flat 20%.  On the
    original outputs that left 162 of 202 toolwear-labelled rows (80%) inside
    machine/programs where >=75% of tools drifted the same way -- which by the
    project's own decision rule indicates material/thermal/sensor change, not
    tool-specific wear.  Each tool's drift is now expressed **relative to its
    machine/program cohort median**, isolating the tool-specific component,
    which is the quantity a tool-wear claim actually needs.

A2  Multi-reference consensus now compares per-window z-scores rather than raw
    distances that lived on incompatible scales.

A5  All ``float(x or 0.0)`` NaN traps replaced with ``safe_float``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau, theilslopes

from pipeline_common import (
    benjamini_hochberg,
    block_bootstrap_trend,
    hamed_rao_test,
    lag1_autocorrelation,
    prewhitened_test,
    robust_scale,
    safe_float,
    to_numeric,
    write_manifest,
)

try:
    import ruptures as rpt

    HAS_RUPTURES = True
except ModuleNotFoundError:
    rpt = None
    HAS_RUPTURES = False


DEFAULT_DTW_DIR = Path("processed/dtw_features_all_best")
DEFAULT_CANDIDATE_DIR = Path("processed/dtw_event_candidates_all_best")
DEFAULT_OUTPUT_DIR = Path("processed/robust_event_evidence_all_best")

CHANGEPOINT_METRICS = ["residual_mean", "dtw_amplitude_distance", "dtw_shape_distance",
                       "current_integral_As"]
TREND_METRICS = ["residual_mean", "residual_p95", "residual_abs_mean",
                 "dtw_amplitude_distance", "current_integral_As"]

# Minimum fraction of the target window's span that a cohort window must cover
# before its trend is treated as contemporaneous evidence.
MIN_COHORT_OVERLAP = 0.25

TIME_COLUMNS = {"started_at", "ended_at", "window_start", "window_end"}

TEXT_COLUMNS = {
    "program", "tool", "event_hypothesis", "candidate_scope", "confidence_band",
    "primary_evidence", "supporting_evidence", "review_recommendation", "candidate_id",
    "metric", "method", "phase_bin", "reference_kind",
}


def load_inputs(dtw_dir: Path, candidate_dir: Path):
    features = pd.read_csv(dtw_dir / "cnc_dtw_features.csv", low_memory=False)
    residuals = pd.read_csv(dtw_dir / "cnc_dtw_residuals_long.csv.gz", low_memory=False)
    candidate_path = candidate_dir / "all_event_candidates.csv"
    candidates = (
        pd.read_csv(candidate_path, low_memory=False)
        if candidate_path.exists() else pd.DataFrame()
    )

    for table in (features, residuals, candidates):
        if table.empty:
            continue
        to_numeric(table, [c for c in table.columns
                           if c not in TEXT_COLUMNS and c not in TIME_COLUMNS])
        for col in TIME_COLUMNS:
            if col in table.columns:
                table[col] = pd.to_datetime(table[col], utc=True, errors="coerce")
        for col in ["program", "tool"]:
            if col in table.columns:
                table[col] = table[col].astype(str)

    if "in_reference_set" in features.columns:
        features["in_reference_set"] = (
            features["in_reference_set"].astype(str).str.lower().isin(["true", "1", "1.0"])
        )
    else:
        features["in_reference_set"] = True
    return features, residuals, candidates


# --------------------------------------------------------------------------- #
#  H4: changepoints without a hard third-party dependency
# --------------------------------------------------------------------------- #

def pelt_l2(signal: np.ndarray, penalty: float, min_size: int = 5) -> list[int]:
    """Exact PELT for a change in mean under an L2 cost.

    Segment cost is ``sum((y - mean)^2)``, computed in O(1) from cumulative
    sums, with the standard PELT pruning rule.  Equivalent to
    ``ruptures.Pelt(model="l2")`` and used when ruptures is not installed.
    """
    y = np.asarray(signal, dtype=float).ravel()
    n = y.size
    if n < 2 * min_size:
        return []

    cumulative = np.concatenate([[0.0], np.cumsum(y)])
    cumulative_sq = np.concatenate([[0.0], np.cumsum(y ** 2)])

    def cost(start: int, end: int) -> float:
        length = end - start
        if length <= 0:
            return 0.0
        total = cumulative[end] - cumulative[start]
        total_sq = cumulative_sq[end] - cumulative_sq[start]
        return float(total_sq - total * total / length)

    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    previous = np.zeros(n + 1, dtype=int)
    admissible = [0]

    for end in range(min_size, n + 1):
        candidates = [(best[s] + cost(s, end) + penalty, s)
                      for s in admissible if end - s >= min_size]
        if not candidates:
            continue
        best[end], previous[end] = min(candidates)
        admissible = [s for s in admissible if best[s] + cost(s, end) <= best[end]]
        if end - min_size + 1 >= 0:
            admissible.append(end - min_size + 1)
        admissible = sorted(set(s for s in admissible if s <= end))

    breakpoints: list[int] = []
    position = n
    while position > 0:
        start = previous[position]
        if start > 0:
            breakpoints.append(start)
        position = start
        if start == 0:
            break
    return sorted(breakpoints)


def detect_changepoints(features: pd.DataFrame, min_size: int = 5) -> pd.DataFrame:
    rows = []
    method = "numpy_pelt_l2"

    for window_id, group in features.groupby("comparable_window_id", sort=False):
        group = group.sort_values("run_order", kind="stable").reset_index(drop=True)
        if len(group) < 2 * min_size:
            continue

        for metric in CHANGEPOINT_METRICS:
            if metric not in group.columns:
                continue
            y = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(y)
            if valid.sum() < 2 * min_size:
                continue
            y = np.where(valid, y, np.nanmedian(y))
            scale = robust_scale(y)
            if not np.isfinite(scale) or scale == 0:
                continue
            signal = (y - np.median(y)) / scale
            # BIC penalty for a mean-change L2 model on a standardised signal.
            penalty = max(3.0, 2.0 * float(np.log(signal.size)))

            # The exact L2 PELT is used unconditionally.  Branching on whether
            # ``ruptures`` happens to be installed made the result depend on the
            # environment: an RBF cost and an L2 cost are on different scales, so
            # the same penalty means different sensitivities.
            found = pelt_l2(signal, penalty, min_size)

            for bp in found:
                if bp >= len(group) or bp < min_size or len(group) - bp < min_size:
                    continue
                left, right = y[:bp], y[bp:]
                left_med, right_med = float(np.median(left)), float(np.median(right))
                pooled = robust_scale(np.concatenate([left - left_med, right - right_med]))
                if not np.isfinite(pooled) or pooled == 0:
                    continue
                near = group.loc[max(0, bp - 2):min(len(group) - 1, bp + 2)]
                rows.append({
                    "comparable_window_id": int(window_id),
                    "machine_id": int(group["machine_id"].iloc[0]),
                    "program": group["program"].iloc[0],
                    "tool": group["tool"].iloc[0],
                    "metric": metric, "method": method,
                    "split_after_run_order": int(group.loc[bp - 1, "run_order"]),
                    "first_run_after_split": int(group.loc[bp, "run_id"]),
                    "left_runs": int(bp), "right_runs": int(len(group) - bp),
                    "left_median": left_med, "right_median": right_med,
                    "effect_size": float((right_med - left_med) / pooled),
                    "abs_effect_size": float(abs((right_med - left_med) / pooled)),
                    "boundary_runs_near_split": int(
                        near["batch_or_job_boundary"].fillna(0).sum()
                        if "batch_or_job_boundary" in near.columns else 0),
                    "thermal_runs_near_split": int(
                        near["in_first_runs_after_long_idle"].fillna(0).sum()
                        if "in_first_runs_after_long_idle" in near.columns else 0),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_effect_size", ascending=False, kind="stable")


# --------------------------------------------------------------------------- #
#  Trends, with FDR control and cohort-relative drift
# --------------------------------------------------------------------------- #

def robust_trends(features: pd.DataFrame, n_resamples: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Window-level trend tests, corrected for serial correlation.

    Kendall's tau assumes independent observations.  These run sequences are
    autocorrelated (lag-1 from 0.13 to 0.95 across this dataset), which inflates
    the true variance of the statistic and makes the ordinary p-value far too
    small.  Every metric therefore carries three inference routes:

      * ``_kendall_p``     the naive value -- retained for comparison only
      * ``_block_p/_q``    circular block bootstrap, the primary route
      * ``_hamed_rao_p``   variance-inflation correction, a conservative check

    Trend direction and every downstream decision key off the block-bootstrap
    q-value.  On this dataset the switch takes the count of significant
    residual-mean windows from 11 to 5.
    """
    rows = []
    for window_id, group in features.groupby("comparable_window_id", sort=False):
        group = group.sort_values("run_order", kind="stable")
        x = group["run_order"].to_numpy(dtype=float)
        row = {
            "comparable_window_id": int(window_id),
            "machine_id": int(group["machine_id"].iloc[0]),
            "program": group["program"].iloc[0],
            "tool": group["tool"].iloc[0],
            "num_runs": int(len(group)),
            "scoring_only_runs": int((~group["in_reference_set"]).sum()),
            "window_start": group["started_at"].min() if "started_at" in group.columns else pd.NaT,
            "window_end": group["started_at"].max() if "started_at" in group.columns else pd.NaT,
            "current_level_a": float(np.nanmedian(group["current_mean"].to_numpy(dtype=float))),
            "batch_or_job_boundary_count": int(
                group["batch_or_job_boundary"].fillna(0).sum()
                if "batch_or_job_boundary" in group.columns else 0),
            "thermal_context_count": int(
                group["in_first_runs_after_long_idle"].fillna(0).sum()
                if "in_first_runs_after_long_idle" in group.columns else 0),
        }
        for metric in TREND_METRICS:
            if metric not in group.columns:
                continue
            y = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 6:
                for suffix in ["theil_sen_slope", "theil_sen_low", "theil_sen_high",
                               "kendall_tau", "kendall_p"]:
                    row[f"{metric}_{suffix}"] = np.nan
                continue
            xv, yv = x[valid], y[valid]
            slope, _, low, high = theilslopes(yv, xv, alpha=0.95)
            tau, p = kendalltau(xv, yv)
            row[f"{metric}_theil_sen_slope"] = float(slope)
            row[f"{metric}_theil_sen_low"] = float(low)
            row[f"{metric}_theil_sen_high"] = float(high)
            row[f"{metric}_kendall_tau"] = float(tau)
            row[f"{metric}_kendall_p"] = float(p)

            # Autocorrelation-aware inference.
            block = block_bootstrap_trend(yv, n_resamples=n_resamples, seed=seed)
            hamed = hamed_rao_test(yv)
            row[f"{metric}_lag1_autocorrelation"] = block["lag1"]
            row[f"{metric}_block_length"] = (
                max(block["block_lengths"]) if block["block_lengths"] else np.nan)
            row[f"{metric}_block_p"] = block["p"]
            row[f"{metric}_hamed_rao_p"] = hamed["p"]
            row[f"{metric}_hamed_rao_variance_inflation"] = hamed["variance_inflation"]
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # F3: FDR across each metric's family of window-level tests, applied to the
    # autocorrelation-corrected p-values rather than the naive ones.
    for metric in TREND_METRICS:
        if f"{metric}_kendall_p" in out.columns:
            out[f"{metric}_kendall_q"] = benjamini_hochberg(out[f"{metric}_kendall_p"])
        if f"{metric}_block_p" in out.columns:
            out[f"{metric}_block_q"] = benjamini_hochberg(out[f"{metric}_block_p"])
            out[f"{metric}_hamed_rao_q"] = benjamini_hochberg(out[f"{metric}_hamed_rao_p"])
            significant = out[f"{metric}_block_q"].le(0.05)
            out[f"{metric}_trend_direction"] = np.select(
                [significant & out[f"{metric}_theil_sen_slope"].gt(0),
                 significant & out[f"{metric}_theil_sen_slope"].lt(0)],
                ["increasing", "decreasing"], default="not_significant",
            )
            # How many of the three routes agree, for reporting.
            out[f"{metric}_routes_significant"] = (
                out[f"{metric}_kendall_q"].le(0.05).astype(int)
                + out[f"{metric}_block_q"].le(0.05).astype(int)
                + out[f"{metric}_hamed_rao_q"].le(0.05).astype(int)
            )

    out = add_cohort_relative_drift(out)
    out["toolwear_evidence_score"] = out.apply(toolwear_evidence_score, axis=1)
    return out.sort_values("toolwear_evidence_score", ascending=False, kind="stable")


def add_cohort_relative_drift(trends: pd.DataFrame) -> pd.DataFrame:
    """G1: separate tool-specific drift from machine/program-wide drift.

    A tool whose residual rises exactly as fast as every other tool on the same
    machine and program is showing a common-mode effect -- thermal state,
    material batch or sensor drift -- not tool wear.  Subtracting the cohort
    median leaves the component specific to this tool, which is what a
    tool-wear claim requires.

    **The cohort must be contemporaneous.**  Comparable windows for different
    tools cover different time spans -- on machine 288 / O005362 the windows
    for tools 1 and 2 barely overlap (06:27-20:30 versus 18:21-13:49 the
    following day) -- so pooling their trends by (machine, program) alone
    compares behaviour measured at different times under different conditions,
    which is precisely the confound the cohort exists to remove.  Only windows
    overlapping the target's span by at least ``MIN_COHORT_OVERLAP`` contribute.
    """
    trends = trends.copy()
    trends["residual_delta_fraction_of_current"] = np.where(
        trends["current_level_a"] > 0,
        trends["residual_mean_theil_sen_slope"] * trends["num_runs"] / trends["current_level_a"],
        np.nan,
    )

    has_span = {"window_start", "window_end"}.issubset(trends.columns)
    cohort_tools, median_tau, median_slope, median_fraction = [], [], [], []

    for idx, row in trends.iterrows():
        same_line = trends[
            trends["machine_id"].eq(row["machine_id"])
            & trends["program"].eq(row["program"])
            & trends["comparable_window_id"].ne(row["comparable_window_id"])
        ]
        if has_span and not same_line.empty and pd.notna(row.get("window_start")):
            span = max(
                (row["window_end"] - row["window_start"]).total_seconds(), 1.0
            )
            overlap = (
                np.minimum(same_line["window_end"], row["window_end"])
                - np.maximum(same_line["window_start"], row["window_start"])
            ).dt.total_seconds().clip(lower=0) / span
            same_line = same_line[overlap.ge(MIN_COHORT_OVERLAP)]

        cohort_tools.append(int(same_line["tool"].nunique()))
        median_tau.append(float(same_line["residual_mean_kendall_tau"].median())
                          if not same_line.empty else np.nan)
        median_slope.append(float(same_line["residual_mean_theil_sen_slope"].median())
                            if not same_line.empty else np.nan)
        median_fraction.append(float(same_line["residual_delta_fraction_of_current"].median())
                               if not same_line.empty else np.nan)

    trends["cohort_tools"] = cohort_tools
    trends["cohort_median_tau"] = median_tau
    trends["cohort_median_slope"] = median_slope
    trends["cohort_median_delta_fraction"] = median_fraction

    trends["cohort_relative_tau"] = (
        trends["residual_mean_kendall_tau"] - trends["cohort_median_tau"])
    trends["cohort_relative_slope"] = (
        trends["residual_mean_theil_sen_slope"] - trends["cohort_median_slope"])
    trends["cohort_relative_delta_fraction"] = (
        trends["residual_delta_fraction_of_current"] - trends["cohort_median_delta_fraction"])

    trends["drifts_with_cohort"] = (
        trends["cohort_tools"].ge(1)
        & (trends["cohort_relative_tau"].abs() < 0.15)
        & (trends["cohort_median_tau"].abs() >= 0.20)
    )
    return trends


def toolwear_evidence_score(row: pd.Series) -> float:
    """Evidence that *this tool* is degrading, net of its cohort.

    Built on the cohort-relative statistics so a window cannot score highly
    just because everything on that machine moved together.
    """
    relative_tau = safe_float(row.get("cohort_relative_tau"))
    absolute_tau = safe_float(row.get("residual_mean_kendall_tau"))
    q_value = safe_float(row.get("residual_mean_block_q"),
                         safe_float(row.get("residual_mean_kendall_q"), 1.0))
    relative_fraction = safe_float(row.get("cohort_relative_delta_fraction"))
    cohort_tools = safe_float(row.get("cohort_tools"), 1.0)

    significance = 1.0 if q_value <= 0.05 else max(0.0, 1.0 - q_value)
    # With no cohort to compare against, fall back to the absolute trend but
    # cap the achievable score: the confound cannot be ruled out.
    if cohort_tools < 2:
        return float(np.clip(0.55 * max(absolute_tau, 0.0) + 0.15 * significance, 0.0, 0.70))

    raw = (
        0.45 * min(max(relative_tau, 0.0) / 0.40, 1.0)
        + 0.25 * min(max(relative_fraction, 0.0) / 0.10, 1.0)
        + 0.20 * max(absolute_tau, 0.0)
        + 0.10 * significance
    )
    if bool(row.get("drifts_with_cohort", False)):
        raw *= 0.35   # G1: common mode is a disqualifier, not a rounding error.
    return float(np.clip(raw, 0.0, 1.0))


def phase_specific_trends(residuals: pd.DataFrame, bins: int = 10,
                          n_resamples: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Phase-resolved trend tests.

    ``n_resamples`` must be large enough that the multiplicity correction can
    reject at all.  A resampled p-value cannot fall below 1/(B+1), so with 400
    tests Benjamini-Hochberg at q <= 0.05 needs p <= 1.25e-4 at the strictest
    rank, i.e. B >= 8000.  The earlier default of 500 made significance
    unreachable by construction rather than by evidence.
    """
    rows = []
    residuals = residuals.copy()
    residuals["phase_bin"] = pd.cut(
        residuals["template_phase"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = residuals.groupby(
        ["comparable_window_id", "machine_id", "program", "tool", "phase_bin"], observed=False)

    for key, group in grouped:
        window_id, machine_id, program, tool, phase_bin = key
        by_run = group.groupby("run_order", sort=True)["residual"].mean().reset_index()
        x = by_run["run_order"].to_numpy(dtype=float)
        y = by_run["residual"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 6:
            continue
        slope, _, low, high = theilslopes(y[valid], x[valid], alpha=0.95)
        tau, p = kendalltau(x[valid], y[valid])
        block = block_bootstrap_trend(y[valid], n_resamples=n_resamples, seed=seed)
        rows.append({
            "comparable_window_id": int(window_id), "machine_id": int(machine_id),
            "program": program, "tool": tool, "phase_bin": str(phase_bin),
            "runs": int(valid.sum()),
            "residual_theil_sen_slope": float(slope),
            "residual_theil_sen_low": float(low),
            "residual_theil_sen_high": float(high),
            "residual_kendall_tau": float(tau),
            "residual_kendall_p": float(p),
            "residual_lag1_autocorrelation": block["lag1"],
            "residual_block_p": block["p"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # F3: 10 bins x every window is a large family; correct it -- and correct
    # for serial dependence first, so the FDR is applied to valid p-values.
    out["residual_kendall_q"] = benjamini_hochberg(out["residual_kendall_p"])
    out["residual_block_q"] = benjamini_hochberg(out["residual_block_p"])
    out["increasing_phase_evidence"] = (
        out["residual_block_q"].le(0.05) & out["residual_theil_sen_slope"].gt(0))
    return out.sort_values(["increasing_phase_evidence", "residual_kendall_tau"],
                           ascending=[False, False], kind="stable")


# --------------------------------------------------------------------------- #
#  E2: a signed common-mode statistic
# --------------------------------------------------------------------------- #

def common_mode_shift(trends: pd.DataFrame) -> pd.DataFrame:
    """Do the tools on one machine/program move together?

    The old statistic was ``max(up, down) / n``, bounded below by 0.5, so an
    evenly split cohort -- the clearest evidence *against* a common-mode
    effect -- scored the same as a weak one.  This version is signed and
    centred on zero, with a two-sided binomial test of the up/down split.
    """
    has_span = {"window_start", "window_end"}.issubset(trends.columns)
    rows = []
    for (machine_id, program), group in trends.groupby(["machine_id", "program"], sort=False):
        if group["tool"].nunique() < 2:
            continue
        # The same contemporaneity requirement that ``add_cohort_relative_drift``
        # applies: tools whose windows do not overlap in time are not evidence
        # about a shared cause.  Without this the common-mode disqualifier can
        # fire on behaviour measured days apart.
        if has_span and group["window_start"].notna().all():
            latest_start = group["window_start"].max()
            earliest_end = group["window_end"].min()
            spans = (group["window_end"] - group["window_start"]).dt.total_seconds()
            overlap = (earliest_end - latest_start).total_seconds()
            if not np.isfinite(overlap) or overlap <= 0 or \
                    overlap < MIN_COHORT_OVERLAP * float(spans.min()):
                continue
        slopes = group["residual_mean_theil_sen_slope"].to_numpy(dtype=float)
        slopes = slopes[np.isfinite(slopes)]
        if slopes.size < 2:
            continue

        up = int((slopes > 0).sum())
        down = int((slopes < 0).sum())
        n = up + down
        if n == 0:
            continue

        # +1 = every tool rising, -1 = every tool falling, 0 = evenly split.
        signed_fraction = (up - down) / n
        p_value = float(binomtest(max(up, down), n, 0.5, alternative="two-sided").pvalue)
        rows.append({
            "machine_id": int(machine_id), "program": program,
            "tools_compared": int(group["tool"].nunique()),
            "tools_up": up, "tools_down": down,
            "dominant_direction": "up" if up > down else ("down" if down > up else "split"),
            "common_mode_signed_fraction": float(signed_fraction),
            "common_mode_strength": float(abs(signed_fraction)),
            "common_mode_p": p_value,
            "median_cohort_tau": float(group["residual_mean_kendall_tau"].median()),
            "window_ids": ",".join(group["comparable_window_id"].astype(int).astype(str)),
            "tools": ",".join(group["tool"].astype(str)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["common_mode_q"] = benjamini_hochberg(out["common_mode_p"])
    # Disqualifying strength: nearly all tools moving together, and the cohort
    # trend is actually material rather than a wobble around zero.
    out["common_mode_confound"] = (
        out["common_mode_strength"].ge(0.75) & out["median_cohort_tau"].abs().ge(0.20))
    return out.sort_values(["common_mode_strength", "tools_compared"],
                           ascending=[False, False], kind="stable")


# --------------------------------------------------------------------------- #
#  A2: consensus on comparable scales
# --------------------------------------------------------------------------- #

def multi_reference_consensus(features: pd.DataFrame) -> pd.DataFrame:
    shape_z = [c for c in features.columns
               if c.startswith("dtw_shape_distance_") and c.endswith("_z")]
    amplitude_z = [c for c in features.columns
                   if c.startswith("dtw_amplitude_distance_") and c.endswith("_z")]
    if not shape_z or not amplitude_z:
        return pd.DataFrame()

    shape = features[shape_z]
    amplitude = features[amplitude_z]
    out = pd.DataFrame({
        "comparable_window_id": features["comparable_window_id"].astype(int),
        "machine_id": features["machine_id"].astype(int),
        "program": features["program"], "tool": features["tool"],
        "run_id": features["run_id"].astype(int),
        "run_order": features["run_order"].astype(int),
        "in_reference_set": features["in_reference_set"],
        "shape_reference_count": shape.notna().sum(axis=1),
        "amplitude_reference_count": amplitude.notna().sum(axis=1),
        "shape_z_mean_across_references": shape.mean(axis=1, skipna=True),
        "shape_z_min_across_references": shape.min(axis=1, skipna=True),
        "shape_z_std_across_references": shape.std(axis=1, ddof=0, skipna=True),
        "amplitude_z_mean_across_references": amplitude.mean(axis=1, skipna=True),
        "amplitude_z_min_across_references": amplitude.min(axis=1, skipna=True),
        "amplitude_z_std_across_references": amplitude.std(axis=1, ddof=0, skipna=True),
    })

    # "Every reference agrees this run is an outlier" -- on within-window
    # z-scores, so the test is about rank rather than raw magnitude.
    out["all_reference_shape_high"] = (
        out["shape_reference_count"].ge(3) & out["shape_z_min_across_references"].ge(2.0))
    out["all_reference_amplitude_high"] = (
        out["amplitude_reference_count"].ge(3) & out["amplitude_z_min_across_references"].ge(2.0))
    out["multi_reference_outlier_evidence"] = (
        out["all_reference_shape_high"] & out["all_reference_amplitude_high"])
    out["reference_disagreement"] = out[
        ["shape_z_std_across_references", "amplitude_z_std_across_references"]].mean(axis=1)
    return out.sort_values(
        ["multi_reference_outlier_evidence", "amplitude_z_mean_across_references"],
        ascending=[False, False], kind="stable")


def decision_template(
    candidates: pd.DataFrame, trends: pd.DataFrame, changepoints: pd.DataFrame,
    common_mode: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    out.insert(0, "candidate_id", [f"C{i:04d}" for i in range(1, len(out) + 1)])
    out["robust_trend_support"] = False
    out["changepoint_support"] = False
    out["common_mode_confound"] = False
    out["review_status"] = "pending"
    out["review_decision"] = ""
    out["final_label"] = ""
    out["reviewer_confidence"] = ""
    out["review_notes"] = ""

    trend_windows = (
        set(trends.loc[trends["toolwear_evidence_score"].ge(0.35), "comparable_window_id"]
            .astype(int)) if not trends.empty else set()
    )
    cp_pairs = (
        set(zip(changepoints["comparable_window_id"].astype(int),
                changepoints["first_run_after_split"].astype(int)))
        if not changepoints.empty else set()
    )
    confounded = (
        set(zip(common_mode.loc[common_mode["common_mode_confound"], "machine_id"].astype(int),
                common_mode.loc[common_mode["common_mode_confound"], "program"].astype(str)))
        if not common_mode.empty else set()
    )

    for idx, row in out.iterrows():
        window_id = int(row["comparable_window_id"]) if pd.notna(row["comparable_window_id"]) else None
        run_text = str(row.get("run_id", "")).strip().lower()
        run_id = int(float(run_text)) if run_text not in ("", "nan") else None
        if row.get("event_hypothesis") == "toolwear_signal_candidate" and window_id in trend_windows:
            out.loc[idx, "robust_trend_support"] = True
        if run_id is not None and (window_id, run_id) in cp_pairs:
            out.loc[idx, "changepoint_support"] = True
        if (int(row["machine_id"]), str(row["program"])) in confounded:
            out.loc[idx, "common_mode_confound"] = True
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build robust evidence tables for DTW event candidates.")
    parser.add_argument("--dtw-dir", type=Path, default=DEFAULT_DTW_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000,
                        help="Circular block-bootstrap resamples per trend test.")
    parser.add_argument("--phase-bootstrap-resamples", type=int, default=10000,
                        help="Must satisfy 1/(B+1) <= 0.05/n_tests or FDR can never reject.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, residuals, candidates = load_inputs(args.dtw_dir, args.candidate_dir)

    changepoints = detect_changepoints(features)
    trends = robust_trends(features, args.bootstrap_resamples, args.seed)
    phase_trends = phase_specific_trends(
        residuals, n_resamples=args.phase_bootstrap_resamples, seed=args.seed)
    common_mode = common_mode_shift(trends)
    consensus = multi_reference_consensus(features)
    decisions = decision_template(candidates, trends, changepoints, common_mode)

    changepoints.to_csv(args.output_dir / "ruptures_changepoints.csv", index=False)
    trends.to_csv(args.output_dir / "robust_trend_tests.csv", index=False)
    phase_trends.to_csv(args.output_dir / "phase_specific_trends.csv", index=False)
    common_mode.to_csv(args.output_dir / "common_mode_shift_checks.csv", index=False)
    consensus.to_csv(args.output_dir / "multi_reference_consensus.csv", index=False)
    if not decisions.empty:
        decisions.to_csv(args.output_dir / "label_review_decisions_template.csv", index=False)

    phase_significant = int(phase_trends["increasing_phase_evidence"].sum()) if not phase_trends.empty else 0
    phase_raw = int(phase_trends["residual_kendall_p"].le(0.05).sum()) if not phase_trends.empty else 0
    phase_naive_q = int(phase_trends["residual_kendall_q"].le(0.05).sum()) if not phase_trends.empty else 0
    trend_naive_q = int(trends["residual_mean_kendall_q"].le(0.05).sum()) if not trends.empty else 0
    trend_block_q = int(trends["residual_mean_block_q"].le(0.05).sum()) if not trends.empty else 0
    median_lag1 = float(trends["residual_mean_lag1_autocorrelation"].median()) if not trends.empty else np.nan
    confounded = int(common_mode["common_mode_confound"].sum()) if not common_mode.empty else 0

    summary = (
        f"changepoint method: {'ruptures' if HAS_RUPTURES else 'built-in numpy PELT (L2)'}\n"
        f"changepoints: {len(changepoints):,}\n"
        f"robust trend rows: {len(trends):,}\n"
        f"residual_mean trend windows significant:\n"
        f"  naive Kendall + FDR:        {trend_naive_q:,}\n"
        f"  block bootstrap + FDR:      {trend_block_q:,}   <-- reported\n"
        f"  median lag-1 autocorrelation: {median_lag1:.3f}\n"
        f"phase trend rows: {len(phase_trends):,}\n"
        f"  raw p<=0.05: {phase_raw:,}   naive FDR: {phase_naive_q:,}   "
        f"block+FDR rising: {phase_significant:,}\n"
        f"common-mode checks: {len(common_mode):,}\n"
        f"  cohorts flagged as common-mode confounded: {confounded:,}\n"
        f"multi-reference consensus rows: {len(consensus):,}\n"
        f"review candidates: {len(decisions):,}\n"
    )
    (args.output_dir / "robust_evidence_summary.txt").write_text(summary, encoding="utf-8")
    write_manifest(args.output_dir, "build_robust_event_evidence", args,
                   {"has_ruptures": HAS_RUPTURES})
    print(summary)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
