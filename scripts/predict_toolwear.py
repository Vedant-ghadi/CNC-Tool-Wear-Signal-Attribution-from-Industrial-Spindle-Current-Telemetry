"""Cohort-relative degradation forecasting with permutation-calibrated alarms.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
This dataset contains **no tool-wear ground truth**: no failure events, no
tool-change log, and the only ``TOOL LIFE OVER`` alarm fires ~30x/day on the
one machine with no tool numbers, with cutting current unchanged either side
(8.65 A -> 8.65 A, -0.08%).  Across 51 tool sequences with >=20 runs there are
only three sustained downward current steps that could be a fresh cutting edge,
so there are no observable tool-life cycles either.

Consequently this module does **not** estimate Remaining Useful Life and does
not classify wear state.  Both would require labels that do not exist, and
training on the pipeline's own algorithmic labels would only reproduce the
heuristic that generated them.

What it does instead is defensible without labels:

  1. HEALTH INDEX     A cohort-relative deviation: how far this tool's cutting
                      current sits from the *other tools running on the same
                      machine and program at the same time*.  Thermal state,
                      material batch and sensor drift are shared by target and
                      cohort, so they cancel -- the same logic that took
                      common-mode contamination from 80% to 0%.

  2. DEGRADATION      A robust (Theil-Sen) trajectory against **cumulative
                      cutting seconds**, not run index.  Run duration varies
                      21 s - 221 s in this data, so run number is a badly
                      wrong proxy for material removed.

  3. THRESHOLD        Two limits, each calibrated empirically:
                      * Trend-detector limit by **permutation** -- shuffling run
                        order preserves the health-index distribution but
                        destroys dependence on position, which is exactly the
                        null "this tool is not degrading".  The limit that
                        yields the target false-alarm rate falls straight out.
                      * Health-index level limit by **cohort quantile** -- the
                        level exceeded by less than alpha of healthy cohort
                        tool-runs, computed leave-one-out.

  4. HORIZON          Projected crossing of the level limit with a bootstrap
                      interval, reported in runs and cutting minutes.  Named
                      "time to alarm threshold", never "remaining useful life":
                      the threshold is calibrated against the cohort, not
                      against a validated end of tool life.

  5. EVALUATION       Prequential forecast error against a persistence
                      baseline, the permutation null check, and a
                      leave-one-tool-out sweep.  All measurable without labels.

Usage
-----
    python scripts/predict_toolwear.py --machine 288 --program O005362 --tool 3
    python scripts/predict_toolwear.py --machine 288 --program O005362 --tool 3 \
        --target-false-alarm 0.05 --horizon-k 5
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, theilslopes

from pipeline_common import (
    circular_block_resample,
    estimate_block_length,
    block_length_grid,
    hamed_rao_test,
    lag1_autocorrelation,
    prewhitened_test,
    robust_scale,
    safe_float,
    write_manifest,
)

try:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


DEFAULT_INPUT = Path("processed/cnc_run_tool_features.csv")
DEFAULT_OUTPUT_DIR = Path("processed/toolwear_prediction")

# Features whose increase is the classical spindle-current wear signature.
# These form the signed, directional health index.
DIRECTIONAL_FEATURES = ["current_mean", "current_median", "current_p95"]
# Additional features used only for the omnidirectional (Mahalanobis) index.
CONTEXT_FEATURES = ["current_integral_As", "current_std", "cutting_s"]

MIN_COHORT_RUNS = 20
MIN_COHORT_CUTTING_S = 10.0
MIN_TIME_OVERLAP = 0.5


# --------------------------------------------------------------------------- #
#  Loading and cohort selection
# --------------------------------------------------------------------------- #

def load_sequences(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    df["tool"] = df["tool"].astype(str)
    df["program"] = df["program"].astype(str)
    for col in DIRECTIONAL_FEATURES + CONTEXT_FEATURES + ["machine_id", "run_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["analysis_ready_run", "alarm_active", "is_mdi", "in_first_runs_after_long_idle"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    return df.sort_values("started_at", kind="stable")


def select_cohort(
    data: pd.DataFrame, machine_id: int, program: str, tool: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    """Target sequence plus the tools it will be measured against.

    A cohort tool must do comparable work (enough runs, real cutting time) and
    must actually be running at the same time as the target -- otherwise it
    cannot absorb the shared thermal and material conditions that the
    cohort-relative construction exists to cancel.
    """
    scope = data[
        data["machine_id"].eq(machine_id)
        & data["program"].eq(program)
        & data["analysis_ready_run"]
    ].copy()
    if scope.empty:
        raise SystemExit(f"No analysis-ready runs for machine {machine_id} / {program}")

    target = scope[scope["tool"].eq(tool)].sort_values("started_at").reset_index(drop=True)
    if len(target) < MIN_COHORT_RUNS:
        raise SystemExit(f"Target tool {tool} has only {len(target)} runs")

    start, end = target["started_at"].min(), target["started_at"].max()
    cohort, rejected = {}, []
    for other, group in scope[scope["tool"].ne(tool)].groupby("tool", sort=True):
        group = group.sort_values("started_at").reset_index(drop=True)
        overlap = float(group["started_at"].between(start, end).mean())
        median_cut = float(group["cutting_s"].median())
        if len(group) < MIN_COHORT_RUNS:
            rejected.append(f"{other}: only {len(group)} runs")
        elif median_cut < MIN_COHORT_CUTTING_S:
            # 2-second "runs" are probe touches, not comparable cutting work.
            rejected.append(f"{other}: median cutting {median_cut:.0f}s below floor")
        elif overlap < MIN_TIME_OVERLAP:
            rejected.append(f"{other}: only {overlap:.0%} time overlap")
        else:
            cohort[str(other)] = group
    return target, cohort, rejected


# --------------------------------------------------------------------------- #
#  Stage 1 — cohort-relative health index
# --------------------------------------------------------------------------- #

def normalise_to_own_median(series: pd.Series) -> tuple[np.ndarray, float]:
    """Express a feature as a fraction of that tool's own median.

    Normalising by the whole-sequence median rather than by an early baseline
    is deliberate: anchoring on the first N runs is what made the original
    pipeline's residuals rise with run order regardless of what the tool was
    doing (audit finding D1).  A whole-sequence median centres the series
    without imposing a direction on it.
    """
    values = series.to_numpy(dtype=float)
    centre = float(np.nanmedian(values))
    if not np.isfinite(centre) or centre == 0:
        return np.full_like(values, np.nan), np.nan
    return values / centre, centre


def interpolate_to_timeline(
    source_times: pd.Series, source_values: np.ndarray, target_times: pd.Series,
) -> np.ndarray:
    """Put a cohort tool's series on the target's run timestamps."""
    x = source_times.astype("int64").to_numpy() / 1e9
    y = np.asarray(source_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.full(len(target_times), np.nan)
    order = np.argsort(x[valid])
    xs, ys = x[valid][order], y[valid][order]
    xt = target_times.astype("int64").to_numpy() / 1e9
    # Clamp at the ends rather than returning NaN: a target run just outside a
    # cohort tool's span is still best served by that tool's nearest value, and
    # NaN there would silently drop the whole cohort row from the median.
    return np.interp(xt, xs, ys)


def build_health_index(
    target: pd.DataFrame, cohort: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Signed directional index plus an omnidirectional Mahalanobis distance."""
    columns = list(dict.fromkeys(
        ["run_id", "tool", "started_at", "cutting_s"] + DIRECTIONAL_FEATURES + CONTEXT_FEATURES
    ))
    out = target[columns].copy().reset_index(drop=True)
    out["run_order"] = np.arange(1, len(out) + 1)
    out["cumulative_cutting_s"] = out["cutting_s"].cumsum() - out["cutting_s"]

    deviations = {}
    for feature in DIRECTIONAL_FEATURES + CONTEXT_FEATURES:
        target_norm, target_centre = normalise_to_own_median(out[feature])
        out[f"{feature}_norm"] = target_norm

        cohort_stack = []
        for name, group in cohort.items():
            group_norm, _ = normalise_to_own_median(group[feature])
            cohort_stack.append(
                interpolate_to_timeline(group["started_at"], group_norm, out["started_at"])
            )
        if cohort_stack:
            with np.errstate(all="ignore"):
                cohort_median = np.nanmedian(np.vstack(cohort_stack), axis=0)
        else:
            cohort_median = np.ones(len(out))

        out[f"{feature}_cohort"] = cohort_median
        deviation = target_norm - cohort_median
        out[f"{feature}_deviation"] = deviation
        deviations[feature] = deviation

    # Signed health index: mean fractional excess over the cohort across the
    # directional features. Units are "fraction of this tool's own median
    # current", so +0.05 means 5% above where the cohort says it should be.
    directional = np.vstack([deviations[f] for f in DIRECTIONAL_FEATURES])
    with np.errstate(all="ignore"):
        out["health_index"] = np.nanmean(directional, axis=0)

    out["health_index_mahalanobis"] = mahalanobis_index(
        np.vstack([deviations[f] for f in DIRECTIONAL_FEATURES + CONTEXT_FEATURES]).T
    )
    return out


def mahalanobis_index(matrix: np.ndarray) -> np.ndarray:
    """Robust multivariate distance from the in-control centre."""
    finite_rows = np.all(np.isfinite(matrix), axis=1)
    if finite_rows.sum() < matrix.shape[1] + 2:
        return np.full(len(matrix), np.nan)
    clean = matrix[finite_rows]
    try:
        from sklearn.covariance import MinCovDet

        model = MinCovDet(random_state=0).fit(clean)
        centre, precision = model.location_, np.linalg.pinv(model.covariance_)
    except Exception:
        centre = np.median(clean, axis=0)
        precision = np.linalg.pinv(np.cov(clean, rowvar=False) + np.eye(clean.shape[1]) * 1e-9)

    out = np.full(len(matrix), np.nan)
    delta = matrix[finite_rows] - centre
    out[finite_rows] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0.0))
    return out


# --------------------------------------------------------------------------- #
#  Stage 3 — CUSUM and its permutation-calibrated limit
# --------------------------------------------------------------------------- #

def short_term_sigma(values: np.ndarray) -> float:
    """Noise scale from first differences.

    Using the raw spread would absorb any trend into the noise estimate and
    make the detector blind to exactly what it is looking for.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    return float(robust_scale(np.diff(values)) / np.sqrt(2.0))


def sequential_kendall_tau(values: np.ndarray, warmup: int) -> np.ndarray:
    """Kendall's tau recomputed on x_1..x_t for every t -- the onset curve."""
    values = np.asarray(values, dtype=float)
    n = values.size
    out = np.full(n, np.nan)
    for t in range(warmup, n):
        y = values[:t + 1]
        mask = np.isfinite(y)
        if mask.sum() > 5:
            out[t] = kendalltau(np.arange(t + 1)[mask], y[mask]).statistic
    return out


def block_bootstrap_test(
    values: np.ndarray, block: int, n_resamples: int, seed: int,
) -> dict[str, float]:
    """One-sided block-bootstrap p-value for the total Kendall tau.

    One-sided because wear is directional: a tool cutting *more easily* over
    time is not wearing out.  A two-sided test alarmed on cohort tool 1, whose
    tau is -0.325.
    """
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 20:
        return {"tau": np.nan, "p": np.nan, "block_length": block, "null_95": np.nan}
    observed = float(kendalltau(np.arange(y.size), y).statistic)
    rng = np.random.default_rng(seed)
    null = np.array([
        kendalltau(np.arange(y.size), circular_block_resample(y, block, rng)).statistic
        for _ in range(n_resamples)
    ])
    return {"tau": observed,
            "p": float((np.sum(null >= observed) + 1) / (n_resamples + 1)),
            "block_length": int(block), "null_95": float(np.quantile(null, 0.95))}


def block_length_grid(n: int, acf_block: int) -> list[int]:
    """Several defensible block lengths, so the result is not one arbitrary choice.

    Block-bootstrap p-values are sensitive to block length, and no single rule
    is authoritative: too short and serial dependence leaks into the null, too
    long and there are too few blocks to resample.  Reporting the whole grid --
    and alarming on the least favourable member of it -- is more honest than
    picking one and hoping.
    """
    candidates = {
        max(2, int(round(n ** (1 / 3)))),
        max(2, int(acf_block)),
        max(2, n // 10),
        max(2, n // 6),
    }
    return sorted(int(np.clip(b, 2, max(3, n // 4))) for b in candidates)


def bootstrap_power(
    values: np.ndarray, block: int, ramp: float, alpha: float,
    n_trials: int, n_resamples: int, seed: int,
) -> float:
    """Can the block bootstrap detect a real trend of this size in this series?

    A test that never rejects is not evidence of absence, so the conservative
    null needs its power reported alongside its p-value.  Synthetic series are
    built by block-resampling the observed data -- same noise, same
    autocorrelation, no trend -- and adding a known linear ramp.
    """
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 20:
        return np.nan
    rng = np.random.default_rng(seed + 7)
    detected = 0
    for _ in range(n_trials):
        synthetic = circular_block_resample(y, block, rng) + np.linspace(0, ramp, y.size)
        observed = kendalltau(np.arange(y.size), synthetic).statistic
        null = np.array([
            kendalltau(np.arange(y.size),
                       circular_block_resample(synthetic, block, rng)).statistic
            for _ in range(n_resamples)
        ])
        # One-sided, matching ``block_bootstrap_test``: reporting power for a
        # two-sided rule understates the detector actually in use.
        p_value = float((np.sum(null >= observed) + 1) / (n_resamples + 1))
        detected += p_value < alpha
    return float(detected / n_trials)


def detect_trend(
    values: np.ndarray, warmup: int, alpha: float, n_resamples: int, seed: int,
) -> dict:
    """Run every inference route and require the conservative one to agree.

    The naive Kendall p-value on these series is optimistic by roughly twelve
    orders of magnitude, so a single test would be indefensible.  The alarm
    fires on the block bootstrap -- which preserves the observed
    autocorrelation -- and the other routes are reported so a reader can see
    the spread.
    """
    y = np.asarray(values, dtype=float)
    finite = y[np.isfinite(y)]
    acf_block = estimate_block_length(finite)
    grid = block_length_grid(finite.size, acf_block)

    naive_tau, naive_p = kendalltau(np.arange(finite.size), finite)
    by_block = {b: block_bootstrap_test(finite, b, n_resamples, seed) for b in grid}
    p_values = [r["p"] for r in by_block.values() if np.isfinite(r["p"])]
    # Alarm on the least favourable block length, not the friendliest one.
    worst_p = float(max(p_values)) if p_values else np.nan
    reference = by_block.get(acf_block) or next(iter(by_block.values()))

    hamed = hamed_rao_test(finite)
    prewhite = prewhitened_test(finite)

    # Wear is directional: a falling health index is not degradation.
    rising = bool(naive_tau > 0)
    alarm = bool(rising and np.isfinite(worst_p) and worst_p <= alpha)

    sequential = sequential_kendall_tau(y, warmup)
    onset = np.nan
    threshold = reference.get("null_95", np.nan)
    if alarm and np.isfinite(threshold):
        crossing = np.flatnonzero(np.nan_to_num(sequential, nan=-np.inf) >= threshold)
        onset = int(crossing[0] + 1) if crossing.size else np.nan

    routes_significant = sum([
        bool(rising and np.isfinite(worst_p) and worst_p <= alpha),
        bool(rising and np.isfinite(hamed["p"]) and hamed["p"] <= alpha),
        bool(rising and np.isfinite(prewhite["p"]) and prewhite["p"] <= alpha),
    ])
    return {
        "lag1_autocorrelation": lag1_autocorrelation(finite),
        "block_length": acf_block,
        "block_length_grid": grid,
        "block_bootstrap_p_by_length": {int(b): r["p"] for b, r in by_block.items()},
        "block_bootstrap_null_95": threshold,
        "rising": rising,
        "naive_kendall_tau": float(naive_tau),
        "naive_kendall_p": float(naive_p),
        "block_bootstrap_p": worst_p,
        "hamed_rao_z": hamed["z"],
        "hamed_rao_p": hamed["p"],
        "hamed_rao_variance_inflation": hamed["variance_inflation"],
        "prewhitened_tau": prewhite["tau"],
        "prewhitened_p": prewhite["p"],
        "routes_significant_of_3": routes_significant,
        "alarm": alarm,
        "onset_run_order": onset,
        "sequential_tau": sequential,
    }


def cohort_level_limit(
    target_hi: np.ndarray, cohort_his: dict[str, np.ndarray], alpha: float,
) -> tuple[float, int]:
    """Health-index level exceeded by fewer than ``alpha`` of cohort tool-runs.

    The permutation null calibrates the *detector*, which depends on ordering.
    It cannot calibrate a level, because shuffling leaves the marginal
    distribution untouched.  The healthy population therefore has to come from
    somewhere else -- and the cohort is exactly that: other tools, same machine,
    same programme, same period, each scored leave-one-out against the rest.
    """
    pooled = np.concatenate([
        values[np.isfinite(values)] for values in cohort_his.values()
    ]) if cohort_his else np.array([])
    if pooled.size < 20:
        finite = target_hi[np.isfinite(target_hi)]
        return (float(np.quantile(finite, 1.0 - alpha)) if finite.size else np.nan), 0
    return float(np.quantile(pooled, 1.0 - alpha)), int(pooled.size)


# --------------------------------------------------------------------------- #
#  Stage 2 & 4 — degradation trajectory and horizon
# --------------------------------------------------------------------------- #

@dataclass
class Degradation:
    slope_per_cutting_s: float
    intercept: float
    slope_low: float
    slope_high: float
    kendall_tau: float
    kendall_p: float
    n_points: int


def fit_degradation(x: np.ndarray, y: np.ndarray) -> Degradation | None:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 8:
        return None
    xv, yv = x[valid], y[valid]
    slope, intercept, low, high = theilslopes(yv, xv, alpha=0.95)
    tau, p = kendalltau(xv, yv)
    return Degradation(float(slope), float(intercept), float(low), float(high),
                       float(tau), float(p), int(valid.sum()))


def forecast_horizon(
    fit: Degradation, x: np.ndarray, y: np.ndarray, limit: float,
    median_run_cutting_s: float, n_bootstrap: int, seed: int,
) -> dict[str, float]:
    """Projected crossing of the level limit, with a bootstrap interval."""
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    last_x = float(xv.max())
    current_level = float(np.median(yv[-5:])) if yv.size >= 5 else float(yv[-1])

    def crossing(slope: float, intercept: float) -> float:
        if slope <= 0:
            return np.inf
        target_x = (limit - intercept) / slope
        return max(target_x - last_x, 0.0)

    point = crossing(fit.slope_per_cutting_s, fit.intercept)

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, xv.size, xv.size)
        if np.unique(xv[idx]).size < 3:
            continue
        try:
            slope, intercept, *_ = theilslopes(yv[idx], xv[idx], alpha=0.95)
        except Exception:
            continue
        samples.append(crossing(float(slope), float(intercept)))
    samples = np.array(samples, dtype=float)
    finite = samples[np.isfinite(samples)]

    def to_runs(seconds: float) -> float:
        if not np.isfinite(seconds) or median_run_cutting_s <= 0:
            return np.inf
        return seconds / median_run_cutting_s

    never = float(np.mean(~np.isfinite(samples))) if samples.size else np.nan
    return {
        "current_health_index": current_level,
        "level_limit": limit,
        "already_above_limit": bool(current_level >= limit),
        "cutting_seconds_to_limit": point,
        "cutting_minutes_to_limit": point / 60.0 if np.isfinite(point) else np.inf,
        "runs_to_limit": to_runs(point),
        "runs_to_limit_ci_low": to_runs(float(np.quantile(finite, 0.05))) if finite.size else np.nan,
        "runs_to_limit_ci_high": to_runs(float(np.quantile(finite, 0.95))) if finite.size else np.nan,
        "bootstrap_share_never_crossing": never,
    }


# --------------------------------------------------------------------------- #
#  Stage 5 — evaluation without labels
# --------------------------------------------------------------------------- #

def prequential_evaluation(
    x: np.ndarray, y: np.ndarray, k: int, warmup: int, rolling_window: int = 30,
) -> dict[str, float]:
    """Forecast k runs ahead at every step, then score against what happened.

    Measures forecast skill directly -- no wear labels required -- against a
    persistence baseline (the next value looks like this one), which is the
    honest benchmark for a slowly varying series.  Two model variants are
    scored: a fit over all history to date, and a fit over the most recent
    ``rolling_window`` runs.  A global fit is dominated by old data and can
    easily be *worse* than persistence, so reporting only one would be
    misleading either way.
    """
    errors = {"global": [], "rolling": [], "persistence": [], "smoothed": []}
    for t in range(warmup, len(y) - k):
        actual = y[t + k]
        if not np.isfinite(actual):
            continue
        valid = np.isfinite(x[:t + 1]) & np.isfinite(y[:t + 1])
        if valid.sum() < 8:
            continue

        xs, ys = x[:t + 1][valid], y[:t + 1][valid]
        try:
            slope, intercept, *_ = theilslopes(ys, xs, alpha=0.95)
            errors["global"].append(abs((slope * x[t + k] + intercept) - actual))
        except Exception:
            continue

        lo = max(0, xs.size - rolling_window)
        if xs.size - lo >= 8:
            try:
                slope_r, intercept_r, *_ = theilslopes(ys[lo:], xs[lo:], alpha=0.95)
                errors["rolling"].append(abs((slope_r * x[t + k] + intercept_r) - actual))
            except Exception:
                errors["rolling"].append(np.nan)
        else:
            errors["rolling"].append(np.nan)

        errors["persistence"].append(abs(y[t] - actual))
        errors["smoothed"].append(abs(float(np.nanmedian(ys[-5:])) - actual))

    if not errors["global"]:
        return {"prequential_points": 0}

    def mae(key: str) -> float:
        values = np.array(errors[key], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan

    persistence_mae = mae("persistence")
    out = {
        "prequential_points": len(errors["global"]),
        "horizon_k": k,
        "rolling_window": rolling_window,
        "mae_global_trend": mae("global"),
        "mae_rolling_trend": mae("rolling"),
        "mae_smoothed_level": mae("smoothed"),
        "mae_persistence": persistence_mae,
    }
    for key, label in [("global", "global_trend"), ("rolling", "rolling_trend"),
                       ("smoothed", "smoothed_level")]:
        value = mae(key)
        out[f"skill_{label}_vs_persistence"] = (
            float(1.0 - value / persistence_mae)
            if persistence_mae and np.isfinite(value) else np.nan
        )
    best = min(
        [("global_trend", out["mae_global_trend"]), ("rolling_trend", out["mae_rolling_trend"]),
         ("smoothed_level", out["mae_smoothed_level"]), ("persistence", persistence_mae)],
        key=lambda pair: pair[1] if np.isfinite(pair[1]) else np.inf,
    )
    out["best_forecaster"] = best[0]
    return out


def leave_one_tool_out(
    data: pd.DataFrame, machine_id: int, program: str, tools: list[str],
    warmup: int, alpha: float, n_resamples: int, seed: int,
) -> pd.DataFrame:
    """Run the whole detector on each cohort tool in turn.

    This is the specificity check, and it is the one that caught the first two
    versions of this detector.  A one-sided CUSUM on the level alarmed on 4 of
    5 tools including one trending *downward*; replacing it with a sequential
    Mann-Kendall statistic still alarmed on 4 of 5, because taking a maximum
    over run order finds a locally rising stretch in any autocorrelated series.
    Only the block-bootstrap null, which preserves that autocorrelation,
    separates the target from its cohort.
    """
    rows = []
    for tool in tools:
        try:
            target, cohort, _ = select_cohort(data, machine_id, program, tool)
        except SystemExit:
            continue
        if len(cohort) < 2:
            continue
        health = build_health_index(target, cohort)
        values = health["health_index"].to_numpy(dtype=float)
        result = detect_trend(values, warmup, alpha, n_resamples, seed)
        fit = fit_degradation(health["cumulative_cutting_s"].to_numpy(dtype=float), values)
        rows.append({
            "tool": tool,
            "runs": len(health),
            "cohort_size": len(cohort),
            "health_index_median": float(np.nanmedian(values)),
            "lag1_autocorrelation": result["lag1_autocorrelation"],
            "naive_kendall_tau": result["naive_kendall_tau"],
            "naive_kendall_p": result["naive_kendall_p"],
            "block_bootstrap_p": result["block_bootstrap_p"],
            "hamed_rao_p": result["hamed_rao_p"],
            "prewhitened_p": result["prewhitened_p"],
            "routes_significant_of_3": result["routes_significant_of_3"],
            "alarm": result["alarm"],
            "onset_run_order": result["onset_run_order"],
            "slope_per_cutting_hour": fit.slope_per_cutting_s * 3600 if fit else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #

def plot_report(health: pd.DataFrame, sequential: np.ndarray, null_95: float,
                level_limit: float, fit: Degradation | None, output_dir: Path,
                title: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    runs = health["run_order"].to_numpy()

    axes[0].plot(runs, health["current_mean_norm"], marker="o", ms=3, lw=1.2,
                 color="#3f5c76", label="tool (normalised)")
    axes[0].plot(runs, health["current_mean_cohort"], lw=1.6, color="#8a6516",
                 label="cohort median")
    axes[0].set_ylabel("current / own median")
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].axhline(0, color="#999", lw=0.8)
    axes[1].plot(runs, health["health_index"], marker="o", ms=3, lw=1.2, color="#3d6b4f")
    if np.isfinite(level_limit):
        axes[1].axhline(level_limit, color="#9f2d20", lw=1.2, ls="--",
                        label=f"cohort level limit {level_limit:.3f}")
    if fit is not None:
        x = health["cumulative_cutting_s"].to_numpy(dtype=float)
        axes[1].plot(runs, fit.slope_per_cutting_s * x + fit.intercept,
                     lw=1.4, color="#9333ea", label="Theil-Sen trajectory")
    axes[1].set_ylabel("health index\n(fractional excess vs cohort)")
    axes[1].legend(fontsize=8, loc="upper left")

    axes[2].plot(runs, sequential, lw=1.5, color="#b35c00")
    axes[2].axhline(0, color="#999", lw=0.8)
    if np.isfinite(null_95):
        axes[2].axhline(null_95, color="#9f2d20", lw=1.2, ls="--",
                        label=f"block-bootstrap 95% null {null_95:.3f}")
        crossing = np.flatnonzero(np.nan_to_num(sequential, nan=-np.inf) >= null_95)
        if crossing.size:
            axes[2].axvline(runs[crossing[0]], color="#9f2d20", lw=1.0, alpha=0.5)
    axes[2].set_ylabel("sequential Kendall tau")
    axes[2].set_xlabel("run order")
    axes[2].legend(fontsize=8, loc="upper left")

    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.savefig(output_dir / "toolwear_prediction.png", dpi=150)
    plt.close(fig)


def restrict_scope(frame: pd.DataFrame, since, until) -> pd.DataFrame:
    out = frame
    if since is not None:
        out = out[out["started_at"] >= since]
    if until is not None:
        out = out[out["started_at"] <= until]
    return out


def analyse_tool(
    data: pd.DataFrame, machine_id: int, program: str, tool: str,
    warmup: int, alpha: float, resamples: int, seed: int, since=None, until=None,
) -> dict | None:
    """One tool, one scope -- the whole inference chain, no side effects."""
    try:
        target, cohort, _ = select_cohort(data, machine_id, program, tool)
    except SystemExit:
        return None
    if len(cohort) < 2:
        return None
    health = build_health_index(target, cohort)
    health = restrict_scope(health, since, until)
    values = health["health_index"].to_numpy(dtype=float)
    if np.isfinite(values).sum() < 20:
        return None
    trend = detect_trend(values, warmup, alpha, resamples, seed)
    fit = fit_degradation(health["cumulative_cutting_s"].to_numpy(dtype=float), values)
    return {
        "tool": tool,
        "runs": int(len(health)),
        "cutting_hours": float(health["cutting_s"].sum() / 3600),
        "lag1_autocorrelation": trend["lag1_autocorrelation"],
        "naive_kendall_tau": trend["naive_kendall_tau"],
        "naive_kendall_p": trend["naive_kendall_p"],
        "block_bootstrap_p": trend["block_bootstrap_p"],
        "hamed_rao_p": trend["hamed_rao_p"],
        "prewhitened_p": trend["prewhitened_p"],
        "routes_significant_of_3": trend["routes_significant_of_3"],
        "alarm": trend["alarm"],
        "onset_run_order": trend["onset_run_order"],
        "slope_per_cutting_hour": fit.slope_per_cutting_s * 3600 if fit else np.nan,
    }


def sweep(
    data: pd.DataFrame, machine_id: int, program: str,
    warmup: int, alpha: float, resamples: int, seed: int,
) -> pd.DataFrame:
    """Every eligible tool on a machine/programme, over two scopes.

    A trend can be real over one span and absent over another: on machine 288
    tool 7 alarms inside its 43-run comparable window but not across its full
    133-run history (a localised episode), while on machine 294 tool 2 does the
    opposite (a slow trend needing a long window to see).  Requiring agreement
    across both scopes is a far stronger criterion than either alone.
    """
    scope = data[data["machine_id"].eq(machine_id) & data["program"].eq(program)
                 & data["analysis_ready_run"]]
    rows = []
    for tool in sorted(scope["tool"].astype(str).unique()):
        sequence = scope[scope["tool"].eq(tool)].sort_values("started_at")
        if sequence.empty:
            continue
        # Two scopes: the tool's whole recorded history, and the most recent
        # half of it.  A trend can be real over one span and absent over the
        # other, so agreement across both is a stricter criterion than either.
        midpoint = sequence["started_at"].iloc[len(sequence) // 2]
        scopes = {"full_sequence": (None, None), "recent_half": (midpoint, None)}
        for label, (since, until) in scopes.items():
            result = analyse_tool(data, machine_id, program, tool, warmup, alpha,
                                  resamples, seed, since, until)
            if result:
                rows.append({"scope": label, **result})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cohort-relative degradation forecasting with permutation-calibrated alarms.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--machine", type=int, required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for the block-bootstrap trend test.")
    parser.add_argument("--power-ramp", type=float, default=0.20,
                        help="Ramp size (fraction of own median current) for the power check.")
    parser.add_argument("--level-alpha", type=float, default=0.05,
                        help="Cohort quantile for the health-index level limit.")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--power-trials", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--horizon-k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--since", type=lambda v: pd.Timestamp(v, tz="UTC"),
                        help="Restrict analysis to runs at or after this timestamp.")
    parser.add_argument("--until", type=lambda v: pd.Timestamp(v, tz="UTC"),
                        help="Restrict analysis to runs at or before this timestamp.")
    parser.add_argument("--sweep", action="store_true",
                        help="Analyse every eligible tool on this machine/programme "
                             "and write a comparison table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_sequences(args.input)

    target, cohort, rejected = select_cohort(data, args.machine, args.program, args.tool)
    if len(cohort) < 2:
        raise SystemExit(
            f"Only {len(cohort)} usable cohort tools; a cohort-relative index needs at "
            f"least 2. Rejected: {rejected}")

    health = restrict_scope(build_health_index(target, cohort), args.since, args.until)
    if len(health) < 20:
        raise SystemExit(f"Only {len(health)} runs after the scope filter; need at least 20.")
    values = health["health_index"].to_numpy(dtype=float)
    x = health["cumulative_cutting_s"].to_numpy(dtype=float)

    if args.sweep:
        table = sweep(data, args.machine, args.program, args.warmup, args.alpha,
                      args.resamples, args.seed)
        table.to_csv(args.output_dir / "tool_sweep.csv", index=False)
        print(table.round(4).to_string(index=False))
        print(f"\nWrote {args.output_dir / 'tool_sweep.csv'}")

    trend = detect_trend(values, args.warmup, args.alpha, args.resamples, args.seed)
    health["sequential_tau"] = trend["sequential_tau"]

    power = bootstrap_power(
        values, trend["block_length"], args.power_ramp, args.alpha,
        args.power_trials, max(args.resamples // 10, 200), args.seed)

    # Cohort health indices, each scored leave-one-out against the rest.
    cohort_his = {}
    for name in cohort:
        # Leave-one-out among the cohort only.  Including the target here would
        # let a genuinely drifting target raise the cohort median, depress every
        # other tool's health index, and so lower the very quantile it is about
        # to be tested against.
        others = {k: v for k, v in cohort.items() if k != name}
        if len(others) >= 2:
            cohort_his[name] = build_health_index(
                cohort[name], others)["health_index"].to_numpy(dtype=float)
    level_limit, cohort_points = cohort_level_limit(values, cohort_his, args.level_alpha)

    fit = fit_degradation(x, values)
    horizon = forecast_horizon(
        fit, x, values, level_limit, float(np.median(health["cutting_s"])),
        args.bootstrap, args.seed) if fit else {}
    prequential = prequential_evaluation(x, values, args.horizon_k, args.warmup)
    loto = leave_one_tool_out(
        data, args.machine, args.program, [args.tool] + sorted(cohort),
        args.warmup, args.alpha, max(args.resamples // 2, 500), args.seed)

    health.to_csv(args.output_dir / "health_index_series.csv", index=False)
    loto.to_csv(args.output_dir / "leave_one_tool_out.csv", index=False)

    report = {
        "target": {"machine_id": args.machine, "program": args.program, "tool": args.tool,
                   "runs": int(len(health)),
                   "cutting_hours": float(health["cutting_s"].sum() / 3600)},
        "cohort": {"tools": sorted(cohort), "size": len(cohort), "rejected": rejected,
                   "pooled_reference_points": cohort_points},
        "health_index": {
            "median": float(np.nanmedian(values)),
            "last_5_median": float(np.nanmedian(values[-5:])),
            "total_rise": float(np.nanmedian(values[-10:]) - np.nanmedian(values[:10])),
            "short_term_sigma": short_term_sigma(values),
        },
        "trend_inference": {k: v for k, v in trend.items() if k != "sequential_tau"},
        "detection_power": {"ramp": args.power_ramp, "power": power,
                            "trials": args.power_trials},
        "level_limit": {"value": level_limit, "alpha": args.level_alpha,
                        "basis": "cohort quantile, leave-one-out"},
        "degradation": asdict(fit) if fit else None,
        "horizon": horizon,
        "prequential": prequential,
        "caveat": (
            "No tool-wear ground truth exists in this dataset. Thresholds are calibrated "
            "against the cohort and against an autocorrelation-preserving block-bootstrap "
            "null, NOT against observed tool failures. Outputs describe when the monitored "
            "statistic is predicted to cross a calibrated limit; they are not remaining "
            "useful life."
        ),
    }
    (args.output_dir / "prediction_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    if not args.skip_plot:
        plot_report(health, trend["sequential_tau"],
                    safe_float(trend.get("block_bootstrap_null_95"), np.nan),
                    level_limit, fit, args.output_dir,
                    f"machine {args.machine} / {args.program} / tool {args.tool} "
                    f"vs cohort {sorted(cohort)}")

    write_manifest(args.output_dir, "predict_toolwear", args, {"cohort": sorted(cohort)})

    lines = [
        f"TARGET  machine {args.machine} / {args.program} / tool {args.tool}",
        f"        {len(health)} runs, {health['cutting_s'].sum() / 3600:.2f} cutting hours",
        f"COHORT  tools {sorted(cohort)}  ({cohort_points:,} pooled reference tool-runs)",
        f"        rejected: {'; '.join(rejected) if rejected else 'none'}",
        "",
        "HEALTH INDEX (fractional excess over contemporaneous cohort)",
        f"        median {np.nanmedian(values):+.4f}   last-5 {np.nanmedian(values[-5:]):+.4f}"
        f"   total rise {report['health_index']['total_rise']:+.4f}",
        f"        lag-1 autocorrelation {trend['lag1_autocorrelation']:+.2f}"
        f"   (block length {trend['block_length']})",
        "",
        "TREND INFERENCE  -- the naive test is not usable on autocorrelated series",
        f"        naive Kendall          tau={trend['naive_kendall_tau']:+.3f}  "
        f"p={trend['naive_kendall_p']:.3g}   <-- assumes independence, do not quote",
        f"        block bootstrap        p={trend['block_bootstrap_p']:.4f} "
        f"(worst over block lengths {trend['block_length_grid']})   <-- drives the alarm",
        "        per block length: " + ", ".join(
            f"L={b}:p={pv:.3f}" for b, pv in
            sorted(trend['block_bootstrap_p_by_length'].items())),
        f"        Hamed-Rao corrected    z={trend['hamed_rao_z']:+.2f}  "
        f"p={trend['hamed_rao_p']:.4f}   (variance inflated "
        f"{trend['hamed_rao_variance_inflation']:.1f}x)",
        f"        Yue-Pilon prewhitened  tau={trend['prewhitened_tau']:+.3f}  "
        f"p={trend['prewhitened_p']:.3g}",
        f"        routes significant: {trend['routes_significant_of_3']} of 3",
        f"        ALARM: {'YES' if trend['alarm'] else 'no'}"
        + (f"   trend detectable from run {trend['onset_run_order']}"
           if np.isfinite(safe_float(trend['onset_run_order'], np.nan)) else ""),
        "",
        f"POWER   block bootstrap detects a +{args.power_ramp:.2f} ramp "
        f"{power:.0%} of the time ({args.power_trials} trials)",
        "",
        f"LEVEL LIMIT  {level_limit:+.4f}  (cohort {1 - args.level_alpha:.0%} quantile, "
        f"leave-one-out)",
    ]
    if fit:
        lines += [
            "",
            "DEGRADATION  Theil-Sen on cumulative cutting seconds",
            f"        slope {fit.slope_per_cutting_s * 3600:+.4f} per cutting hour "
            f"[{fit.slope_low * 3600:+.4f}, {fit.slope_high * 3600:+.4f}]",
        ]
    if horizon:
        if horizon.get("already_above_limit"):
            lines += ["", "HORIZON  already above the level limit"]
        elif np.isfinite(horizon.get("runs_to_limit", np.inf)):
            lines += [
                "",
                f"HORIZON  {horizon['runs_to_limit']:.0f} runs to limit "
                f"[{horizon['runs_to_limit_ci_low']:.0f}, "
                f"{horizon['runs_to_limit_ci_high']:.0f}]",
                f"        {horizon['cutting_minutes_to_limit']:.0f} cutting minutes",
            ]
        else:
            lines += ["", "HORIZON  trajectory does not reach the limit (slope <= 0)"]
    if prequential.get("prequential_points"):
        lines += [
            "",
            f"PREQUENTIAL  {args.horizon_k}-step ahead, "
            f"{prequential['prequential_points']} forecasts",
            f"        persistence      MAE {prequential['mae_persistence']:.4f}  (baseline)",
            f"        smoothed level   MAE {prequential['mae_smoothed_level']:.4f}"
            f"   skill {prequential['skill_smoothed_level_vs_persistence']:+.1%}",
            f"        rolling trend    MAE {prequential['mae_rolling_trend']:.4f}"
            f"   skill {prequential['skill_rolling_trend_vs_persistence']:+.1%}",
            f"        global trend     MAE {prequential['mae_global_trend']:.4f}"
            f"   skill {prequential['skill_global_trend_vs_persistence']:+.1%}",
            f"        best forecaster: {prequential['best_forecaster']}",
        ]
    if not loto.empty:
        lines += ["", "LEAVE-ONE-TOOL-OUT  (specificity check)"]
        for _, row in loto.iterrows():
            marker = "<-- target" if str(row["tool"]) == str(args.tool) else ""
            lines.append(
                f"        tool {str(row['tool']):<3} alarm={str(row['alarm']):<5} "
                f"tau={row['naive_kendall_tau']:+.3f} "
                f"block_p={row['block_bootstrap_p']:.4f} "
                f"routes={int(row['routes_significant_of_3'])}/3 {marker}")
    lines += ["", "CAVEAT  " + report["caveat"]]

    summary = "\n".join(lines) + "\n"
    (args.output_dir / "prediction_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
