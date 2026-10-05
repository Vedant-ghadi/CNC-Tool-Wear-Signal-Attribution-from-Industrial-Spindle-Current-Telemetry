"""Shared helpers for the CNC DTW labelling pipeline.

Centralises the small utilities that were previously duplicated (and in several
places subtly wrong) across the individual stage scripts:

* ``safe_float`` / ``safe_max`` -- NaN-safe scalar coercion.  The old
  ``float(x or 0.0)`` idiom does NOT catch NaN, because ``np.nan`` is truthy,
  so ``np.nan or 0.0`` evaluates to ``np.nan``.
* ``parse_bool``           -- consistent CSV boolean parsing.
* ``robust_z`` / ``mad``   -- median/MAD scaling with a documented fallback.
* ``benjamini_hochberg``   -- multiplicity control for the trend test families.
* ``first_last_medians``   -- non-overlapping head/tail comparison windows.
* ``write_manifest``       -- reproducibility record for every output folder.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  NaN-safe scalar handling
# --------------------------------------------------------------------------- #

def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float, mapping None/NaN/inf/non-numeric to ``default``.

    Replaces ``float(x or 0.0)``, which silently propagates NaN because NaN is
    truthy in Python.
    """
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def safe_max(*values: Any, default: float = 0.0) -> float:
    """``max`` over values, ignoring NaN/None entirely.

    ``max(np.nan, 0.0)`` returns NaN because NaN comparisons are all False and
    ``max`` keeps the first element.  This does the intended thing.
    """
    clean = [safe_float(v, default=np.nan) for v in values]
    clean = [v for v in clean if np.isfinite(v)]
    if not clean:
        return float(default)
    return float(max(clean))


def parse_bool(series: pd.Series) -> pd.Series:
    """Parse a CSV round-tripped boolean column into real booleans."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def to_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce the listed columns to numeric in place where present."""
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


# --------------------------------------------------------------------------- #
#  Robust scaling
# --------------------------------------------------------------------------- #

def mad(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    """Median absolute deviation, scaled to be a consistent sigma estimate."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)) * 1.4826)


def robust_scale(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    """MAD-based sigma, falling back to the standard deviation when MAD is 0.

    A zero MAD happens whenever more than half the values are identical, which
    is common in small windows; the std fallback keeps the z-score finite.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    scale = mad(arr)
    if np.isfinite(scale) and scale > 0:
        return float(scale)
    if arr.size > 1:
        std = float(np.std(arr, ddof=1))
        if np.isfinite(std) and std > 0:
            return std
    return np.nan


def robust_z_by_group(
    frame: pd.DataFrame,
    column: str,
    group_column: str = "comparable_window_id",
    exclude_mask: pd.Series | None = None,
) -> pd.Series:
    """Median/MAD z-score of ``column`` within each group.

    ``exclude_mask`` marks rows that must not contribute to the centre and
    scale but still receive a score.  This is how reference runs (whose
    self-distance is 0 by construction) are kept out of their own baseline.
    """
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    if column not in frame.columns:
        return out

    for _, idx in frame.groupby(group_column, sort=False).groups.items():
        idx = list(idx)
        values = pd.to_numeric(frame.loc[idx, column], errors="coerce").astype(float)
        if exclude_mask is not None:
            basis = values[~exclude_mask.reindex(idx).fillna(False).to_numpy()]
        else:
            basis = values
        basis = basis[np.isfinite(basis)]
        if basis.empty:
            out.loc[idx] = 0.0
            continue
        centre = float(basis.median())
        scale = robust_scale(basis)
        if not np.isfinite(scale) or scale == 0:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (values - centre) / scale
    return out


# --------------------------------------------------------------------------- #
#  Multiplicity control
# --------------------------------------------------------------------------- #

def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values).

    NaN inputs are treated as p=1 and returned as NaN so they never count as
    discoveries.  Implemented locally so the pipeline does not depend on a
    specific SciPy version exposing ``false_discovery_control``.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out

    pv = p[valid]
    n = pv.size
    order = np.argsort(pv, kind="stable")
    ranked = pv[order]
    q = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    restored = np.empty(n, dtype=float)
    restored[order] = q
    out[valid] = restored
    return out


def permutation_p_value(
    statistic: float,
    values: np.ndarray,
    split: int,
    n_permutations: int = 2000,
    seed: int = 0,
    min_side: int | None = None,
) -> float:
    """Permutation p-value for a median-difference changepoint.

    The observed statistic is the **maximum** absolute effect over all
    admissible split positions, so the null distribution must be built the same
    way: each shuffle is scanned over every admissible split and its own maximum
    taken.  Scoring the shuffles at one fixed split -- as the earlier version
    did -- ignores the selection over splits and makes the p-value
    anti-conservative.

    ``min_side`` is the minimum number of observations either side of a split;
    when omitted it is inferred from ``split`` so that callers passing only the
    chosen split still get a valid, if slightly narrower, scan.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 4 or split < 1 or split >= n or not np.isfinite(statistic):
        return np.nan

    if min_side is None:
        min_side = max(1, min(split, n - split))
    lo, hi = min_side, n - min_side
    if lo >= hi:
        lo, hi = 1, n

    rng = np.random.default_rng(seed)
    observed = abs(statistic)
    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(values)
        best = 0.0
        for candidate in range(lo, hi):
            left, right = shuffled[:candidate], shuffled[candidate:]
            scale = robust_scale(
                np.concatenate([left - np.median(left), right - np.median(right)])
            )
            if not np.isfinite(scale) or scale == 0:
                continue
            effect = abs((np.median(right) - np.median(left)) / scale)
            if effect > best:
                best = effect
        if best >= observed:
            count += 1
    return float((count + 1) / (n_permutations + 1))


def first_last_medians(values: np.ndarray, fraction: float = 0.1, minimum: int = 3):
    """Median of the first and last block, guaranteed not to overlap.

    The previous implementation used ``n // 10`` with a floor of 3 for both
    ends, so for n < 6 the two blocks shared elements and the resulting delta
    was biased toward zero.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return np.nan, np.nan, 0
    block = max(minimum, int(round(n * fraction)))
    block = min(block, n // 2)
    if block < 1:
        return np.nan, np.nan, 0
    return float(np.median(arr[:block])), float(np.median(arr[-block:])), int(block)


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #

def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def write_manifest(output_dir: Path, stage: str, args: Any, extra: dict | None = None) -> None:
    """Record what produced this folder, so an output can be traced to its run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _encode(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    arg_dict = {k: _encode(v) for k, v in vars(args).items()} if hasattr(args, "__dict__") else {}

    packages = {}
    for name in ["pandas", "numpy", "scipy", "sklearn", "matplotlib", "ruptures"]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[name] = None

    manifest = {
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "arguments": arg_dict,
        "packages": packages,
    }
    if extra:
        manifest["details"] = extra

    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
#  Autocorrelation-aware trend inference
# --------------------------------------------------------------------------- #
#
#  Kendall's tau test assumes independent observations.  CNC run sequences are
#  strongly serially correlated -- lag-1 runs 0.13 to 0.95 across this dataset,
#  and ~0.80 on the raw current series -- which inflates the true variance of
#  the test statistic and makes the ordinary p-value far too small.  On the
#  worst case measured here the naive p-value was optimistic by roughly twelve
#  orders of magnitude (1.8e-14 against a corrected 0.018).
#
#  Three corrections are provided.  The circular block bootstrap is the primary
#  one: it preserves serial dependence up to the block length while destroying
#  longer-range trend, which is precisely the null being tested.  Hamed & Rao
#  variance inflation and Yue-Pilon prewhitening are reported alongside it as
#  cross-checks -- Hamed-Rao in particular is very conservative on short series
#  and should not be used alone.

def kendall_s(y: np.ndarray) -> float:
    """Kendall's S against a strictly increasing index: sum_{i<j} sign(y_j - y_i).

    Used instead of ``scipy.stats.kendalltau`` inside bootstrap loops.  It is a
    monotone transform of tau when x has no ties, so p-values are identical,
    and the vectorised form is fast enough to run thousands of resamples.
    """
    y = np.asarray(y, dtype=float)
    difference = np.sign(y[None, :] - y[:, None])
    return float(np.triu(difference, k=1).sum())


def lag1_autocorrelation(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return np.nan
    centred = arr - arr.mean()
    denominator = float(np.sum(centred ** 2))
    if denominator <= 0:
        return np.nan
    return float(np.sum(centred[:-1] * centred[1:]) / denominator)


def estimate_block_length(values: Sequence[float] | np.ndarray) -> int:
    """First lag at which autocorrelation falls inside the white-noise band."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 20:
        return max(2, n // 4)
    centred = arr - arr.mean()
    denominator = float(np.sum(centred ** 2))
    if denominator <= 0:
        return max(2, int(round(n ** (1 / 3))))
    critical = 1.96 / np.sqrt(n)
    for lag in range(1, min(n // 3, 50)):
        acf = float(np.sum(centred[:n - lag] * centred[lag:]) / denominator)
        if abs(acf) < critical:
            return int(np.clip(lag, 2, max(2, n // 4)))
    return int(np.clip(n // 5, 2, max(2, n // 4)))


def block_length_grid(n: int, acf_block: int) -> list[int]:
    """Several defensible block lengths, so no single arbitrary choice decides."""
    candidates = {
        max(2, int(round(n ** (1 / 3)))),
        max(2, int(acf_block)),
        max(2, n // 10),
        max(2, n // 6),
    }
    return sorted({int(np.clip(b, 2, max(3, n // 4))) for b in candidates})


def circular_block_resample(values: np.ndarray, block: int, rng) -> np.ndarray:
    """Blocks sampled with replacement from the circularly-extended series."""
    n = values.size
    extended = np.concatenate([values, values])
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    return np.concatenate([extended[s:s + block] for s in starts])[:n]


def hamed_rao_test(values: np.ndarray) -> dict[str, float]:
    """Mann-Kendall with Hamed & Rao (1998) variance inflation.

    Hamed & Rao require the autocorrelation to be estimated from the series
    **after the trend has been removed**.  Estimating it from the raw series
    lets a real trend masquerade as serial dependence, inflating the variance
    and destroying the power of the test -- which is what the earlier version
    of this function did (variance inflation reached 16.8x, and no window
    survived the correction at any effect size).
    """
    from scipy.stats import theilslopes

    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 10:
        return {"z": np.nan, "p": np.nan, "variance_inflation": np.nan}

    s_statistic = kendall_s(y)
    base_variance = n * (n - 1) * (2 * n + 5) / 18.0

    # Detrend with Sen's slope before estimating the autocorrelation structure.
    x = np.arange(n, dtype=float)
    slope, intercept, *_ = theilslopes(y, x, alpha=0.95)
    detrended = y - (slope * x + intercept)

    ranks = np.argsort(np.argsort(detrended)) + 1.0
    centred = ranks - ranks.mean()
    denominator = float(np.sum(centred ** 2))
    critical = 1.96 / np.sqrt(n)
    inflation = 1.0
    if denominator > 0:
        for lag in range(1, n - 3):
            acf = float(np.sum(centred[:n - lag] * centred[lag:]) / denominator)
            if abs(acf) > critical:
                inflation += (2.0 * (n - lag) * (n - lag - 1) * (n - lag - 2)
                              / (n * (n - 1) * (n - 2)) * acf)
    inflation = max(inflation, 1e-9)
    variance = base_variance * inflation
    z = (s_statistic - np.sign(s_statistic)) / np.sqrt(variance)
    from scipy.stats import norm as _norm
    return {"z": float(z), "p": float(2 * (1 - _norm.cdf(abs(z)))),
            "variance_inflation": float(inflation)}


def prewhitened_test(values: np.ndarray) -> dict[str, float]:
    """Yue-Pilon trend-free prewhitening (Yue et al., 2002).

    The procedure is: estimate the slope with Sen's estimator, remove it,
    estimate the AR(1) coefficient on the *detrended* residual, prewhiten that
    residual, add the trend back, then test.  The earlier version applied
    ``y[1:] - r * y[:-1]`` directly to the raw series, which is the older
    von Storch prewhitening; that removes part of the trend along with the
    autocorrelation and is exactly the loss of power the Yue-Pilon procedure
    was introduced to avoid.
    """
    from scipy.stats import kendalltau as _kendalltau, theilslopes

    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 10:
        return {"ar1": np.nan, "tau": np.nan, "p": np.nan}

    x = np.arange(n, dtype=float)
    slope, intercept, *_ = theilslopes(y, x, alpha=0.95)
    detrended = y - slope * x

    r = lag1_autocorrelation(detrended)
    if not np.isfinite(r) or r <= 0.05:
        tau, p = _kendalltau(x, y)
        return {"ar1": float(r) if np.isfinite(r) else 0.0,
                "tau": float(tau), "p": float(p)}

    # Prewhiten the residual, then restore the deterministic trend.
    prewhitened = detrended[1:] - r * detrended[:-1]
    restored = prewhitened + slope * x[1:]
    tau, p = _kendalltau(np.arange(restored.size), restored)
    return {"ar1": float(r), "tau": float(tau), "p": float(p)}


def block_bootstrap_trend(
    values: Sequence[float] | np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
    one_sided: bool = False,
) -> dict[str, Any]:
    """Autocorrelation-preserving p-value for a monotone trend.

    Reports the p-value at several block lengths and takes the least
    favourable, so the conclusion does not rest on one arbitrary choice.
    """
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 12:
        return {"kendall_s": np.nan, "p": np.nan, "p_by_block": {},
                "block_lengths": [], "lag1": np.nan}

    observed = kendall_s(y)
    acf_block = estimate_block_length(y)
    grid = block_length_grid(n, acf_block)

    rng = np.random.default_rng(seed)
    p_by_block: dict[int, float] = {}
    for block in grid:
        null = np.array([
            kendall_s(circular_block_resample(y, block, rng)) for _ in range(n_resamples)
        ])
        # (count + 1) / (B + 1): a resampled p-value must never be exactly
        # zero, or the FDR step downstream receives an impossible q-value.
        exceed = int(np.sum(null >= observed) if one_sided
                     else np.sum(np.abs(null) >= abs(observed)))
        p_by_block[int(block)] = float((exceed + 1) / (n_resamples + 1))
    worst = float(max(p_by_block.values())) if p_by_block else np.nan
    return {"kendall_s": observed, "p": worst, "p_by_block": p_by_block,
            "block_lengths": grid, "lag1": lag1_autocorrelation(y)}
