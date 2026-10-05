"""Band-constrained Dynamic Time Warping with correct, consistent normalisation.

Replaces the inline implementation that previously lived in
``build_dtw_features.py``.  Two defects are fixed here:

1. **Normalisation depended on ``return_path``.**  The old code computed
   ``path_len = max(len(path), 1)`` and divided by it.  When the caller asked
   for distance only, ``path`` was empty, ``path_len`` became 1, and the
   distance was returned *unnormalised* -- roughly 20x larger than the same
   comparison made with ``return_path=True``.  Since the medoid reference was
   scored with the path and the other three references without it, the four
   reference distances lived on incompatible scales and any statistic computed
   across them (agreement std, CV, cross-reference minima) was meaningless.
   The warping path length is now always computed, so the returned distance
   does not depend on what the caller asked for.

2. **Backtracking could leave the band undetected.**  The old backtrack took
   ``min()`` over three neighbouring costs without checking they had ever been
   filled.  Outside the Sakoe-Chiba band the cost is ``inf``; ``min`` over
   equal infinities returns the first tuple, so the walk silently collapsed to
   the diagonal instead of following the optimal path.  The forward pass now
   records the predecessor it actually used, and the backtrack follows that.

The distance returned is the root-mean-square deviation per aligned point,
which is comparable across series of different lengths -- the raw accumulated
cost is not, because it grows with the number of aligned pairs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Normalisation = Literal["path", "sum", "none"]

_DIAG, _UP, _LEFT = 0, 1, 2


def z_normalize(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance scaling used for shape-only comparison."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std == 0.0 or not np.isfinite(std):
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def sakoe_chiba_band(n: int, m: int, band_fraction: float) -> int:
    """Half-width of the warping band.

    Always at least ``|n - m|`` so that a monotone path from (0,0) to (n,m)
    exists; otherwise short-vs-long comparisons would return NaN.
    """
    return max(abs(n - m), int(np.ceil(max(n, m) * band_fraction)), 1)


def constrained_dtw(
    a: np.ndarray,
    b: np.ndarray,
    band_fraction: float = 0.10,
    return_path: bool = False,
    normalisation: Normalisation = "path",
) -> tuple[float, list[tuple[int, int]]]:
    """DTW distance under a Sakoe-Chiba band.

    Parameters
    ----------
    a, b
        1-D sequences.  ``a`` is treated as the reference/template.
    band_fraction
        Band half-width as a fraction of the longer sequence.
    return_path
        Whether to return the alignment path.  **Does not affect the distance.**
    normalisation
        ``"path"`` divides the accumulated squared cost by the number of
        aligned pairs (default, comparable across lengths); ``"sum"`` divides
        by ``n + m``; ``"none"`` returns the raw accumulated cost.

    Returns
    -------
    (distance, path)
        ``path`` is a list of 0-indexed ``(i, j)`` pairs, empty when
        ``return_path`` is False.  ``distance`` is NaN if no valid alignment
        exists.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return np.nan, []

    band = sakoe_chiba_band(n, m, band_fraction)

    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    pred = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        centre = int(round(i * m / n))
        j_start = max(1, centre - band)
        j_end = min(m, centre + band)
        if j_start > j_end:
            continue

        # Squared point distances for this row, computed in one vectorised shot.
        d_row = (a[i - 1] - b[j_start - 1:j_end]) ** 2

        prev_row = cost[i - 1]
        cur_row = cost[i]
        cur_pred = pred[i]
        for offset, j in enumerate(range(j_start, j_end + 1)):
            diag = prev_row[j - 1]
            up = prev_row[j]
            left = cur_row[j - 1]
            # Explicit comparison, so an all-inf cell stays inf with pred = -1
            # instead of silently defaulting to the diagonal.
            best, direction = diag, _DIAG
            if up < best:
                best, direction = up, _UP
            if left < best:
                best, direction = left, _LEFT
            if not np.isfinite(best):
                continue
            cur_row[j] = d_row[offset] + best
            cur_pred[j] = direction

    total = cost[n, m]
    if not np.isfinite(total):
        return np.nan, []

    # The path is always walked: its length is needed for normalisation, so the
    # returned distance cannot depend on whether the caller wanted the path.
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        direction = pred[i, j]
        if direction == _DIAG:
            i, j = i - 1, j - 1
        elif direction == _UP:
            i = i - 1
        elif direction == _LEFT:
            j = j - 1
        else:
            # Unreachable for a finite total cost; guard against silent garbage.
            return np.nan, []
    path.reverse()

    if normalisation == "path":
        denominator = max(len(path), 1)
    elif normalisation == "sum":
        denominator = n + m
    else:
        denominator = 1
    distance = float(np.sqrt(total / denominator))

    return distance, (path if return_path else [])


def dtw_shape_and_amplitude(
    reference: np.ndarray,
    trace: np.ndarray,
    band_fraction: float = 0.10,
    return_path: bool = False,
) -> tuple[float, float, list[tuple[int, int]]]:
    """Shape distance (z-normalised) and amplitude distance (raw units).

    The alignment path returned is the *shape* path, which is what the residual
    calculation uses: alignment is decided on shape, and the residual then
    measures the amplitude difference at the aligned points.
    """
    shape, path = constrained_dtw(
        z_normalize(reference), z_normalize(trace),
        band_fraction=band_fraction, return_path=return_path,
    )
    amplitude, _ = constrained_dtw(
        reference, trace, band_fraction=band_fraction, return_path=False,
    )
    return shape, amplitude, path


def _self_check() -> None:
    """Assertions covering the two defects this module exists to fix."""
    rng = np.random.default_rng(0)
    a = np.sin(np.linspace(0, 6, 300))
    b = a + 0.05 * rng.standard_normal(300)

    with_path, path = constrained_dtw(a, b, return_path=True)
    without_path, empty = constrained_dtw(a, b, return_path=False)
    assert empty == []
    assert abs(with_path - without_path) < 1e-12, (
        f"distance must not depend on return_path: {with_path} vs {without_path}"
    )

    # Identical series align exactly along the diagonal at zero cost.
    same, diag_path = constrained_dtw(a, a, return_path=True)
    assert same < 1e-12, same
    assert len(diag_path) == 300, len(diag_path)

    # Symmetry.
    assert abs(constrained_dtw(a, b)[0] - constrained_dtw(b, a)[0]) < 1e-9

    # Path is monotone, contiguous and spans both series end to end.
    assert path[0] == (0, 0) and path[-1] == (299, 299)
    for (pi, pj), (ci, cj) in zip(path, path[1:]):
        step = (ci - pi, cj - pj)
        assert step in {(1, 1), (1, 0), (0, 1)}, step

    # Length invariance: comparing a signal to a resampled copy of itself
    # should stay near zero regardless of the length difference.
    half = np.sin(np.linspace(0, 6, 150))
    assert constrained_dtw(a, half)[0] < 0.05, constrained_dtw(a, half)[0]

    print("dtw_core self-check passed")


if __name__ == "__main__":
    _self_check()
