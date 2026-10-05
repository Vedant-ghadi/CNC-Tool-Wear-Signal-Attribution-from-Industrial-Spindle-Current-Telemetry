"""Phase-resolved trend tests on raw and mean-removed DTW residuals.

Answers RQ4.  Two questions are separated here that the original phase analysis
conflated:

1. **Resolution.**  A resampled p-value can be no smaller than 1/(B+1).  With
   400 phase tests, Benjamini-Hochberg at q <= 0.05 requires the smallest
   p-value to reach 0.05/400 = 1.25e-4, which is unreachable unless
   B >= 8000.  The earlier analysis used B = 500, so after the (count+1)/(B+1)
   correction no phase test *could* be significant regardless of the data.
   This script uses B = 10000.

2. **Confounding by the run-level offset.**  Residuals are computed against a
   z-normalised alignment, so a run-level shift in load enters every phase bin
   equally.  If localisation is real it must survive subtracting each run's own
   mean residual before binning.  Both versions are computed and compared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes

from pipeline_common import benjamini_hochberg, block_bootstrap_trend


def phase_trends(residuals: pd.DataFrame, column: str, bins: int,
                 n_resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    grouped = residuals.groupby(
        ["comparable_window_id", "machine_id", "program", "tool", "phase_bin"],
        observed=True)
    for key, group in grouped:
        window_id, machine_id, program, tool, phase_bin = key
        by_run = group.groupby("run_order", sort=True)[column].mean().reset_index()
        x = by_run["run_order"].to_numpy(dtype=float)
        y = by_run[column].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 6:
            continue
        slope, _, _, _ = theilslopes(y[valid], x[valid], alpha=0.95)
        tau, p = kendalltau(x[valid], y[valid])
        block = block_bootstrap_trend(y[valid], n_resamples=n_resamples, seed=seed)
        rows.append({
            "comparable_window_id": int(window_id), "machine_id": int(machine_id),
            "program": program, "tool": tool, "phase_bin": str(phase_bin),
            "runs": int(valid.sum()), "series": column,
            "theil_sen_slope": float(slope),
            "kendall_tau": float(tau), "kendall_p": float(p),
            "block_p": block["p"], "lag1": block["lag1"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["kendall_q"] = benjamini_hochberg(out["kendall_p"])
    out["block_q"] = benjamini_hochberg(out["block_p"])
    out["rising"] = out["block_q"].le(0.05) & out["theil_sen_slope"].gt(0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtw-dir", type=Path,
                        default=Path("processed/dtw_features_all_best"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("processed/phase_localisation"))
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    residuals = pd.read_csv(args.dtw_dir / "cnc_dtw_residuals_long.csv.gz", low_memory=False)
    for col in ["program", "tool"]:
        residuals[col] = residuals[col].astype(str)
    residuals["phase_bin"] = pd.cut(
        residuals["template_phase"], bins=np.linspace(0, 1, args.bins + 1),
        include_lowest=True)

    # The run-level offset: mean residual across ALL phase bins of that run.
    run_offset = residuals.groupby(
        ["comparable_window_id", "run_order"])["residual"].transform("mean")
    residuals["residual_centred"] = residuals["residual"] - run_offset

    raw = phase_trends(residuals, "residual", args.bins, args.resamples, args.seed)
    centred = phase_trends(residuals, "residual_centred", args.bins, args.resamples, args.seed)
    raw.to_csv(args.output_dir / "phase_trends_raw.csv", index=False)
    centred.to_csv(args.output_dir / "phase_trends_mean_removed.csv", index=False)

    def profile(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby("comparable_window_id").agg(
            machine_id=("machine_id", "first"), tool=("tool", "first"),
            bins=("kendall_tau", "size"),
            rising_bins=("rising", "sum"),
            mean_abs_tau=("kendall_tau", lambda s: float(np.mean(np.abs(s)))),
            tau_min=("kendall_tau", "min"), tau_max=("kendall_tau", "max"),
            all_same_sign=("kendall_tau", lambda s: bool((s > 0).all() or (s < 0).all())),
        )

    raw_profile, centred_profile = profile(raw), profile(centred)
    comparison = raw_profile.join(centred_profile, lsuffix="_raw", rsuffix="_centred")
    comparison.to_csv(args.output_dir / "phase_comparison.csv")

    min_p = 1.0 / (args.resamples + 1)
    bh_needed = 0.05 / max(len(raw), 1)
    lines = [
        "PHASE-RESOLVED TREND TESTS  (RQ4)",
        "",
        f"resamples per test: {args.resamples:,}   smallest attainable p: {min_p:.2e}",
        f"tests in family: {len(raw)}   smallest p needed for BH at q<=0.05: {bh_needed:.2e}",
        f"resolution adequate: {'YES' if min_p <= bh_needed else 'NO'}",
        "",
        f"{'':32}{'RAW':>10}{'MEAN-REMOVED':>16}",
        f"{'phase bins rising (block+FDR)':32}{int(raw['rising'].sum()):>10}{int(centred['rising'].sum()):>16}",
        f"{'raw p <= 0.05':32}{int(raw['kendall_p'].le(0.05).sum()):>10}{int(centred['kendall_p'].le(0.05).sum()):>16}",
        f"{'naive FDR q <= 0.05':32}{int(raw['kendall_q'].le(0.05).sum()):>10}{int(centred['kendall_q'].le(0.05).sum()):>16}",
        f"{'mean |tau| across all bins':32}{np.mean(np.abs(raw['kendall_tau'])):>10.3f}"
        f"{np.mean(np.abs(centred['kendall_tau'])):>16.3f}",
        f"{'windows with all bins same sign':32}"
        f"{int(raw_profile['all_same_sign'].sum()):>10}{int(centred_profile['all_same_sign'].sum()):>16}",
        "",
        "INTERPRETATION",
        "  If phase structure is real, mean-removed bins retain trend.",
        "  If it is the run-level offset re-expressed, mean removal collapses it.",
    ]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "phase_localisation_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
