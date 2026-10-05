# CNC Tool-Wear Signal Attribution from Spindle-Current Telemetry

This repository contains the analysis pipeline developed for an MSc Data Science dissertation on distinguishing gradual tool-degradation signals from confounding variation in production CNC spindle-current telemetry.

The project studies a practical predictive-maintenance problem: spindle current can respond to tool wear, but it also changes with thermal state, material/setup changes, machine drift and other production effects. Because the source data contained no reliable ground-truth record of tool condition, the project was framed as evidence-based attribution rather than supervised remaining-useful-life prediction.

## What the Pipeline Does

The code implements a reproducible nine-stage Python workflow:

1. Parse raw CNC telemetry files and segment machining cycles.
2. Derive per-machine cutting thresholds and run-level features.
3. Identify comparable machine-program-tool windows.
4. Build comparable time-series datasets for each retained window.
5. Compute band-constrained dynamic time warping features and phase-resolved residuals.
6. Detect trends, changepoints and cohort-relative behaviour.
7. Build robust evidence with autocorrelation-aware inference and false-discovery-rate correction.
8. Score candidate events and assign confidence bands.
9. Resolve algorithmic labels and project them back to timestamp-level rows.

The pipeline also includes evaluation scripts for scalar-feature baselines, phase localisation, leave-one-tool-out specificity, feasibility checks and automated verification.

## Key Methods

- Python data pipelines with Pandas, NumPy and SciPy
- Dynamic Time Warping with a Sakoe-Chiba band constraint
- Leave-one-out median reference templates
- Kendall rank correlation and Theil-Sen trend estimation
- Circular block bootstrap for serial dependence
- False-discovery-rate correction for multiple testing
- Cohort-relative drift analysis to separate tool-specific signals from common-mode effects
- Reproducibility manifests and fixed random seeds

## Results Summary

On the dissertation dataset, the workflow segmented more than one million production telemetry samples into machining cycles and scored 1,252 cycle-tool segments. Correcting for serial dependence reduced apparent rising degradation trends from 11 windows to 4; cohort-relative attribution further reduced these to 3 tool-attributable cases.

The project also found that scalar current summaries matched or exceeded DTW-derived residuals for detection, while the alignment-based representation remained useful for diagnostic analysis of signal structure.

## Data Availability

The raw dataset is not included. It was supplied under a commercial agreement and is not redistributable.

To run the pipeline, place compatible CSV telemetry files in a local `data/` directory with the expected CNC telemetry fields. The `.gitignore` intentionally excludes raw data, processed outputs, generated figures and reports.

## Repository Structure

```text
.
├── scripts/
│   ├── run_pipeline.py
│   ├── preprocess_cnc.py
│   ├── build_dtw_features.py
│   ├── build_robust_event_evidence.py
│   ├── predict_toolwear.py
│   └── ...
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_pipeline.py --data-dir data --suffix all_best --all-windows
```

Additional analyses:

```bash
python scripts/predict_toolwear.py
python scripts/baseline_benchmark.py
python scripts/phase_localisation_test.py
python scripts/feasibility_checks.py
python scripts/verify_fixes.py
```

## Notes

The labels produced by this workflow are evidence-based hypotheses, not validated tool-condition ground truth. They should be interpreted as suspected explanatory categories unless independently reviewed by domain experts.

