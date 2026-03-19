# IWLS Validation Experiments

This experiment package validates hypotheses H1-H3 for stability-aware source selection under covariate shift.

## Goals
- Reproduce synthetic A/B/C setting comparisons with no-target-label selection.
- Benchmark stability-aware ranking against discrepancy-based, pooled-IWLS, and random baselines.
- Run pooled-focused ablations over stabilization controls (alpha/beta/gamma, clipping, ridge, ratio estimator, candidate pool size).
- Generate paper-ready PDF figures, CSV tables, and SymPy validation reports.

## Setup
- Python environment: `experiments/.venv`
- Install dependencies:
  - `uv pip install --python experiments/.venv/bin/python numpy pandas matplotlib seaborn scipy sympy pytest ruff mypy pymupdf`

## Run
- `PYTHONPATH=experiments/iwls_validation/src experiments/.venv/bin/python experiments/iwls_validation/run_experiments.py --config experiments/iwls_validation/configs/default.json --output-dir experiments/iwls_validation/output --paper-fig-dir paper/figures --paper-table-dir paper/tables --paper-data-dir paper/data`

## Outputs
- `experiments/iwls_validation/output/results_summary.json`
- `experiments/iwls_validation/output/sympy_validation_report.txt`
- `paper/figures/fig_main_results.pdf`
- `paper/figures/fig_stability_tradeoff.pdf`
- `paper/figures/fig_ablation_pooled_gap.pdf`
- `paper/tables/table_main_metrics.csv`
- `paper/tables/table_significance.csv`
- `paper/tables/table_ablation_vs_pooled.csv`
- `paper/data/iwls_results.csv`

## Real-data hooks
- `real_settings` in config accepts locally prepared CSVs converted from AdaTime/UDA-4-TSC-style tasks.
- Required keys per real setting: `name`, `source_paths`, `target_unlabeled_path`, `target_test_path`, `feature_columns`, `target_column`.
- Missing files or schema mismatch are reported in `results_summary.json` under `real_data_warnings`.

## Statistical method
Significance uses Shapiro-Wilk normality check, then paired t-test if normal otherwise paired Wilcoxon; Holm correction is applied across method comparisons and across pooled-focused ablation configurations.

## Safety checks
- Config schema is validated strictly and rejects wrapped recovery envelopes (`payload/artifacts/notes/...`) to prevent upstream malformed inputs from silently executing.
