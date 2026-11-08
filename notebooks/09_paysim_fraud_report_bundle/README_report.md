# PaySim experiment report bundle

Generated: 2026-11-02T06:23:48Z  
Runtime: Kaggle  
Dataset SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`  
Runs: 48 successful, 0 failed, 48 expected.

## Files

- `runs.csv`: per-run status, timing, validation/test metrics and split hashes.
- `predictions.csv`: validation/test probabilities for LightGBM and TabICLv2, with row indices and provenance hashes.
- `summary.csv`: paired single-model metric summary, when at least one complete model group exists.
- `ensemble_summary.csv`: per-seed/train-size blend metrics; `ensemble_metrics_summary.csv` aggregates these by train size.
- `lightgbm_tuning.csv`: validation scores for candidate LightGBM settings.
- `config.json`, `split_config.json`, `train_subsets.json`: protocol, runtime, temporal split, and exact sampled indices.
- `report_summary.json`: concise machine-readable report facts and caveats.
- `manifest.json`: SHA-256 and size for each bundled file, except the manifest itself.

Raw PaySim data is intentionally excluded. The dataset is simulated; do not present results as evidence of real-world bank fraud performance. Check `completed_runs` against `expected_runs` before using results as a complete experiment. The notebook uses CPU for TabICLv2 irrespective of Kaggle accelerator selection.
