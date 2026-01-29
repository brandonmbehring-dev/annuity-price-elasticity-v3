# Notebook Output Baselines

Comprehensive intermediate output baselines for regression testing of all RILA notebooks at 1e-12 precision.

## Directory Structure

```
notebooks/
├── nb00_data_pipeline/         # Symlink to ../aws_mode/ (existing baselines)
├── nb01_price_elasticity/      # NB01 intermediate outputs
│   ├── 01_data_prep/           # Training data, feature matrices
│   ├── 02_bootstrap_model/     # Model coefficients, baseline forecast
│   ├── 03_rate_scenarios/      # Rate options, dollar/pct changes
│   ├── 04_confidence_intervals/# CI outputs
│   ├── 05_export/              # BI export, metadata
│   └── capture_metadata.json
├── nb02_forecasting/           # NB02 intermediate outputs
│   ├── 01_data_prep/           # Training/test features
│   ├── 02_model_training/      # Ridge params, CV scores
│   ├── 03_predictions/         # Predictions, intervals
│   ├── 04_metrics/             # Model/benchmark metrics
│   ├── 05_export/              # Tableau export
│   └── capture_metadata.json
└── README.md
```

## Baseline Capture

Run the capture script to generate baselines from notebook execution:

```bash
# Full capture (all notebooks)
python scripts/capture_notebook_baselines.py

# Specific notebook
python scripts/capture_notebook_baselines.py --notebook nb01

# Convert existing pickles to parquet
python scripts/capture_notebook_baselines.py --convert-pickles
```

## Format Standards

- **DataFrames**: Parquet format (not pickle)
- **Arrays**: NumPy `.npy` format
- **Config/Metadata**: JSON format
- **Precision**: All numeric values validated at 1e-12

## Reproducibility Requirements

1. **Fixed Random Seed**: All stochastic operations use `random_state=42`
2. **Deterministic Operations**: NaN handling, sorting order preserved
3. **Environment Lock**: Python 3.13, pandas 2.x, numpy 2.x

## Relationship to Existing Baselines

| Location | Content | Status |
|----------|---------|--------|
| `baselines/aws_mode/` | NB00 stages 01-10 | AUTHORITATIVE |
| `baselines/nb01/` | NB01 final outputs | AUTHORITATIVE |
| `baselines/nb02/` | NB02 final outputs | AUTHORITATIVE |
| `baselines/notebooks/` | Consolidated + intermediates | NEW |
| `notebooks/tests/reference_outputs/` | Legacy pickles | DEPRECATED |

## Validation Tests

```bash
# Run notebook output equivalence tests
pytest tests/test_notebook_output_equivalence.py -v

# Run with detailed diff on failure
pytest tests/test_notebook_output_equivalence.py -v --tb=long
```

## Capture Metadata Schema

Each notebook directory includes `capture_metadata.json`:

```json
{
  "notebook": "nb01_price_elasticity",
  "capture_timestamp": "2026-01-16T15:40:51.638375",
  "random_seed": 42,
  "python_version": "3.13",
  "pandas_version": "2.2.0",
  "outputs": {
    "filtered_data": {"shape": [159, 599], "columns": ["date", "sales", ...]},
    "model_coefficients": {"shape": [100, 4], "columns": ["coef_0", ...]}
  }
}
```

## Adding New Baselines

1. Modify notebook to export intermediate output
2. Add capture logic to `scripts/capture_notebook_baselines.py`
3. Add fixture to `tests/conftest.py`
4. Add test case to `tests/test_notebook_output_equivalence.py`
5. Re-run capture script

---
Generated: 2026-01-16
