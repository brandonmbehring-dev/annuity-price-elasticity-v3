# Notebook Output Baselines

Comprehensive intermediate output baselines for regression testing of all RILA notebooks at 1e-12 precision.

## Directory Structure

```
notebooks/
├── rila_6y20b/                     # 6Y20B product baselines
│   ├── nb00_data_pipeline/
│   │   ├── final_dataset.parquet   # (252, 562) final features
│   │   ├── weekly_aggregated.parquet
│   │   └── capture_metadata.json
│   ├── nb01_price_elasticity/
│   │   ├── 01_data_prep/           # Training data, feature matrices
│   │   ├── 02_bootstrap_model/     # Model coefficients, baseline forecast
│   │   ├── 03_rate_scenarios/      # Rate options, dollar/pct changes
│   │   ├── 04_confidence_intervals/
│   │   ├── 05_export/
│   │   └── capture_metadata.json
│   └── nb02_forecasting/
│       ├── forecast_results.parquet
│       ├── performance_metrics.json
│       └── capture_metadata.json
├── rila_1y10b/                     # 1Y10B product baselines (mirror structure)
│   ├── nb00_data_pipeline/
│   ├── nb01_price_elasticity/
│   └── nb02_forecasting/
├── nb00_data_pipeline/             # LEGACY - kept for compatibility
├── nb01_price_elasticity/          # LEGACY - migrated to rila_6y20b/
├── nb02_forecasting/               # LEGACY - migrated to rila_6y20b/
└── README.md
```

## Baseline Capture

Run the capture script to generate baselines from notebook execution:

```bash
# Full capture (all products, all notebooks)
python scripts/capture_notebook_baselines.py --all

# Specific product
python scripts/capture_notebook_baselines.py --product 6Y20B

# Specific notebook
python scripts/capture_notebook_baselines.py --product 6Y20B --notebook nb01
```

## Format Standards

- **DataFrames**: Parquet format (not pickle)
- **Arrays**: NumPy `.npy` format
- **Config/Metadata**: JSON format
- **Precision**: All numeric values validated at 1e-12
- **Random Seed**: Fixed seed=42 for bit-identical bootstrap results

## Reproducibility Requirements

1. **Fixed Random Seed**: All stochastic operations use `random_state=42`
2. **Deterministic Operations**: NaN handling, sorting order preserved
3. **Environment Lock**: Python 3.13, pandas 2.x, numpy 2.x
4. **Fixture-Based**: All baselines captured from fixture data (CI-compatible)

## Products Supported

| Product | Type | Buffer | Term | Status |
|---------|------|--------|------|--------|
| 6Y20B | RILA | 20% | 6 years | ✅ Complete |
| 1Y10B | RILA | 10% | 1 year | ⏳ Pending |

## Validation Tests

```bash
# Run notebook output equivalence tests
pytest tests/integration/test_notebook_equivalence.py -v

# Run with detailed diff on failure
pytest tests/integration/test_notebook_equivalence.py -v --tb=long

# Full test-all including equivalence
make test-all
```

## Capture Metadata Schema

Each notebook directory includes `capture_metadata.json`:

```json
{
  "notebook": "nb01_price_elasticity",
  "product": "6Y20B",
  "capture_timestamp": "2026-01-30T15:40:51.638375",
  "random_seed": 42,
  "python_version": "3.13",
  "pandas_version": "2.2.0",
  "numpy_version": "2.0.0",
  "outputs": {
    "filtered_data": {"shape": [159, 599], "dtype_summary": {...}},
    "model_coefficients": {"shape": [100, 4], "columns": ["coef_0", ...]}
  }
}
```

## Success Criteria

| Metric | Target |
|--------|--------|
| 6Y20B R² | 0.63712543970409 ± 1e-12 |
| 6Y20B MAPE | 13.593098809121642 ± 1e-12 |
| Bootstrap seed | 42 (bit-identical) |
| All numeric values | ± 1e-12 tolerance |

---
Generated: 2026-01-30
