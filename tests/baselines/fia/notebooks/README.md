# FIA Notebook Baselines

**Created**: 2026-01-25
**Source**: FIA notebook execution fixtures from 2026-01-09
**Purpose**: Enable mathematical equivalence testing for FIA notebooks

---

## Directory Structure

```
tests/baselines/fia/
├── reference/
│   ├── metadata.json           # Capture metadata
│   └── intermediates/          # Pipeline stage outputs
│       ├── 01_filtered_sales.parquet     # Cleaned PruSecure sales
│       ├── 02_daily_sales.parquet        # Daily time series
│       ├── 03_competitive_rates.parquet  # WINK competitive rates
│       ├── 04_integrated_data.parquet    # Merged data
│       ├── 05_weekly_aggregated.parquet  # Weekly rollup
│       └── 06_final_dataset.parquet      # Model-ready features
└── notebooks/
    └── nb01_price_elasticity/  # Inference outputs
        ├── bootstrap_distributions_pct.parquet
        ├── bootstrap_distributions_dollars.parquet
        ├── confidence_intervals_pct.parquet
        ├── confidence_intervals_dollars.parquet
        └── bi_export.parquet
```

## Usage

```python
import pandas as pd
from pathlib import Path

baseline_path = Path('tests/baselines/fia/reference/intermediates')

# Load baseline for comparison
baseline = pd.read_parquet(baseline_path / '06_final_dataset.parquet')

# Compare with new implementation
result = run_fia_pipeline()
pd.testing.assert_frame_equal(result, baseline, atol=1e-12)
```

## Validation Requirements

For FIA production deployment:
1. All intermediate stages must match baselines at 1e-12 precision
2. Inference outputs must match baseline distributions
3. Leakage checklist must pass (no lag-0 competitors)
