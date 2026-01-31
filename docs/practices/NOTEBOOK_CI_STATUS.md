# Notebook CI Integration Status

**Last Updated:** 2026-01-30
**Status:** ✅ ALL 7 NOTEBOOKS RUNNING IN CI

---

## Current CI Coverage

| Notebook | CI Status | Data Source |
|----------|-----------|-------------|
| `architecture_walkthrough.ipynb` | ✅ Runs in CI | Fixtures |
| `rila_6y20b/00_data_pipeline.ipynb` | ✅ Runs in CI | Fixtures |
| `rila_6y20b/01_price_elasticity_inference.ipynb` | ✅ Runs in CI | Fixtures via transform |
| `rila_6y20b/02_time_series_forecasting.ipynb` | ✅ Runs in CI | Fixtures via transform |
| `rila_1y10b/00_data_pipeline.ipynb` | ✅ Runs in CI | Fixtures |
| `rila_1y10b/01_price_elasticity_inference.ipynb` | ✅ Runs in CI | Fixtures via transform |
| `rila_1y10b/02_time_series_forecasting.ipynb` | ✅ Runs in CI | Fixtures via transform |

**Result:** 7 of 7 production notebooks run in CI (100%)

---

## Implementation Details

### Task 5.2 Solution: Minimal OFFLINE_MODE Fix

Rather than a full DI refactor, we implemented a minimal fix:

1. **NB00 notebooks** already had `OFFLINE_MODE` toggle - we made it functional:
   - Added fixture loading paths for raw data (sales, WINK, market weights, economic indicators)
   - Added `if not OFFLINE_MODE:` guard around `save_dataset(df_weekly_final, "final_dataset")` to avoid overwriting CI fixtures

2. **NB01/NB02 notebooks** load from `outputs/datasets/`:
   - `scripts/setup_notebook_fixtures.py` copies and transforms fixture data
   - Column renaming: `competitor_weighted_*` → `competitor_mid_*` (legacy compatibility)

3. **Makefile integration**:
   - `make setup-notebook-fixtures` - Prepares transformed fixture data
   - `make test-notebooks` - Validates all 7 notebooks
   - `make test-all` - Unit tests + notebooks (CI target)

### Fixture Setup Script

`scripts/setup_notebook_fixtures.py`:
- Copies `final_weekly_dataset.parquet` → `outputs/datasets/final_dataset.parquet`
- Applies column renaming for legacy compatibility
- Copies `market_weighted_competitive_rates.parquet` → `outputs/datasets/WINK_competitive_rates.parquet`

### Key Files Modified

1. **`notebooks/production/rila_6y20b/00_data_pipeline.ipynb`**
   - `OFFLINE_MODE = True` (CI default)
   - Fixture loading for all data sources
   - Skip `save_dataset` for final_dataset in OFFLINE_MODE

2. **`notebooks/production/rila_1y10b/00_data_pipeline.ipynb`**
   - Same changes as 6Y20B

3. **`src/config/builders/pipeline_builders.py`**
   - Fixed column names: `prudential_rate_t0` → `prudential_rate_current`
   - Fixed column names: `sales_target_contract_t0` → `sales_target_contract_current`

4. **`Makefile`**
   - Updated `setup-notebook-fixtures` to use Python script
   - Updated `test-notebooks` to include all 7 notebooks

---

## Commands

```bash
# Full notebook validation
make test-notebooks        # 7 notebooks, ~70 seconds

# Full CI suite (unit tests + notebooks)
make test-all

# AWS-only (live data)
make test-notebooks-aws
```

---

## Fixture Files Used

| Data Type | Fixture File | Usage |
|-----------|--------------|-------|
| Raw Sales | `raw_sales_data.parquet` | NB00 in OFFLINE_MODE |
| Raw WINK | `raw_wink_data.parquet` | NB00 in OFFLINE_MODE |
| Market Weights | `market_share_weights.parquet` | NB00 in OFFLINE_MODE |
| DGS5 | `economic_indicators/dgs5.parquet` | NB00 in OFFLINE_MODE |
| VIXCLS | `economic_indicators/vixcls.parquet` | NB00 in OFFLINE_MODE |
| CPI | `economic_indicators/cpi.parquet` | NB00 in OFFLINE_MODE |
| Final Dataset | `final_weekly_dataset.parquet` | NB01/NB02 via transform |
| WINK Rates | `market_weighted_competitive_rates.parquet` | NB01/NB02 |

---

## Related Documentation

- `CLAUDE.md` - Testing architecture section
- `docs/MODE_TOGGLE_GUIDE.md` - OFFLINE_MODE documentation
- `tests/fixtures/rila/capture_metadata.json` - Fixture capture details
