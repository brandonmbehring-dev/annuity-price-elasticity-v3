# Notebook Quick Start Guide

**Time to first analysis: <10 minutes**

This guide answers the most common question new developers ask: **"Which notebook do I run first?"**

## TL;DR - Start Here

**For RILA 6Y20B analysis (most common):**

1. Open: `notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb`
2. Run all cells (`Ctrl+A` then `Shift+Enter`)
3. Wait 2-5 minutes for results
4. Check output: `BI_TEAM/price_elasticity_FlexGuard_*.png`

**That's it.** You just ran a complete price elasticity analysis.

---

## Quick Decision Tree

```
START: What do you want to do?
│
├─ Run production inference for RILA 6Y20B?
│  └─ Go to: notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb
│
├─ Run production inference for RILA 1Y10B?
│  └─ Go to: notebooks/production/rila_1y10b/01_price_elasticity_inference.ipynb
│
├─ Understand the architecture?
│  └─ Go to: notebooks/onboarding/architecture_walkthrough.ipynb
│
├─ Explore data or try new features?
│  └─ Go to: notebooks/eda/rila_6y20b/ (any notebook)
│
└─ Check historical analysis?
   └─ Go to: notebooks/archive/ (read-only reference)
```

---

## Notebook Organization

The notebooks are organized into three directories with clear purposes:

###  `production/` - Production-Ready Analysis

**Purpose:** Stable, tested notebooks used for business-critical analysis.

**Directory Structure:**
```
production/
├── rila_6y20b/          # RILA 6-Year 20% Buffer product
│   ├── 00_data_pipeline.ipynb
│   ├── 01_price_elasticity_inference.ipynb  ⭐ START HERE
│   └── 02_time_series_forecasting.ipynb
├── rila_1y10b/          # RILA 1-Year 10% Buffer product
│   ├── 00_data_pipeline.ipynb
│   ├── 01_price_elasticity_inference.ipynb
│   └── 02_time_series_forecasting.ipynb
└── fia/                 # Fixed Index Annuity (if configured)
    └── 00_RUNME_PE_FIA_v2_1.ipynb
```

**Typical Workflow:**
1. **Skip `00_data_pipeline.ipynb`** - Outputs already exist in `outputs/datasets/`
2. **Run `01_price_elasticity_inference.ipynb`** - Main analysis (2-5 minutes)
3. **Optional: `02_time_series_forecasting.ipynb`** - Future projections (5-10 minutes)

**When to run `00_data_pipeline.ipynb`:**
- Fresh data available from AWS
- Need to update fixture data
- Troubleshooting data quality issues

---

###  `eda/` - Exploratory Data Analysis

**Purpose:** Exploratory notebooks for data investigation and feature development.

**Directory Structure:**
```
eda/
└── rila_6y20b/
    ├── 01_EDA_sales_RILA.ipynb              # Sales patterns and trends
    ├── 02_EDA_rates_RILA.ipynb              # Competitive rate analysis
    ├── 03_EDA_RILA_feature_engineering.ipynb # Feature engineering experiments
    ├── 04_RILA_feature_selection.ipynb       # Feature selection methods
    └── 05_RILA_Time_Forward_Cross_Validation.ipynb # Model validation
```

**Who Should Use EDA Notebooks:**
- Data scientists exploring new features
- Analysts investigating sales patterns
- Developers debugging feature engineering
- Researchers validating model assumptions

**Status:** Exploratory - may contain experimental code and incomplete analysis

---

###  `archive/` - Historical Reference

**Purpose:** Historical notebooks preserved for reference and reproducibility.

**What's Here:**
- Development snapshots during refactoring
- Original implementations before TDD refactor
- Executed notebooks with timestamps (e.g., `*_executed_20260126.ipynb`)

**Usage:** Read-only reference. Do not modify or run these notebooks.

---

## Understanding OFFLINE_MODE

**Important:** The `OFFLINE_MODE` toggle is in `notebooks/production/*/00_data_pipeline.ipynb` (cell 3), not in the inference notebooks.

### What is OFFLINE_MODE?

```python
# In 00_data_pipeline.ipynb:
OFFLINE_MODE = True   # Use local fixture data
OFFLINE_MODE = False  # Fetch from AWS S3
```

**Default: `True` (fixture mode)** - Uses pre-loaded test data, no AWS credentials needed.

### Decision Tree: AWS vs Fixture

```
Do you have AWS credentials configured?
│
YES → Do you need the latest production data?
│     │
│     YES → Set OFFLINE_MODE = False in 00_data_pipeline.ipynb
│     │     Run entire notebook
│     │     Then run 01_price_elasticity_inference.ipynb
│     │
│     NO → Keep OFFLINE_MODE = True (default)
│            Skip 00_data_pipeline.ipynb
│            Run 01_price_elasticity_inference.ipynb directly
│
NO → Keep OFFLINE_MODE = True (default)
     Skip 00_data_pipeline.ipynb
     Run 01_price_elasticity_inference.ipynb directly
```

### How Mode Affects Notebooks

| Notebook | OFFLINE_MODE = True | OFFLINE_MODE = False |
|----------|---------------------|----------------------|
| `00_data_pipeline.ipynb` | Loads `data/fixture/*.parquet` | Fetches from AWS S3 |
| `01_price_elasticity_inference.ipynb` | Reads `outputs/datasets/*.parquet` | Reads `outputs/datasets/*.parquet` |
| `02_time_series_forecasting.ipynb` | Reads `outputs/datasets/*.parquet` | Reads `outputs/datasets/*.parquet` |

**Key Insight:** The inference notebooks (01, 02) **automatically use whatever data** the pipeline notebook (00) generated. You don't set mode in inference notebooks.

### Switching Modes

**To switch from fixture to AWS:**
1. Open `notebooks/production/rila_6y20b/00_data_pipeline.ipynb`
2. Change cell 3: `OFFLINE_MODE = False`
3. Run entire notebook (produces new `outputs/datasets/*.parquet`)
4. Re-run `01_price_elasticity_inference.ipynb` (now uses AWS data)

**To switch from AWS to fixture:**
1. Open `notebooks/production/rila_6y20b/00_data_pipeline.ipynb`
2. Change cell 3: `OFFLINE_MODE = True`
3. Run entire notebook (produces new `outputs/datasets/*.parquet`)
4. Re-run `01_price_elasticity_inference.ipynb` (now uses fixture data)

---

## Common Workflows

### Workflow 1: Quick Test Run (5 minutes)

**Goal:** Verify installation and see example output

```bash
# 1. Open notebook
notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb

# 2. Run all cells (uses existing fixture data)
# Menu: Run → Run All Cells
# Or: Select all cells (Ctrl+A) then Shift+Enter

# 3. Check outputs
BI_TEAM/price_elasticity_FlexGuard_Sample_*.png
BI_TEAM/price_elasticity_FlexGuard_*.csv
```

**Expected Runtime:** 2-5 minutes

**Expected Output:**
- 2 PNG visualizations (percentage and dollar impacts)
- 8 CSV files for BI analysis
- Console output showing model metrics

---

### Workflow 2: Production Analysis with AWS Data (30 minutes)

**Goal:** Run analysis with latest production data

```bash
# 1. Configure AWS credentials (if not done)
aws configure  # Or set AWS_PROFILE

# 2. Run data pipeline
notebooks/production/rila_6y20b/00_data_pipeline.ipynb
# Set OFFLINE_MODE = False
# Run all cells

# 3. Run inference
notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb
# Run all cells

# 4. Optional: Run forecasting
notebooks/production/rila_6y20b/02_time_series_forecasting.ipynb
# Run all cells
```

**Expected Runtime:**
- Data pipeline: 5-15 minutes
- Inference: 2-5 minutes
- Forecasting: 5-10 minutes

---

### Workflow 3: Explore Sales Patterns (20 minutes)

**Goal:** Understand RILA sales trends and competitive dynamics

```bash
# 1. Sales exploration
notebooks/eda/rila_6y20b/01_EDA_sales_RILA.ipynb

# 2. Competitive rate analysis
notebooks/eda/rila_6y20b/02_EDA_rates_RILA.ipynb

# 3. Feature engineering insights
notebooks/eda/rila_6y20b/03_EDA_RILA_feature_engineering.ipynb
```

---

### Workflow 4: Architecture Learning (30 minutes)

**Goal:** Understand the codebase structure and design patterns

```bash
# 1. Architecture overview
notebooks/onboarding/architecture_walkthrough.ipynb

# 2. Read documentation
docs/onboarding/MENTAL_MODEL.md
docs/onboarding/GETTING_STARTED.md
```

---

## Expected Outputs

### From `01_price_elasticity_inference.ipynb`

**Console Output:**
```
RILA Price Elasticity Configuration Loaded
Target Variable: sales_target_current
Model Features: 4 features
Bootstrap Estimators: 1,000
...
[PASS] Random seed initialized: 42
...
Bootstrap ensemble training completed
...
RILA PRICE ELASTICITY ANALYSIS - EXECUTION COMPLETE
```

**Files Created:**
```
BI_TEAM/
├── price_elasticity_FlexGuard_Sample_2026-01-28.png        # Percentage chart
├── price_elasticity_FlexGuard_Dollars_Sample_2026-01-28.png # Dollar chart
├── price_elasticity_FlexGuard_bootstrap_distributions_2026-01-28.csv
├── price_elasticity_FlexGuard_confidence_intervals_2026-01-28.csv
└── [6 more CSV files...]
```

**Visual Output:**
- Chart 1: Price elasticity confidence intervals (percentage change)
  - X-axis: Rate change (0-450 basis points)
  - Y-axis: Sales impact (percentage)
  - Shows 95% confidence intervals with bootstrap distributions

- Chart 2: Price elasticity confidence intervals (dollar impact)
  - X-axis: Rate change (0-450 basis points)
  - Y-axis: Revenue impact (dollars)
  - Shows 95% confidence intervals with bootstrap distributions

---

## Troubleshooting

### Error: `FileNotFoundError: outputs/datasets/final_dataset.parquet`

**Cause:** Data pipeline hasn't been run yet.

**Fix:**
```bash
# Option 1: Run data pipeline
notebooks/production/rila_6y20b/00_data_pipeline.ipynb

# Option 2: Verify fixture data exists
ls data/fixture/rila_6y20b.parquet
```

---

### Error: `ModuleNotFoundError: No module named 'src'`

**Cause:** Package not installed in editable mode.

**Fix:**
```bash
cd /home/sagemaker-user/RILA_6Y20B_refactored
pip install -e .
```

---

### Error: `NoCredentialsError` (AWS)

**Cause:** OFFLINE_MODE = False but no AWS credentials configured.

**Fix:**
```bash
# Option 1: Configure AWS credentials
aws configure

# Option 2: Switch to fixture mode
# In 00_data_pipeline.ipynb, set:
OFFLINE_MODE = True
```

---

### Notebook Takes Forever to Run

**Cause:** Large bootstrap sample size or complex feature engineering.

**Reduce Runtime:**
```python
# In config file or notebook configuration cell:
n_estimators = 1000  # Default
n_estimators = 100   # Faster testing (10x speedup)
```

**Note:** Reduced bootstrap samples decrease confidence interval precision.

---

### Visualizations Don't Appear

**Cause:** Missing matplotlib backend or notebook display issue.

**Fix:**
```python
# Add to first cell:
%matplotlib inline
import matplotlib.pyplot as plt
```

---

## Next Steps

**After running your first notebook:**

- **Understand the code:** [docs/onboarding/MENTAL_MODEL.md](./MENTAL_MODEL.md)
- **Common tasks:** [docs/onboarding/COMMON_TASKS.md](./COMMON_TASKS.md)
- **Configuration:** [docs/reference/CONFIGURATION_REFERENCE.md](../reference/CONFIGURATION_REFERENCE.md)
- **API reference:** [docs/api/index.html](../api/index.html)

**For further support:**
- Team Slack: #rila-elasticity-support
- Documentation: [docs/](../../docs/)
- GitHub Issues: [Report a bug](https://github.com/your-org/RILA_6Y20B_refactored/issues)

---

## Quick Reference

**Primary notebook:** `notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb`

**Expected runtime:** 2-5 minutes (fixture mode)

**Key outputs:** `BI_TEAM/*.png` and `BI_TEAM/*.csv`

**Mode toggle:** `notebooks/production/*/00_data_pipeline.ipynb` (cell 3)

**Documentation:** See [docs/onboarding/](./README.md) for complete guides
