# Troubleshooting Guide

**Error → Cause → Fix patterns for common issues.**

---

## Quick Diagnostics

Before diving into specific errors, try these:

```bash
# 1. Verify environment is active
which python  # Should point to conda/venv, not system

# 2. Verify package is installed
python -c "from src.notebooks import create_interface; print('OK')"

# 3. Run quick validation
make quick-check
```

---

## Import Errors

### "ModuleNotFoundError: No module named 'src'"

**Cause**: Package not installed in development mode.

**Fix**:
```bash
pip install -e .
```

Or ensure you're running from the repository root.

### "ImportError: cannot import name 'X' from 'src.Y'"

**Cause**: Module reorganization or typo.

**Fix**:
1. Check the correct import path in `src/Y/__init__.py`
2. Run `grep -r "class X" src/` to find the definition

### "ModuleNotFoundError: No module named 'pandas'"

**Cause**: Dependencies not installed or wrong environment.

**Fix**:
```bash
# Verify environment
conda activate annuity-price-elasticity-v3

# Or reinstall dependencies
pip install -r requirements.txt
```

---

## Interface & Configuration Errors

### "NotImplementedError: Product type 'X' is not yet supported"

**Cause**: Trying to use an unsupported product type (e.g., MYGA).

**Fix**:
- Only RILA and FIA are currently supported
- Check product code: `6Y20B`, `6Y10B`, `10Y20B` for RILA
- See `src/config/product_registry.py` for supported products

### "ValueError: Unknown adapter type: X"

**Cause**: Invalid `environment` parameter.

**Fix**: Use one of:
- `"fixture"` - Test data (no AWS)
- `"aws"` - Production data
- `"local"` - Local file system

```python
# Correct
interface = create_interface("6Y20B", environment="fixture")

# Wrong
interface = create_interface("6Y20B", environment="test")  # Use "fixture" instead
```

### "ValueError: AWS adapter requires 'config' in adapter_kwargs"

**Cause**: Missing AWS configuration for production mode.

**Fix**:
```python
aws_config = {
    "sts_endpoint_url": "https://sts.us-east-1.amazonaws.com",
    "role_arn": "arn:aws:iam::ACCOUNT:role/ROLE",
    "xid": "your_user_id",
    "bucket_name": "bucket-name",
}

interface = create_interface(
    "6Y20B",
    environment="aws",
    adapter_kwargs={"config": aws_config}
)
```

---

## Causal Identification Errors

### "CRITICAL: Lag-0 competitor features in model"

**Cause**: Using `competitor_*_current` or `competitor_*_t0` features.

**Why it matters**: Lag-0 competitor features create simultaneity bias, violating causal identification.

**Fix**: Use lagged features only:
```python
# WRONG
features = ["prudential_rate_current", "competitor_mid_current"]

# RIGHT
features = ["prudential_rate_current", "competitor_mid_t2"]
```

See `knowledge/analysis/CAUSAL_FRAMEWORK.md` for the full explanation.

### "Data contains N lag-0 competitor columns"

**Cause**: Warning (not error) that data has lag-0 columns.

**Action**: This is informational. The data may include lag-0 for completeness, but ensure you don't use them in modeling.

---

## Data Errors

### "FileNotFoundError: Fixtures directory not found"

**Cause**: Wrong path to fixtures.

**Fix**:
```python
from pathlib import Path

# Check current working directory
import os
print(os.getcwd())

# Fixtures are in tests/fixtures/rila/
# If running from repo root, this should work:
interface = create_interface("6Y20B", environment="fixture")

# If running from a subdirectory, specify path:
interface = create_interface(
    "6Y20B",
    environment="fixture",
    adapter_kwargs={"fixtures_dir": Path("../../tests/fixtures/rila")}
)
```

### "ValueError: No data available. Call load_data() first"

**Cause**: Calling `run_inference()` before `load_data()`.

**Fix**:
```python
interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()  # Must call this first!
results = interface.run_inference(df)
```

### "CRITICAL: Sales fixture is empty"

**Cause**: Fixture file exists but has no data.

**Fix**:
1. Check the fixture file: `tests/fixtures/rila/raw_sales_data.parquet`
2. Verify it has data: `pd.read_parquet("tests/fixtures/rila/raw_sales_data.parquet")`
3. If empty, regenerate fixtures from AWS or use a different fixture

### "KeyError: 'column_name'"

**Cause**: Expected column missing from DataFrame.

**Fix**:
```python
# Check what columns exist
print(df.columns.tolist())

# Check if column has different name
similar = [c for c in df.columns if 'prudential' in c.lower()]
print(f"Similar columns: {similar}")
```

---

## Model & Inference Errors

### "RuntimeError: Inference failed in center_baseline"

**Cause**: Model fitting failed due to data issues.

**Diagnose**:
```python
# Check data quality
print(f"Shape: {df.shape}")
print(f"Nulls:\n{df.isnull().sum()}")
print(f"Target stats:\n{df['sales_target_current'].describe()}")

# Check features exist
for f in features:
    if f not in df.columns:
        print(f"MISSING: {f}")
```

**Common causes**:
1. Missing features in data
2. Too many null values
3. Perfect collinearity between features

### Constraint Violations

#### "Coefficient sign violation: competitor should be negative"

**Cause**: Model found positive coefficient for competitor rate.

**Diagnose**:
```python
# Check correlations
corr = df[["sales_target_current", "competitor_mid_t2", "prudential_rate_current"]].corr()
print(corr)

# High multicollinearity?
print(df[["competitor_mid_t2", "competitor_mid_t3"]].corr())
```

**Fixes**:
1. Remove highly correlated features
2. Use different lag (t3 instead of t2)
3. Check for data quality issues
4. Review sample size (may need more data)

#### "Coefficient sign violation: own rate should be positive"

**Cause**: Model found negative coefficient for own rate.

**This is economically wrong** (cap rate is a yield, not a price).

**Diagnose**:
1. Check if `prudential_rate_current` has correct data
2. Check for outliers in rate data
3. Verify date alignment between sales and rates

---

## Test Failures

### "FAILED test_equivalence - values differ by 1e-6"

**Cause**: Model output changed from baseline (may be acceptable).

**Diagnose**:
```python
# Compare values
import pandas as pd

baseline = pd.read_parquet("tests/baselines/rila/coefficients.parquet")
current_result = results["coefficients"]

for k, v in current_result.items():
    baseline_v = baseline[k].iloc[0]
    diff = abs(v - baseline_v)
    if diff > 1e-12:
        print(f"{k}: baseline={baseline_v:.10f}, current={v:.10f}, diff={diff:.2e}")
```

**Fix**:
- If change is intentional, update baseline
- If unexpected, investigate root cause

### "pytest: error: unrecognized arguments"

**Cause**: Wrong pytest syntax.

**Fix**:
```bash
# Correct
pytest tests/unit/test_interface.py -v

# Wrong
pytest -v tests/unit/test_interface.py  # Order matters for some flags
```

---

## AWS Errors

### "botocore.exceptions.NoCredentialsError"

**Cause**: AWS credentials not configured.

**Fix**: See `AWS_SETUP_GUIDE.md` for full setup.

Quick check:
```bash
# Verify credentials exist
aws sts get-caller-identity
```

### "Access Denied" or "403 Forbidden"

**Cause**: IAM role doesn't have required permissions.

**Required permissions**:
- `s3:GetObject`
- `s3:ListBucket`
- `sts:AssumeRole` (if using role assumption)

### "Connection timeout"

**Cause**: Network issues or VPN required.

**Fix**:
1. Check VPN connection if required
2. Verify endpoint URL is correct
3. Check firewall rules

---

## Performance Issues

### "Inference is very slow"

**Cause**: Large bootstrap iterations or data size.

**Fix**:
```python
# Reduce bootstrap for quick tests
results = interface.run_inference(
    df,
    config={"n_bootstrap": 100}  # Default is 1000
)
```

### "Out of memory"

**Cause**: Large dataset + many features.

**Fix**:
1. Reduce feature count
2. Sample data for testing
3. Use smaller bootstrap iterations

```python
# Sample for testing
df_sample = df.sample(n=100, random_state=42)
results = interface.run_inference(df_sample, config={"n_bootstrap": 50})
```

---

## Environment Issues

### "Kernel died" (Jupyter)

**Cause**: Memory overflow or package conflict.

**Fix**:
1. Restart kernel
2. Clear all outputs: Kernel → Restart & Clear Output
3. Check memory usage: `!free -h` (Linux) or Activity Monitor (Mac)

### "DLL load failed" (Windows)

**Cause**: Incompatible native libraries.

**Fix**:
```bash
# Reinstall numpy/scipy
pip uninstall numpy scipy
pip install numpy scipy
```

### Git-related errors

#### "Pre-commit hook failed"

**Cause**: Code doesn't pass quality checks.

**Fix**:
```bash
# Run the checks manually
make lint

# Fix issues
black src/
```

---

## Getting More Help

1. **Search existing docs**: `grep -r "error message" knowledge/`
2. **Check LESSONS_LEARNED**: `knowledge/integration/LESSONS_LEARNED.md`
3. **Ask Claude Code**: Use AI-assisted debugging
4. **Check git history**: `git log --oneline | head -20`

---

## Error Message Template

When asking for help, include:

```
## Error
[Exact error message]

## Context
- What I was trying to do: [task]
- Command/code that failed: [code]
- Environment: [conda env name, Python version]

## What I've tried
1. [First attempt]
2. [Second attempt]

## Relevant files
- [file paths that might be involved]
```
