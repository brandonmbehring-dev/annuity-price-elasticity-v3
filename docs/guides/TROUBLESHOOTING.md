# Troubleshooting Guide

**Purpose**: Quick solutions to common issues, organized by user pain point.
**Last Updated**: 2026-01-31

---

## Model Performance Issues

### "My elasticity estimate is implausibly high"

**Likely cause**: Lag-0 competitor features or aggregation lookahead.

**Diagnosis checklist**:
- [ ] Check feature names for `_t0`, `_current`, `_lag_0` patterns
- [ ] Verify aggregation computed within training window only
- [ ] Run shuffled target test (expect AUC ~0.50)

**Quick check**:
```python
from src.validation.leakage_gates import detect_lag0_features
forbidden = detect_lag0_features(model.feature_names_)
print(f"Lag-0 features: {forbidden}")
```

**Solution**: See [Episode 01: Lag-0 Competitor Rates](../knowledge/episodes/episode_01_lag0_competitor_rates.md)

---

### "My own-rate coefficient is NEGATIVE"

**Likely cause**: Product mix confounding (Simpson's Paradox) or endogeneity.

**Diagnosis checklist**:
- [ ] Are multiple products pooled in one model?
- [ ] Does coefficient sign flip when running by-product?
- [ ] Are demand controls included?

**Quick check**:
```python
# Check by product
for product in df['product'].unique():
    subset = df[df['product'] == product]
    corr = subset['prudential_rate'].corr(subset['sales'])
    print(f"{product}: {corr:.3f}")
```

**Solutions**:
- [Episode 04: Product Mix Confounding](../knowledge/episodes/episode_04_product_mix_confounding.md)
- [Episode 10: Own Rate Endogeneity](../knowledge/episodes/episode_10_own_rate_endogeneity.md)

---

### "Model R² is suspiciously high (>85%)"

**Likely cause**: Data leakage in features or target.

**Diagnosis checklist**:
- [ ] Run shuffled target test
- [ ] Check for lag-0 competitor features
- [ ] Verify rolling stats computed post-split
- [ ] Check CV uses TimeSeriesSplit (not random KFold)

**Automated leakage scan**:
```bash
pytest tests/anti_patterns/ -v -m leakage
```

**Solution**: See `knowledge/practices/LEAKAGE_CHECKLIST.md`

---

### "Validation gates fail but model seems reasonable"

**Likely cause**: Feature engineering contamination not caught by automated gates.

**Verification checklist**:
1. [ ] Lag-0 competitor rates excluded? (`grep` for patterns)
2. [ ] Rolling stats computed post-split?
3. [ ] Feature selection on training data only?
4. [ ] Market weights from historical data only?

**Manual audit**:
```python
# Check feature construction order
from src.data.pipelines import get_pipeline_steps
steps = get_pipeline_steps()
print(steps)  # Verify split happens BEFORE feature engineering
```

**Solution**: Review `LEAKAGE_CHECKLIST.md` items 3-5

---

## Test & Environment Issues

### "Notebook fails with 'No fixture found' error"

**Cause**: Missing fixture symlinks.

**Solution**:
```bash
make setup-notebook-fixtures  # Creates symlinks
```

**Verify**:
```bash
ls -la outputs/datasets/  # Should show symlinks
```

See: `docs/onboarding/OFFLINE_DEVELOPMENT.md`

---

### "Tests pass locally but fail in CI"

**Likely causes**:
1. AWS credentials not available in CI
2. Path differences (absolute vs relative)
3. Missing fixtures

**Diagnosis**:
```bash
# Check which tests require AWS
pytest --collect-only -m aws

# Run tests without AWS
pytest -m "not aws"
```

**Solution**: Use fixture-based tests for CI
```bash
make test-notebooks  # Uses fixtures, not AWS
```

---

### "pytest shows 'marker not defined' warning"

**Cause**: Using a marker not registered in `pyproject.toml`.

**Solution**: Add marker to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "your_marker: description of marker",
]
```

**Current markers**: `slow`, `quick`, `aws`, `unit`, `integration`, `baseline`, `notebook`, `property`, `leakage`, `performance`, `memory`, `known_answer`, `monte_carlo`, `adversarial`

---

### "Import error: No module named 'src'"

**Cause**: Python path not configured.

**Solution**:
```bash
# Option 1: Install in editable mode
pip install -e .

# Option 2: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Option 3: Run from project root
cd /path/to/annuity-price-elasticity-v3
python -c "from src.notebooks import create_interface; print('OK')"
```

---

## Data Issues

### "DataFrame has unexpected shape after loading"

**Cause**: Filter parameters or date ranges differ from expected.

**Diagnosis**:
```python
df = load_data()
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Products: {df['product'].unique()}")
```

**Common fixes**:
- Verify product filter matches expected (e.g., "FlexGuard 6Y 20%")
- Check date range includes expected period
- Verify mature data cutoff (default: 50 days)

---

### "Missing columns after feature engineering"

**Cause**: Feature engineering expects specific input columns.

**Diagnosis**:
```python
required_cols = ['prudential_rate', 'competitor_rate', 'date', 'product']
missing = [c for c in required_cols if c not in df.columns]
print(f"Missing: {missing}")
```

**Solution**: Run full pipeline from data extraction
```python
from src.notebooks import create_interface
interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()  # Loads with all required columns
```

---

### "NaN values appearing in features"

**Causes**:
1. Lag features have NaN at start of series
2. Missing competitor data for some dates
3. Division by zero in calculated features

**Diagnosis**:
```python
# Find columns with NaN
nan_cols = df.columns[df.isna().any()]
for col in nan_cols:
    print(f"{col}: {df[col].isna().sum()} NaN values")
```

**Solutions**:
- Drop initial rows (for lag features): `df = df.iloc[10:]`
- Forward fill (for sporadic missing): `df = df.ffill()`
- Check calculation logic for division by zero

---

## Notebook Issues

### "Notebook kernel dies during execution"

**Causes**:
1. Memory exhaustion (large data)
2. Infinite loop
3. Segmentation fault

**Solutions**:
- Reduce data size: `df = df.sample(frac=0.1)`
- Use chunked processing
- Restart kernel and run cells sequentially

**Memory check**:
```python
import psutil
print(f"Memory: {psutil.Process().memory_info().rss / 1e9:.2f} GB")
```

---

### "Notebook output differs from expected"

**Causes**:
1. Random seed not set
2. Data changed since baseline
3. Code updated

**Solution**:
```python
# Set seeds for reproducibility
import numpy as np
import random
np.random.seed(42)
random.seed(42)

# Verify data hasn't changed
print(f"Data hash: {hash(tuple(df.values.tobytes()))}")
```

---

## Git & Version Control

### "Pre-commit hook fails"

**Common failures**:
1. Code formatting (black/ruff)
2. Type errors (mypy)
3. Test failures

**Solutions**:
```bash
# Auto-fix formatting
make lint

# Run tests
make test

# Skip hooks (not recommended)
git commit --no-verify
```

---

### "Merge conflict in notebook"

**Cause**: Notebooks have complex JSON structure.

**Solution**: Use nbstripout to remove outputs before committing
```bash
pip install nbstripout
nbstripout --install  # Installs git filter
```

---

## Performance Issues

### "Model training is very slow"

**Causes**:
1. Large dataset (>100K rows)
2. Too many bootstrap samples
3. Inefficient feature engineering

**Solutions**:
```python
# Reduce bootstrap samples for development
model = BootstrapRidge(n_bootstrap=100)  # Default: 10000

# Use subset of data
df_dev = df.sample(n=10000, random_state=42)

# Profile to find bottleneck
import cProfile
cProfile.run('model.fit(X, y)')
```

---

### "Fixture loading is slow"

**Cause**: Parquet files are large.

**Solutions**:
1. Use session-scoped fixtures (load once per test session)
2. Create smaller fixtures for unit tests
3. Use `@pytest.mark.slow` and skip during development

```bash
# Skip slow tests
pytest -m "not slow"
```

---

## Quick Reference: Common Commands

```bash
# Run all tests
make test

# Run quick smoke test
make quick-check

# Run specific test file
pytest tests/unit/data/test_extraction.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Validate imports
python -c "from src.notebooks import create_interface; print('OK')"

# Check leakage gates
pytest tests/anti_patterns/ -v

# Run notebooks with fixtures
make test-notebooks

# Clean cache
make clean
```

---

## Getting Help

1. **Check documentation**: `docs/INDEX.md` → relevant section
2. **Search episodes**: `docs/knowledge/episodes/` → specific bug pattern
3. **Run diagnostics**: `make quick-check` → identifies common issues
4. **Open issue**: [GitHub Issues](https://github.com/brandon-behring/annuity-price-elasticity/issues)

---

## Related Documentation

- `knowledge/practices/LEAKAGE_CHECKLIST.md` - Pre-deployment validation
- `knowledge/practices/ANTI_PATTERNS.md` - Common mistakes
- `docs/onboarding/OFFLINE_DEVELOPMENT.md` - Development setup
- `docs/knowledge/episodes/` - Bug postmortems (10 episodes)
