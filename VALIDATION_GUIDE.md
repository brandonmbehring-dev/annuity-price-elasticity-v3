# Validation Guide - Ensuring Mathematical Equivalence

## Overview

This guide explains how to validate that your refactored code maintains mathematical
equivalence at 1e-12 precision with the original implementation.

**Key Principle**: Throughout refactoring, you must maintain bit-for-bit mathematical equivalence
with the original implementation. Any deviation must be intentional and documented.

## Quick Reference

| Command | Purpose | Duration | When to Use |
|---------|---------|----------|-------------|
| `pytest tests/unit/ -v` | Unit tests | ~2 min | After every change |
| `pytest tests/integration/ -m "not aws" -v` | Integration tests | ~5 min | After module changes |
| `pytest tests/e2e/ -v` | E2E tests | ~3 min | After major changes |
| `python validate_equivalence.py` | Full equivalence | ~10 min | Before committing |
| `python prepare_reintegration.py` | Reintegration check | ~15 min | Before reintegration |

## Validation Levels

### Level 1: Unit Tests (Fast - ~2 minutes)

**Purpose**: Verify individual function/method correctness

**Run**:
```bash
pytest tests/unit/ -v
```

**Expected**: ~1,200 tests pass

**What it validates**:
- Individual function logic
- Edge case handling
- Input validation
- Error handling

**When to run**: After every code change

**Example output**:
```
tests/unit/features/test_feature_engineering.py::test_create_lag_features PASSED
tests/unit/features/test_feature_engineering.py::test_create_rolling_stats PASSED
...
========================= 1200 passed in 120.45s =========================
```

### Level 2: Integration Tests (Medium - ~5 minutes)

**Purpose**: Verify module interactions and pipeline stages

**Run**:
```bash
pytest tests/integration/ -m "not aws" -v
```

**Expected**: ~800 tests pass

**What it validates**:
- Module integration
- Pipeline stage outputs match baselines
- Data flow between components
- Configuration handling

**When to run**: After modifying multiple modules or major refactoring

**Key tests**:
- **Stage-by-stage equivalence**: `test_pipeline_stage_equivalence.py`
- **Bootstrap validation**: `test_bootstrap_statistical_equivalence.py`
- **Cross-product validation**: `test_cross_product_equivalence.py`

**Example output**:
```
tests/integration/test_pipeline_stage_equivalence.py::test_stage_01_product_filtering PASSED
tests/integration/test_pipeline_stage_equivalence.py::test_stage_02_sales_cleanup PASSED
...
========================= 800 passed in 300.12s =========================
```

### Level 3: End-to-End Tests (Slow - ~3 minutes)

**Purpose**: Verify complete pipeline execution

**Run**:
```bash
pytest tests/e2e/ -v
```

**Expected**: ~200 tests pass

**What it validates**:
- Complete pipeline from raw data to predictions
- Multi-product compatibility
- Notebook execution
- Configuration variations

**When to run**: After major architectural changes

**Example output**:
```
tests/e2e/test_full_pipeline_offline.py::test_rila_pipeline_equivalence PASSED
tests/e2e/test_multi_product_pipeline.py::test_fia_pipeline PASSED
tests/e2e/test_multi_product_pipeline.py::test_myga_pipeline PASSED
...
========================= 200 passed in 180.34s =========================
```

### Level 4: Mathematical Equivalence (Critical - ~10 minutes)

**Purpose**: Validate mathematical equivalence at 1e-12 precision

**Run**:
```bash
python validate_equivalence.py
```

**What it checks**:
1. **Stage-by-stage validation** (10 stages at 1e-12 precision)
2. **Bootstrap coefficient distributions** (CV < 5%)
3. **Final model predictions vs baseline** (1e-12 precision)
4. **Economic constraint satisfaction**

**Expected output**:
```
================================================================================
MATHEMATICAL EQUIVALENCE VALIDATION
================================================================================

1. Stage-by-Stage Validation (1e-12 precision)
--------------------------------------------------------------------------------
Running pytest tests/integration/test_pipeline_stage_equivalence.py...
✓ Stage 01: Product Filtering - PASSED
✓ Stage 02: Sales Cleanup - PASSED
✓ Stage 03: Product Cleanup - PASSED
✓ Stage 04: Merge Sales and Products - PASSED
✓ Stage 05: Add Competitive Rates - PASSED
✓ Stage 06: Add Economic Features - PASSED
✓ Stage 07: Feature Engineering - PASSED
✓ Stage 08: Add Market Share - PASSED
✓ Stage 09: Aggregate to Weekly - PASSED
✓ Stage 10: Final Preparation - PASSED

2. Bootstrap Statistical Validation (CV < 5%)
--------------------------------------------------------------------------------
Running pytest tests/integration/test_bootstrap_statistical_equivalence.py...
✓ Coefficient distributions stable
✓ Bootstrap convergence verified

3. End-to-End Pipeline Validation
--------------------------------------------------------------------------------
Running pytest tests/e2e/test_full_pipeline_offline.py...
✓ Full pipeline equivalence maintained

4. Economic Constraint Validation
--------------------------------------------------------------------------------
Running pytest tests/property_based/test_economic_constraints.py...
✓ All economic constraints satisfied

================================================================================
VALIDATION SUMMARY
================================================================================
✓ Stage By Stage: PASSED
✓ Bootstrap Statistical: PASSED
✓ E2E Pipeline: PASSED
✓ Economic Constraints: PASSED

RESULT: Mathematical Equivalence MAINTAINED ✓
Your refactored code is ready for reintegration.

Detailed report saved to: validation_report.json
================================================================================
```

**Generated files**:
- `validation_report.json`: Detailed validation results

### Level 5: Performance Baselines (Optional - ~5 minutes)

**Purpose**: Check for performance regressions

**Run**:
```bash
pytest tests/performance/ -m "not slow" -v
```

**What it validates**:
- Feature engineering < 2s
- Feature selection < 10s
- Bootstrap 1000 iterations < 30s
- No memory leaks
- Reasonable resource usage

**When to run**: Before reintegration

**Example output**:
```
tests/performance/test_feature_engineering_performance.py::test_feature_engineering_performance PASSED
tests/performance/test_bootstrap_performance.py::test_bootstrap_1000_performance PASSED
...
========================= 50 passed in 300.12s =========================
```

## Continuous Validation During Development

### Recommended Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Before Changes: Establish Baseline                           │
│    pytest -v  (all tests should pass)                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. During Development: Fast Feedback Loop                       │
│    • Make small change                                          │
│    • pytest tests/unit/module/ -v  (test that module)           │
│    • Repeat                                                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. After Major Change: Integration Validation                   │
│    pytest tests/integration/ -m "not aws" -v                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Before Commit: Full Equivalence Check                        │
│    python validate_equivalence.py                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Before Reintegration: Final Validation                       │
│    python prepare_reintegration.py                              │
└─────────────────────────────────────────────────────────────────┘
```

### Daily Development Cycle

**Morning** (establish baseline):
```bash
pytest -v  # Should all pass
```

**During development** (after each change):
```bash
pytest tests/unit/path/to/modified/module/ -v  # Fast feedback
```

**End of day** (before committing):
```bash
python validate_equivalence.py  # Verify equivalence maintained
```

### Pre-Commit Checklist

Before committing code:

- [ ] Unit tests pass: `pytest tests/unit/ -v`
- [ ] Integration tests pass: `pytest tests/integration/ -m "not aws" -v`
- [ ] Mathematical equivalence maintained: `python validate_equivalence.py`
- [ ] No obvious performance regression
- [ ] Changelog updated: `CHANGELOG_REFACTORING.md`

## Interpreting Test Results

### Success Indicators

**All tests pass**:
```
========================= 2500 passed in 600.45s =========================
```
✓ You're good to continue

**Some tests skipped**:
```
=================== 2450 passed, 50 skipped in 580.12s ===================
```
✓ This is normal (AWS tests are skipped offline)

### Failure Patterns

#### Unit Test Failures

**Symptom**:
```
FAILED tests/unit/features/test_feature_engineering.py::test_create_lag_features - AssertionError
```

**Cause**: Logic error in refactored code

**Impact**: Low (isolated to specific function)

**Fix**:
1. Run test with verbose output: `pytest tests/unit/features/test_feature_engineering.py::test_create_lag_features -vsx`
2. Review test assertion and expected behavior
3. Debug the specific function
4. Fix the logic error
5. Re-run test

#### Integration Test Failures

**Symptom**:
```
FAILED tests/integration/test_pipeline_stage_equivalence.py::test_stage_05_sales_cleanup
AssertionError: Arrays are not equal at 1e-12 precision
```

**Cause**: Module interaction broken or computation changed

**Impact**: Medium (affects pipeline stage output)

**Fix**:
1. Run specific stage test: `pytest tests/integration/test_pipeline_stage_equivalence.py::test_stage_05_sales_cleanup -vsx`
2. Check what changed in that pipeline stage
3. Compare actual output with baseline:
   ```python
   import pandas as pd
   actual = pd.read_parquet("path/to/actual/output.parquet")
   baseline = pd.read_parquet("tests/baselines/rila/reference/intermediates/05_sales_cleaned.parquet")
   diff = actual.compare(baseline)
   print(diff)
   ```
4. Fix the issue to match baseline exactly

#### Equivalence Test Failures (CRITICAL)

**Symptom**:
```
✗ Stage By Stage: FAILED
✗ Mathematical equivalence BROKEN
```

**Cause**: Mathematical equivalence broken (THIS IS CRITICAL)

**Impact**: HIGH (breaks mathematical guarantee)

**Fix**: This requires careful investigation

1. **Identify which stage failed**:
   ```bash
   pytest tests/integration/test_pipeline_stage_equivalence.py -v
   ```

2. **Run failing stage in isolation**:
   ```bash
   pytest tests/integration/test_pipeline_stage_equivalence.py::test_stage_XX -vsx
   ```

3. **Compare outputs in detail**:
   ```python
   import pandas as pd
   import numpy as np

   actual = pd.read_parquet("actual_output.parquet")
   expected = pd.read_parquet("tests/baselines/rila/reference/intermediates/XX_stage.parquet")

   # Check column differences
   print("Columns in actual:", set(actual.columns))
   print("Columns in expected:", set(expected.columns))

   # Check row count
   print(f"Rows actual: {len(actual)}, expected: {len(expected)}")

   # Find numeric differences
   for col in actual.select_dtypes(include=[np.number]).columns:
       if col in expected.columns:
           diff = np.abs(actual[col] - expected[col]).max()
           print(f"{col}: max diff = {diff}")
   ```

4. **Common causes** (see next section)

### Common Equivalence Issues

#### 1. Floating-Point Precision

**Symptom**: Tests fail at 1e-12 but pass at 1e-6

**Cause**: Different computation order causing minor floating-point differences

**Example**:
```python
# Original (baseline)
result = (a + b) + c

# Refactored (slightly different)
result = a + (b + c)  # May differ at 1e-12 due to floating-point arithmetic
```

**Fix options**:
1. **Best**: Match exact computation order from original
2. **Acceptable**: Use consistent computation order (update baseline if needed)
3. **Last resort**: Relax tolerance to 1e-10 (ONLY if justified)

**How to decide**: If the difference is purely floating-point and doesn't affect results meaningfully,
you may relax tolerance slightly. Document this in CHANGELOG_REFACTORING.md.

#### 2. Random Seeds

**Symptom**: Bootstrap tests fail, results vary between runs

**Cause**: Missing or different `random_state` parameter

**Example**:
```python
# Original
bootstrap_model = BootstrapModel(random_state=42)

# Refactored (WRONG - missing random_state)
bootstrap_model = BootstrapModel()  # Will fail equivalence
```

**Fix**: Ensure all stochastic operations set consistent random seeds:
```python
# Feature engineering
np.random.seed(42)

# Model training
model = RidgeRegression(random_state=42)

# Bootstrap
bootstrap = Bootstrap(random_state=42, n_iterations=1000)
```

#### 3. DataFrame Ordering

**Symptom**: Tests fail with "order differs" even though values are same

**Cause**: Unstable sort without tie-breaker columns

**Example**:
```python
# Original
df = df.sort_values("date")  # Multiple rows per date

# Refactored (same result, different order)
df = df.sort_values("date")  # Order within same date is undefined
```

**Fix**: Add tie-breaker columns to ensure stable sorting:
```python
df = df.sort_values(["date", "product_id", "index"])  # Stable sort
```

Or reset index after sorting:
```python
df = df.sort_values("date").reset_index(drop=True)
```

#### 4. Feature Engineering Changes

**Symptom**: Stage 7 (feature engineering) fails

**Cause**: Feature creation logic differs from original

**Common issues**:
- Different lag window sizes
- Different rolling window calculations
- Missing feature transformations
- Different feature naming

**Fix**: Review feature engineering code line-by-line against original:
```bash
# Compare your code with original
diff src/features/feature_engineering.py original/src/features/feature_engineering.py
```

Ensure:
- Same lag features created
- Same rolling window parameters
- Same transformations applied
- Same feature names used

#### 5. Missing Data Handling

**Symptom**: Row counts differ or NA values handled differently

**Cause**: Different missing data handling strategy

**Example**:
```python
# Original
df = df.dropna(subset=["sales"])

# Refactored (WRONG - drops more rows)
df = df.dropna()  # Drops rows with ANY missing value
```

**Fix**: Match exact missing data handling:
```python
df = df.dropna(subset=["sales"])  # Drop only if sales is missing
```

#### 6. Data Type Changes

**Symptom**: Tests fail with "dtype mismatch"

**Cause**: Changed column data types

**Example**:
```python
# Original
df["product_id"] = df["product_id"].astype(str)

# Refactored (WRONG)
df["product_id"] = df["product_id"].astype(int)  # Type mismatch
```

**Fix**: Ensure all column types match original:
```python
# Check types in baseline
baseline = pd.read_parquet("tests/baselines/rila/reference/intermediates/XX_stage.parquet")
print(baseline.dtypes)

# Match them exactly
df["product_id"] = df["product_id"].astype(str)
```

## Validation Thresholds

| Test Type | Precision | Acceptable? | Notes |
|-----------|-----------|-------------|-------|
| Pipeline stage outputs | 1e-12 | **Required** | Critical for equivalence |
| AIC/BIC calculations | 1e-12 | **Required** | Model selection criteria |
| Bootstrap coefficients | 1e-12 | **Required** | Statistical inference |
| Feature engineering | 1e-6 | Acceptable | Some tolerance OK |
| Unit test assertions | 1e-6 | Acceptable | General calculations |
| Row counts | Exact | **Required** | Must match exactly |
| Column names | Exact | **Required** | Must match exactly |
| Data types | Exact | **Required** | Must match exactly |

**Golden Rule**: When in doubt, match exactly. Only relax tolerance if you understand why and document it.

## Validation Report

After running `python validate_equivalence.py`, check `validation_report.json`:

```json
{
  "timestamp": "2026-01-29T20:00:00",
  "checks": {
    "stage_by_stage": "PASSED",
    "bootstrap_statistical": "PASSED",
    "e2e_pipeline": "PASSED",
    "economic_constraints": "PASSED"
  },
  "ready_for_reintegration": true
}
```

**Key fields**:
- `ready_for_reintegration`: Must be `true` before reintegration
- `checks`: All must be "PASSED"

If `ready_for_reintegration` is `false`, review which checks failed and fix them.

## Debugging Workflow

### When Tests Fail

1. **Isolate the failure**:
   ```bash
   pytest tests/path/to/test.py::test_specific_test -vsx
   ```

2. **Show full traceback**:
   ```bash
   pytest tests/path/to/test.py::test_specific_test -vsx --tb=long
   ```

3. **Drop into debugger**:
   ```bash
   pytest tests/path/to/test.py::test_specific_test --pdb
   ```

4. **Compare with baseline**:
   ```python
   import pandas as pd
   actual = pd.read_parquet("actual_output.parquet")
   baseline = pd.read_parquet("tests/baselines/path/to/baseline.parquet")

   # Show differences
   print(actual.compare(baseline))

   # Check specific columns
   print((actual["column"] - baseline["column"]).describe())
   ```

5. **Review your changes**:
   - What did you modify?
   - Did you change any algorithms?
   - Did you change any parameters?
   - Did you change data types or column names?

6. **Fix and re-test**:
   ```bash
   pytest tests/path/to/test.py::test_specific_test -v
   ```

### When Equivalence is Broken

If mathematical equivalence is broken, work backward from the failure:

1. **Identify failing stage**:
   ```bash
   pytest tests/integration/test_pipeline_stage_equivalence.py -v
   ```
   Suppose Stage 7 fails.

2. **Test previous stage**:
   ```bash
   pytest tests/integration/test_pipeline_stage_equivalence.py::test_stage_06 -v
   ```
   If Stage 6 passes, the issue is isolated to Stage 7.

3. **Review Stage 7 code**:
   - What does Stage 7 do? (Feature engineering)
   - What files are involved? (`src/features/feature_engineering.py`)
   - What did you change in those files?

4. **Compare outputs column by column**:
   ```python
   import pandas as pd
   import numpy as np

   actual = pd.read_parquet("stage_07_output.parquet")
   baseline = pd.read_parquet("tests/baselines/rila/reference/intermediates/07_features_created.parquet")

   # Check each numeric column
   for col in actual.select_dtypes(include=[np.number]).columns:
       if col in baseline.columns:
           max_diff = np.abs(actual[col] - baseline[col]).max()
           if max_diff > 1e-12:
               print(f"MISMATCH: {col} - max diff: {max_diff}")
   ```

5. **Fix the specific issue**

6. **Re-validate full equivalence**:
   ```bash
   python validate_equivalence.py
   ```

## Troubleshooting

### "Fixture not found" error

**Cause**: Package extraction incomplete

**Fix**:
```bash
python validate_package.py  # Check package integrity
```

### "Baseline mismatch" error

**Cause**: Mathematical equivalence broken

**Fix**: Run individual stage tests to pinpoint:
```bash
pytest tests/integration/test_pipeline_stage_equivalence.py -v
```

### "Import error" when running tests

**Cause**: Dependencies not installed or Python path issue

**Fix**:
```bash
# Reinstall dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Tests pass locally but fail in validation script

**Cause**: Environment differences or caching

**Fix**:
```bash
# Clear pytest cache
pytest --cache-clear

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Re-run validation
python validate_equivalence.py
```

## Success Criteria

Your refactoring is ready for reintegration when:

✓ **All unit tests pass** (100% - ~1,200 tests)
✓ **All integration tests pass** (100% - ~800 tests)
✓ **All E2E tests pass** (100% - ~200 tests)
✓ **Mathematical equivalence maintained** (1e-12 precision)
✓ **Performance baselines met** (no significant regressions)
✓ **No new warnings or errors**
✓ **Documentation updated**
✓ **Changelog complete**

Run `python prepare_reintegration.py` to verify all criteria.

## Final Checklist

Before reintegration:

- [ ] All tests pass: `pytest -v`
- [ ] Equivalence validated: `python validate_equivalence.py`
- [ ] Performance OK: `pytest tests/performance/ -m "not slow" -v`
- [ ] Changelog updated: `CHANGELOG_REFACTORING.md`
- [ ] No temporary debug code left in source
- [ ] No new dependencies (or documented if necessary)
- [ ] Documentation reflects changes
- [ ] Validation report generated: `python prepare_reintegration.py`

---

**Ready to validate? Run `python validate_equivalence.py`!**
