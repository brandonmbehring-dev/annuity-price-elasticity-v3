# Equivalence Test Failure Investigation Guide

## Triage by Difference Magnitude

| Difference | Likely Cause | Action |
|------------|--------------|--------|
| < 1e-14 | Floating-point noise | Usually safe to ignore |
| 1e-14 to 1e-12 | Platform/library version diff | Investigate, may be acceptable |
| 1e-12 to 1e-6 | Algorithmic change | Review code changes carefully |
| 1e-6 to 1e-3 | Significant logic change | Must investigate before proceeding |
| > 1e-3 | Major behavioral change | Stop, root cause required |

## Step-by-Step Investigation

### Step 1: Identify Changed Columns

```python
from src.validation_support.mathematical_equivalence import DataFrameEquivalenceValidator

validator = DataFrameEquivalenceValidator()
result = validator.validate_refactoring_equivalence(baseline, current, "operation_name")

# Get detailed diff report
print(result.business_impact_assessment)
print(result.non_equivalent_columns)
```

### Step 2: Narrow to First Divergence

```python
# Find first row where values differ
for col_info in result.non_equivalent_columns:
    col = col_info['column']
    diff = (baseline[col] - current[col]).abs()
    first_diff_idx = diff[diff > 1e-12].index[0]
    print(f"{col}: First difference at row {first_diff_idx}")
```

### Step 3: Trace Upstream

1. Which transformation produces this column?
2. What are its inputs?
3. Did input data change?
4. Did transformation logic change?

### Step 4: Common Root Causes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All columns differ slightly | Random seed changed | Pin seed in config |
| One column significantly different | Logic change in that feature | Review git diff for that module |
| Shape mismatch (rows) | Filter condition changed | Check date ranges, exclusion criteria |
| Shape mismatch (cols) | Column added/removed | Verify intentional, update baseline |
| NaN pattern changed | Missing data handling | Check fillna, dropna logic |
| Categorical encoding diff | Sort order changed | Use explicit sort before encoding |

## Debugging Commands

```bash
# Compare specific columns
python -c "
import pandas as pd
b = pd.read_parquet('tests/baselines/pre_refactoring/final.parquet')
c = pd.read_parquet('data/output/final.parquet')
diff = (b['P_lag_0'] - c['P_lag_0']).abs()
print(f'Max diff: {diff.max()}')
print(f'Rows with diff > 1e-12: {(diff > 1e-12).sum()}')
"

# Visual diff of specific rows
python scripts/quick_refactoring_check.py --baseline tests/baselines/pre_refactoring/ --rows 0,1,2
```

## Decision Tree: Proceed or Fix?

```
Equivalence test failed
    │
    ├─ Difference < 1e-12?
    │   └─ YES → Acceptable, proceed (platform noise)
    │
    ├─ Change was intentional?
    │   ├─ YES → Update baseline, document reason
    │   └─ NO → Fix the regression
    │
    ├─ Difference affects downstream results?
    │   ├─ YES → Must fix or justify
    │   └─ NO → May be acceptable, document
    │
    └─ Urgency?
        ├─ Blocking release → Quick fix, tech debt ticket
        └─ Can wait → Root cause properly
```

## Documenting Accepted Differences

If a difference is intentional or acceptable, document it:

```python
# In test file
@pytest.mark.xfail(reason="Intentional: improved rounding in v2.1, diff < 1e-10")
def test_legacy_column_equivalence():
    ...
```

Or in baseline metadata:

```json
{
  "accepted_differences": {
    "P_lag_0": {
      "max_diff": 1e-11,
      "reason": "Floating-point optimization in numpy 1.24",
      "date": "2026-01-16",
      "approved_by": "brandon"
    }
  }
}
```
