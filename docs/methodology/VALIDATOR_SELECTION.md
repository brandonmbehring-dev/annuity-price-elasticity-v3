# Validator Selection Guide

## Consolidated Validator (Phase 11)

**All mathematical equivalence validation is now consolidated into a single module:**

```python
from src.validation_support.mathematical_equivalence import (
    # Feature Selection Validation
    MathematicalEquivalenceValidator,
    validate_mathematical_equivalence_comprehensive,

    # DataFrame Transformation Validation
    DataFrameEquivalenceValidator,
    validate_pipeline_stage_equivalence,
    enforce_transformation_equivalence,

    # General Validation
    validate_baseline_equivalence,

    # Constants
    TOLERANCE,  # 1e-12
    BOOTSTRAP_STATISTICAL_TOLERANCE,  # 1e-6

    # Exception
    MathematicalEquivalenceError,
)
```

## Usage Examples

### Data Pipeline Refactoring

```python
from src.validation_support.mathematical_equivalence import DataFrameEquivalenceValidator

validator = DataFrameEquivalenceValidator()
result = validator.validate_refactoring_equivalence(baseline_df, refactored_df, "stage_name")
assert result.validation_passed, result.business_impact_assessment
```

### Feature Selection Refactoring

```python
from src.validation_support.mathematical_equivalence import (
    MathematicalEquivalenceValidator
)

validator = MathematicalEquivalenceValidator()
validator.capture_baseline_results(aic_df, constraints_df, bootstrap_list, final_model)
report = validator.run_comprehensive_validation(test_aic, test_constraints, ...)
assert report['overall_validation_passed']
```

### Quick Pipeline Stage Validation

```python
from src.validation_support.mathematical_equivalence import validate_pipeline_stage_equivalence

result = validate_pipeline_stage_equivalence(original_df, refactored_df, "weekly_aggregation")
assert result.validation_passed
```

### Fail-Fast Enforcement

```python
from src.validation_support.mathematical_equivalence import enforce_transformation_equivalence

# Raises MathematicalEquivalenceError if validation fails
validated_df = enforce_transformation_equivalence(original_df, transformed_df, "transform_name")
```

## Validator Capabilities

| Feature | MathematicalEquivalenceValidator | DataFrameEquivalenceValidator |
|---------|----------------------------------|-------------------------------|
| AIC validation | Yes | No |
| Bootstrap stability | Yes | No |
| Economic constraints | Yes | No |
| DataFrame comparison | Basic | Full |
| Column-wise analysis | No | Yes |
| MLflow integration | No | Yes |
| Business impact reporting | Yes | Yes |
| Tolerance configuration | Full | Full |

## Migration Note

Prior modules have been deleted:
- `src/data/mathematical_equivalence_validator.py` → Use `DataFrameEquivalenceValidator`
- `src/features/selection/mathematical_equivalence_validator.py` → Use `MathematicalEquivalenceValidator`

All imports should now use `src.validation_support.mathematical_equivalence`.
