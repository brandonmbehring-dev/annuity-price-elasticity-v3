# Test Baselines Directory

This directory contains baseline artifacts for regression testing and mathematical equivalence validation.

## Directory Structure

```
tests/baselines/
├── rila/                      # RILA product baselines
│   ├── reference/             # Current reference artifacts
│   ├── pre_refactoring_reference/  # Historical pre-v2 artifacts
│   └── notebooks/             # Notebook-specific baselines
├── fia/                       # FIA product baselines
│   ├── reference/             # Current reference artifacts
│   └── notebooks/             # Notebook-specific baselines
├── pre_refactoring_reference/ # Legacy v1 artifacts (deprecated)
└── notebooks/                 # Shared notebook baselines
```

## Purpose

1. **Mathematical Equivalence Testing**: Ensures refactoring maintains numerical precision (1e-12)
2. **Regression Prevention**: Catches unintended behavioral changes
3. **Historical Reference**: Preserves pre-v2 artifacts for comparison

## Artifact Types

| Directory | Purpose | Status |
|-----------|---------|--------|
| `artifacts/coefficients/` | Model coefficient baselines | Placeholder - populate when capturing |
| `artifacts/models/` | Serialized model objects | Placeholder - populate when capturing |
| `artifacts/features/` | Feature engineering outputs | Placeholder - populate when capturing |

## Empty Directories

Empty subdirectories are intentional placeholders for future baseline capture.
The structure is pre-created to ensure consistent paths when baselines are generated.

To capture new baselines:
```python
from tests.baselines.capture import capture_baseline
capture_baseline(results, product='rila', stage='inference')
```

## Related Documentation

- `knowledge/practices/testing.md` - 6-layer validation architecture
- `tests/test_intermediate_stage_equivalence.py` - Stage-by-stage tests
- `tests/test_offline_aws_equivalence.py` - AWS data equivalence tests
