# Test Coverage Progress

**Target**: Uniform 60% coverage across core modules.

_Started: 2026-01-29_

---

## Current Status (2026-01-29)

**Overall**: 40%

### Priority Modules (Need Tests)

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| `src/validation/data_schemas.py` | 0% | 60% | P0 |
| `src/validation/config_schemas.py` | 0% | 60% | P0 |
| `src/validation/coefficient_patterns.py` | 11% | 60% | P0 |
| `src/validation/pipeline_validation_helpers.py` | 28% | 60% | P1 |
| `src/data/forecasting_atomic_ops.py` | 23% | 60% | P1 |
| `src/data/extraction.py` | 48% | 60% | P1 |
| `src/data/dvc_manager.py` | 50% | 60% | P2 |
| `src/data/output_manager.py` | 52% | 60% | P2 |
| `src/features/selection/stability/*` | 10-40% | 60% | P2 |

### Modules OK (60%+ or Low Priority)

| Module | Coverage | Notes |
|--------|----------|-------|
| `src/config/*` | 76-100% | Good coverage |
| `src/core/*` | 67-100% | Good coverage |
| `src/data/adapters/*` | 77-96% | Good coverage |
| `src/features/aggregation/*` | 85%+ | Good coverage |
| `src/notebooks/*` | 75%+ | Good coverage |
| `src/products/*` | 100% | Full coverage |
| `src/visualization/*` | 0% | Deprioritized (visual output) |

---

## Session Log

### 2026-01-29 - Initial Assessment
- Overall coverage: 40%
- Created tracking document
- Identified priority modules
- Visualization modules deprioritized (0% is acceptable for visual output)

---

## Next Actions

1. Add unit tests for `src/validation/data_schemas.py`
2. Add unit tests for `src/validation/config_schemas.py`
3. Add unit tests for `src/validation/coefficient_patterns.py`
4. Add integration tests for `src/data/extraction.py`
5. Add unit tests for `src/data/forecasting_atomic_ops.py`

---

## How to Update

```bash
# Generate fresh coverage report
python -m pytest --cov=src --cov-report=term -q --tb=no | grep "^src/" | awk '{print $NF, $1}' | sort -n

# Update this document with new percentages
```
