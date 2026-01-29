# Mathematical Equivalence Workflow

## When to Use

- Any refactoring that touches data transformations
- Config changes that affect pipeline outputs
- Feature engineering modifications

## Workflow Steps

### Step 1: Capture Baseline (BEFORE changes)

```bash
# Choose based on refactoring type:
python scripts/capture_refactoring_baseline.py   # General refactoring
python scripts/capture_aws_baseline.py           # AWS data changes
python scripts/regenerate_ci_baseline.py         # CI baseline updates
```

### Step 2: Write Equivalence Test FIRST (TDD)

```python
def test_refactoring_equivalence():
    baseline = load_baseline("tests/baselines/pre_refactoring/")
    result = run_pipeline_with_changes()
    assert_frame_equal(result, baseline, rtol=1e-12)
```

### Step 3: Implement Changes

- Make refactoring changes
- Run equivalence test after each significant change
- If test fails: investigate before proceeding (see [FAILURE_INVESTIGATION.md](FAILURE_INVESTIGATION.md))

### Step 4: Validate at 1e-12 Precision

```bash
pytest tests/integration/test_config_equivalence.py -v
```

### Step 5: Lifecycle Decision

| Decision | Criteria |
|----------|----------|
| **Keep tests** | Ongoing refactoring, high-risk module, active development |
| **Archive tests** | Validation passed, stable pipeline, one-time migration |

Archive location: `tests/archived/YYYY_module_refactor/`

## Quick Reference

| Phase | Command |
|-------|---------|
| Capture (refactoring) | `python scripts/capture_refactoring_baseline.py` |
| Capture (AWS) | `python scripts/capture_aws_baseline.py` |
| Validate | `pytest tests/integration/ -k equivalence` |
| Full suite | `pytest tests/ -v` |

## CI/CD Integration (Tiered)

| Branch | Equivalence Tests | Behavior |
|--------|-------------------|----------|
| `main` | **Mandatory** | PR blocked if tests fail |
| `feature/*` | Advisory | Tests run, report only |
| `experiment/*` | Skip | No equivalence enforcement |

## Related Documents

- [TOLERANCE_REFERENCE.md](TOLERANCE_REFERENCE.md) - When to use which tolerance
- [FAILURE_INVESTIGATION.md](FAILURE_INVESTIGATION.md) - Debugging failed tests
- [VALIDATOR_SELECTION.md](VALIDATOR_SELECTION.md) - Which validator for which scenario
