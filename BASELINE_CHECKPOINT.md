# Baseline Checkpoint

**Established**: 2026-01-29
**Source**: Archives 1-3 (2026-01-29_215912) + v2 configuration files

## Test Results

| Category | Count |
|----------|-------|
| Passed | 1817 |
| Failed | 46 |
| Skipped | 27 |
| Errors | 23 |

## Known Issues (Not Blocking)

### Pandas 3.0 API Changes (6 tests)
- `fillna(method='ffill')` deprecated → use `ffill()`
- Frequency `'M'` → `'ME'` for month-end

### Numpy Type Comparison (10 tests)
- `np.True_ is True` → `np.True_ == True` or `bool(np.True_)`

### Missing Class References (4 test files excluded)
- `BootstrapInference` class referenced but never implemented
- Affected: test_full_pipeline_offline.py, test_bootstrap_statistical_equivalence.py, etc.

### Missing AWS Baselines (23 errors)
- Integration tests require AWS mode baselines not in archive
- Expected in offline development

## Verification Commands

```bash
cd ~/Claude/annuity-price-elasticity-v3
source venv/bin/activate

# Quick smoke test
python -c "from src.notebooks import create_interface; print('Import OK')"

# Full test suite
pytest tests/ -v --tb=short
```

## Next Steps

1. Add v2 tooling (Makefile, scripts/)
2. Install pre-commit hooks
3. Initialize git
4. Begin incremental refactoring
