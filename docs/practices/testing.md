# Test-First Development Pattern

**Source**: Extracted from `~/Claude/causal_inference_mastery` proven patterns
**Last Updated**: 2026-01-20
**Copied from**: ~/Claude/lever_of_archimedes/patterns/testing.md

**Principle**: NO code without tests. Tests created BEFORE or WITH code changes.

---

## 6-Layer Validation Architecture

### Layer 1: Type Safety
- Python: Type hints everywhere (enforced by mypy)
- Catch type errors at lint time, not runtime

### Layer 2: Input Validation
- Check preconditions at function entry
- Fail fast with explicit errors
- Example:
  ```python
  def process_data(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
      if len(df) < min_rows:
          raise ValueError(f"Need {min_rows} rows, got {len(df)}")
      # ... rest of function
  ```

### Layer 3: Unit Tests
- Test each function in isolation
- Happy path + error cases + edge cases
- Target: 80%+ coverage for modules
- Location: `tests/test_[module].py`

### Layer 4: Integration Tests
- Test multi-function workflows
- Verify components work together
- Use realistic data
- Location: `tests/integration/`

### Layer 5: End-to-End Tests
- Test complete user workflows
- From input to output
- Verify system meets requirements
- Location: `tests/e2e/`

### Layer 6: Property-Based Tests
- Test invariants that should ALWAYS hold
- Generate random inputs
- Catch edge cases you didn't think of
- Example: Mathematical equivalence (refactored = original)

---

## Coverage Targets

**Production Code**:
- Modules: 80%+
- Core systems: 90%+
- Scripts: 60%+ (harder to test)

**Test Quality** matters more than coverage percentage:
- Real assertions about correctness
- Not just "doesn't crash"
- Tests fail when bugs introduced

---

## Workflow

```bash
# 1. Check current status
pytest tests/ --cov=src --cov-report=term-missing

# 2. Run quick validation
make quick-check  # Imports + pattern validator

# 3. Run full test suite
make test-all

# 4. Run specific tests
pytest tests/test_features.py -v

# 5. Commit (pre-commit hook verifies tests exist)
git add . && git commit -m "feat: Add feature with tests"
```

---

## Test Structure

**For Modules**:
```python
# tests/test_mymodule.py
import pytest
from src.module import my_function

class TestMyFunction:
    """Tests for my_function."""

    def test_happy_path(self):
        """Test normal operation."""
        result = my_function(valid_input)
        assert result == expected_output

    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(ValueError):
            my_function(invalid_input)

    def test_edge_cases(self):
        """Test boundary conditions."""
        assert my_function(empty_input) == expected_empty_result
        assert my_function(extreme_input) == expected_extreme_result
```

---

## RILA-Specific Testing

### Mathematical Equivalence Tests

When refactoring, ensure mathematical equivalence:

```python
def test_mathematical_equivalence():
    """Refactored code produces identical results."""
    original_result = original_function(data)
    refactored_result = refactored_function(data)

    # 1e-12 precision target
    max_diff = np.abs(original_result - refactored_result).max()
    assert max_diff < 1e-12, f"Max difference: {max_diff}"
```

### Leakage Prevention Tests

```python
def test_no_lag_0_competitors():
    """Competitor features must be lagged."""
    features = create_features(data)
    competitor_cols = [c for c in features.columns if 'competitor' in c.lower()]

    for col in competitor_cols:
        assert 'lag_0' not in col, f"Lag-0 competitor found: {col}"
```

### Coefficient Sign Tests

```python
def test_coefficient_signs():
    """Coefficients have expected economic signs."""
    model = fit_model(data)

    # Own rate should be positive (yield economics)
    assert model.coef_['P_lag_0'] > 0, "Own rate coefficient should be positive"

    # Competitor rate should be negative (substitution)
    assert model.coef_['C_weighted_mean_lag_2'] < 0, "Competitor coefficient should be negative"
```

---

## Exemptions

**Only exempt with `# TEST_EXEMPT: <reason>` in first 30 lines if**:
- Configuration files (YAML, TOML, JSON) with no logic
- Constants only (no functions)
- Generated files
- Temporary experimental code

**Invalid reasons**: "Too hard", "Don't have time", "Code is simple"

---

## Enforcement

- **Pre-commit hook**: Advisory warning (doesn't block, tracks)
- **Pattern validator**: Checks for test coverage
- **CI/CD**: Tests must pass before merge
- **Code review**: Tests required for new functionality

---

**Lesson Learned** (from prior projects):

Created modules without tests → Zero confidence in functionality, difficult to refactor.

Prevention: This protocol ensures test-first development always.

---

**Next**: Use this pattern in ALL projects. Tests are not optional.
