# Contributing to RILA Price Elasticity Model

Thank you for your interest in contributing to the RILA Price Elasticity Model! This document provides guidelines for contributing to the codebase.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Coding Standards](#coding-standards)
4. [Testing Requirements](#testing-requirements)
5. [Documentation Requirements](#documentation-requirements)
6. [Pull Request Process](#pull-request-process)
7. [Code Review Guidelines](#code-review-guidelines)

---

## Getting Started

### Prerequisites

1. **Onboarding**: Complete [docs/onboarding/GETTING_STARTED.md](docs/onboarding/GETTING_STARTED.md)
2. **Environment Setup**: AWS access configured, SageMaker notebook running
3. **Repository Access**: Clone repository and set up development environment

### First Contribution

**Recommended Path**:
1. Read [docs/onboarding/USER_JOURNEYS.md](docs/onboarding/USER_JOURNEYS.md) (Journey 5: Feature Development)
2. Browse [docs/development/MODULE_HIERARCHY.md](docs/development/MODULE_HIERARCHY.md) to understand code organization
3. Review [docs/development/CODING_STANDARDS.md](docs/development/CODING_STANDARDS.md) for style guidelines
4. Find a "good first issue" or ask team lead for starter tasks

---

## Development Workflow

### Branch Strategy

```bash
# Create feature branch from main (or current development branch)
git checkout -b feature/your-feature-name

# Make changes, commit frequently
git add <files>
git commit -m "Brief description of change"

# Keep branch up to date
git fetch origin
git rebase origin/main

# Push to remote
git push origin feature/your-feature-name
```

**Branch Naming Conventions**:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `refactor/description` - Code refactoring
- `docs/description` - Documentation updates
- `test/description` - Test additions/fixes

### Commit Message Guidelines

**Format**:
```
Brief summary (50 characters or less)

Optional detailed description explaining:
- What changed and why
- Any breaking changes
- Related issues or tickets

Co-Authored-By: Name <email> (if pair programming)
```

**Good Examples**:
```
Add feature selection stability analysis

Implements block bootstrap for feature selection stability testing.
Addresses issue #42.

Fix competitor lag check in leakage checklist

Lag-0 competitor features were not properly detected due to
naming convention mismatch. Updates pattern to catch both
'_lag_0' and '_t0' suffixes.

Refactor data pipeline for multi-product support

Extracts product-specific logic into strategy pattern to support
RILA, FIA, and MYGA product types with shared pipeline code.
```

**Bad Examples**:
```
Update code  (too vague)
🎉 Add new feature  (no emojis in commits)
WIP  (work-in-progress commits should be squashed before PR)
```

---

## Coding Standards

**Complete Guidelines**: [docs/development/CODING_STANDARDS.md](docs/development/CODING_STANDARDS.md)

### Key Standards

1. **Python Style**: Follow PEP 8 with Black formatter
   ```bash
   # Format code before committing
   black src/ tests/
   ```

2. **Type Hints**: Use type hints for all public functions
   ```python
   def calculate_elasticity(
       price: float,
       demand: pd.Series,
       confidence_level: float = 0.95
   ) -> Dict[str, float]:
       """Calculate price elasticity with confidence intervals."""
       ...
   ```

3. **Docstrings**: Use Google-style docstrings
   ```python
   def train_model(X: pd.DataFrame, y: pd.Series) -> BootstrapModel:
       """Train bootstrap ridge regression ensemble.

       Args:
           X: Feature matrix (n_samples, n_features)
           y: Target variable (n_samples,)

       Returns:
           Trained bootstrap ensemble model with 10K estimators

       Raises:
           ValueError: If X and y have different lengths
       """
       ...
   ```

4. **No Emojis**: Do not use emojis in code, commits, or documentation

5. **Error Handling**: Use custom exceptions from `src/core/exceptions.py`
   ```python
   from src.core.exceptions import DataValidationError

   if missing_rate > 0.05:
       raise DataValidationError(
           f"Feature has {missing_rate:.1%} missing data (max: 5%)"
       )
   ```

---

## Testing Requirements

**Complete Guide**: [docs/development/TESTING_GUIDE.md](docs/development/TESTING_GUIDE.md)

### Minimum Requirements

1. **Unit Tests**: All new functions must have unit tests
   ```bash
   # Run unit tests
   pytest tests/unit/ -v
   ```

2. **Integration Tests**: New features need integration tests
   ```bash
   # Run integration tests
   pytest tests/integration/ -v
   ```

3. **Test Coverage**: Maintain > 80% coverage
   ```bash
   # Check coverage
   pytest --cov=src --cov-report=html
   # View: htmlcov/index.html
   ```

4. **Leakage Tests**: Model changes require leakage validation
   ```bash
   # Run leakage gates
   make leakage-audit
   ```

### Test Structure

```python
# tests/unit/features/test_my_feature.py

import pytest
import pandas as pd
from src.features.my_feature import calculate_something

class TestMyFeature:
    """Tests for my_feature module."""

    def test_calculate_something_basic(self):
        """Test basic calculation with known inputs."""
        result = calculate_something(input_data=pd.DataFrame({'x': [1, 2, 3]}))
        expected = pd.Series([2, 4, 6])
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_something_edge_cases(self):
        """Test edge cases (empty input, NaN handling)."""
        # Empty input
        result_empty = calculate_something(input_data=pd.DataFrame())
        assert len(result_empty) == 0

        # NaN handling
        result_nan = calculate_something(input_data=pd.DataFrame({'x': [1, None, 3]}))
        assert not result_nan.isna().any()

    def test_calculate_something_validation(self):
        """Test input validation and error handling."""
        with pytest.raises(ValueError, match="Input must be DataFrame"):
            calculate_something(input_data=[1, 2, 3])  # Wrong type
```

---

## Documentation Requirements

### Code Documentation

1. **Module Docstrings**: Every module needs a docstring
   ```python
   """Feature selection using AIC-based stepwise regression.

   This module implements forward/backward feature selection with
   economic constraint validation to prevent spurious correlations.

   Example:
       >>> from src.features.selection import select_features
       >>> selected = select_features(X, y, max_features=5)
   """
   ```

2. **Class Docstrings**: Document purpose and usage
   ```python
   class BootstrapEnsemble:
       """Bootstrap Ridge regression ensemble for uncertainty quantification.

       Trains multiple Ridge regression models on bootstrap samples to
       estimate prediction uncertainty via confidence intervals.

       Attributes:
           n_estimators: Number of bootstrap samples (default: 10000)
           alpha: Ridge regularization parameter (default: 1.0)

       Example:
           >>> ensemble = BootstrapEnsemble(n_estimators=1000)
           >>> ensemble.fit(X_train, y_train)
           >>> predictions = ensemble.predict(X_test, confidence_level=0.95)
       """
   ```

3. **Function Docstrings**: Use Google style (see above)

### External Documentation

**Update documentation when**:
- Adding new public APIs → Update [docs/api/API_REFERENCE.md](docs/api/API_REFERENCE.md)
- Changing architecture → Update [docs/architecture/](docs/architecture/)
- Adding operations procedures → Update [docs/operations/](docs/operations/)
- Creating common tasks → Update [docs/onboarding/COMMON_TASKS.md](docs/onboarding/COMMON_TASKS.md)

---

## Pull Request Process

### Before Opening PR

1. **Run Full Test Suite**
   ```bash
   # All tests must pass
   pytest tests/ -v

   # Check test coverage
   pytest --cov=src --cov-report=term-missing

   # Run leakage validation (if model changes)
   make leakage-audit
   ```

2. **Format Code**
   ```bash
   # Format with Black
   black src/ tests/

   # Check imports
   isort src/ tests/
   ```

3. **Update Documentation**
   - Update relevant markdown files
   - Add docstrings to new functions
   - Update CHANGELOG.md if significant change

### PR Template

```markdown
## Description
[Brief description of changes]

## Motivation
[Why is this change needed? What problem does it solve?]

## Changes Made
- [ ] Added/modified feature X
- [ ] Updated tests
- [ ] Updated documentation

## Testing
- [ ] All tests pass locally
- [ ] Added new tests for new functionality
- [ ] Test coverage maintained (> 80%)
- [ ] Leakage validation passed (if model changes)

## Documentation
- [ ] Updated API reference (if public API changed)
- [ ] Updated relevant guides
- [ ] Added/updated docstrings

## Checklist
- [ ] Code follows coding standards
- [ ] No emojis in commits or code
- [ ] Branch up to date with main
- [ ] Ready for review

## Related Issues
Closes #XX (if applicable)
```

### PR Review Timeline

- **Initial Review**: Within 2 business days
- **Follow-Up**: Within 1 business day
- **Approval**: Requires 1 approving review from team lead or designated reviewer

---

## Code Review Guidelines

### For Authors

**Responding to Feedback**:
- Address all comments (even if just to acknowledge)
- Ask questions if feedback unclear
- Make requested changes or explain why not
- Mark conversations as resolved after addressing

**Updating PR**:
```bash
# Make changes based on feedback
git add <files>
git commit -m "Address review feedback: <specific change>"
git push origin feature/your-feature-name

# Or amend last commit if minor fix
git add <files>
git commit --amend --no-edit
git push --force-with-lease origin feature/your-feature-name
```

### For Reviewers

**What to Review**:
1. **Correctness**: Does code do what it claims?
2. **Testing**: Are tests adequate and passing?
3. **Style**: Follows coding standards?
4. **Documentation**: Clear docstrings and updated docs?
5. **Maintainability**: Is code readable and maintainable?
6. **Performance**: Any obvious performance issues?
7. **Security**: No data leakage, no credential exposure?

**How to Give Feedback**:
- Be constructive and specific
- Explain the "why" behind suggestions
- Distinguish between required changes and suggestions
- Acknowledge good work

**Example Comments**:
```
# Good
"This function could be simplified using pandas groupby:
[code suggestion]
This would reduce lines and improve readability."

# Better
"Required: Add type hints to this function for consistency with our standards.
Suggestion: Consider extracting this logic into a helper function for reusability."

# Best
"Nice work on the edge case handling! One suggestion: consider adding
a test for when both inputs are empty arrays to ensure graceful handling."
```

---

## Special Cases

### Emergency Hotfixes

For critical production issues:
1. Branch from `main`: `git checkout -b hotfix/critical-issue`
2. Make minimal fix
3. Test thoroughly (including leakage checks)
4. Create PR with "HOTFIX" label
5. Request expedited review
6. After merge, backport fix to development branches

### Dependency Updates

When updating dependencies:
1. Update `requirements.txt` or `environment.yml`
2. Test full suite: `pytest tests/ -v`
3. Check for breaking changes in dependency release notes
4. Update documentation if API changes affect our usage
5. Note dependency version change in commit message

### Documentation-Only Changes

For documentation PRs:
- No code changes required
- Run spell check
- Verify all links work
- Preview rendered markdown
- Smaller review timeline (same day)

---

## Questions?

- **Coding Questions**: Ask in team chat or check [docs/onboarding/TROUBLESHOOTING.md](docs/onboarding/TROUBLESHOOTING.md)
- **Architecture Questions**: Review [docs/architecture/](docs/architecture/) or ask team lead
- **Process Questions**: Ask team lead or check this document

---

## Related Documentation

- [docs/development/CODING_STANDARDS.md](docs/development/CODING_STANDARDS.md) - Complete coding standards
- [docs/development/TESTING_GUIDE.md](docs/development/TESTING_GUIDE.md) - Testing best practices
- [docs/development/MODULE_HIERARCHY.md](docs/development/MODULE_HIERARCHY.md) - Code organization
- [docs/onboarding/USER_JOURNEYS.md](docs/onboarding/USER_JOURNEYS.md) - Learning paths

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
