# Domain Decision Examples

**Last Updated:** 2026-01-30
**Purpose:** Worked examples of AI-assisted domain modeling decisions

---

## Overview

This document provides detailed examples of domain-specific decisions made during AI-assisted development. These decisions require understanding of:
- Annuity product economics
- Causal inference principles
- Time series data handling
- Regulatory constraints

---

## Decision 1: Lag-0 Competitor Prohibition

### Context

**Problem**: Initial models included lag-0 (contemporaneous) competitor rate features, producing suspiciously high R² values.

**Constraints**:
- Causal identification requires temporal ordering
- Competitor rates published with delay
- Decision-makers cannot observe "future" competitor rates
- Model must reflect real business process

### The Trap

```python
# DANGEROUS: This "works" but is causally invalid
features["competitor_rate_t0"] = competitor_rates["rate"]
# R² = 0.95  <- Too good to be true!
```

### Analysis

**Why it's wrong**:

1. **Temporal ordering violation**: Can't use information not available at decision time
2. **Reverse causality**: Maybe our rates influence theirs, not vice versa
3. **Spurious correlation**: Both respond to same market conditions
4. **Business reality**: Competitor rates have publication lag (1-4 weeks)

**The tell-tale signs**:
- R² > 0.90 (suspiciously high for behavioral data)
- Model fails shuffled target test
- Coefficients don't match economic theory

### Decision

**Minimum 2-week lag for all competitor features**

### Rationale

1. **Causal identification**: 2 weeks ensures temporal separation
2. **Business reality**: Reflects actual information availability
3. **Publication lag**: Aligns with WINK data release schedule
4. **Conservative**: Better to underfit than overfit on leakage

### Implementation

```python
# src/features/engineering/timeseries.py
def create_competitor_features(
    df: pd.DataFrame,
    min_lag: int = 2  # Minimum 2 weeks
) -> pd.DataFrame:
    """Create lagged competitor features.

    Args:
        df: DataFrame with competitor rate columns
        min_lag: Minimum lag in weeks (default 2, enforced)

    Returns:
        DataFrame with lagged features only
    """
    if min_lag < 2:
        raise ConstraintViolationError(
            message=f"min_lag={min_lag} violates causal identification",
            business_impact="Model would use future information",
            required_action="Use min_lag >= 2"
        )

    competitor_cols = [c for c in df.columns if "competitor" in c.lower()]
    result = df.copy()

    for col in competitor_cols:
        for lag in range(min_lag, 18):  # t-2 through t-17
            result[f"{col}_t{lag}"] = df[col].shift(lag)

    # Explicitly do NOT create t0 or t1
    return result

# Validation in tests
def test_no_lag0_competitor_features(features_df):
    """Ensure no lag-0 competitor features exist."""
    competitor_cols = [c for c in features_df.columns if "competitor" in c.lower()]
    lag0_cols = [c for c in competitor_cols if "_t0" in c or c.endswith("_current")]

    assert len(lag0_cols) == 0, f"Found forbidden lag-0 features: {lag0_cols}"
```

### Enforcement

```python
# tests/anti_patterns/test_lag0_detection.py
@pytest.mark.leakage
def test_pipeline_excludes_lag0_competitors():
    """Full pipeline must not create lag-0 competitor features."""
    df = load_test_data()
    features = run_full_pipeline(df)

    # Check column names
    for col in features.columns:
        if "competitor" in col.lower():
            assert "_t0" not in col, f"Lag-0 detected: {col}"
            assert not col.endswith("_current"), f"Current detected: {col}"
```

---

## Decision 2: Coefficient Sign Constraints

### Context

**Problem**: Models occasionally produce economically invalid coefficients (e.g., negative own-rate elasticity).

**Constraints**:
- Economic theory prescribes coefficient signs
- Signs should match yield economics logic
- Invalid signs indicate data or specification issues
- Need to catch before deployment

### Economic Theory

| Coefficient | Expected Sign | Economic Rationale |
|-------------|---------------|-------------------|
| Own cap rate | **Positive** | Higher rates attract customers |
| Competitor rates | **Negative** | Substitution effect |
| Buffer level | **Positive** | More protection is better |
| Volatility | **Negative** | Uncertainty reduces purchases |

### Why Signs Matter

**1. Yield Economics Logic**:
- Cap rate IS the yield for these products
- Higher yield = more attractive = positive coefficient
- This differs from bond pricing (inverse relationship)

**2. Substitution Effect**:
- Higher competitor rates pull customers away
- Negative coefficient represents market share loss

**3. Model Validity**:
- Wrong signs suggest specification error
- Could indicate data leakage or multicollinearity
- Requires investigation before deployment

### Decision

**Enforce sign constraints with configurable strictness**

### Implementation

```python
# src/validation/coefficient_validator.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class CoefficientConstraint:
    """Defines expected sign for a coefficient."""
    name: str
    expected_sign: Literal["positive", "negative", "any"]
    tolerance: float = 0.0  # Allow small violations

RILA_CONSTRAINTS = [
    CoefficientConstraint("prudential_rate", "positive"),
    CoefficientConstraint("competitor_weighted", "negative"),
    CoefficientConstraint("buffer_level", "positive"),
    CoefficientConstraint("vix", "negative"),
]

def validate_coefficient_signs(
    coefficients: dict[str, float],
    constraints: list[CoefficientConstraint],
    strict: bool = True
) -> ValidationResult:
    """Validate coefficient signs against economic theory.

    Args:
        coefficients: Fitted model coefficients
        constraints: Expected sign constraints
        strict: If True, violations raise errors; if False, warnings only

    Returns:
        ValidationResult with any violations
    """
    violations = []
    warnings = []

    for constraint in constraints:
        # Find matching coefficient (partial match)
        matches = [k for k in coefficients if constraint.name in k.lower()]

        for coef_name in matches:
            value = coefficients[coef_name]

            if constraint.expected_sign == "positive" and value < -constraint.tolerance:
                msg = f"{coef_name}={value:.4f} should be positive"
                if strict:
                    violations.append(msg)
                else:
                    warnings.append(msg)

            elif constraint.expected_sign == "negative" and value > constraint.tolerance:
                msg = f"{coef_name}={value:.4f} should be negative"
                if strict:
                    violations.append(msg)
                else:
                    warnings.append(msg)

    return ValidationResult(
        is_valid=len(violations) == 0,
        errors=tuple(violations),
        warnings=tuple(warnings)
    )
```

### Production Gate

```python
# In inference pipeline
def run_inference(df: pd.DataFrame, strict_signs: bool = True) -> InferenceResult:
    """Run inference with sign validation."""
    model = fit_model(df)

    # Validate signs
    sign_result = validate_coefficient_signs(
        model.coefficients,
        RILA_CONSTRAINTS,
        strict=strict_signs
    )

    if not sign_result.is_valid:
        raise ConstraintViolationError(
            message="Coefficient signs violate economic theory",
            business_impact="Model predictions may be directionally wrong",
            required_action=f"Investigate: {sign_result.errors}"
        )

    return InferenceResult(model=model, validation=sign_result)
```

---

## Decision 3: Feature Naming Conventions

### Context

**Problem**: Inconsistent feature naming made code maintenance difficult and created confusion about temporal alignment.

**Original names**:
- `prudential_rate_current` (what does "current" mean?)
- `competitor_mid` (mid of what?)
- `rate_week_1` (1 week ago? Week 1 of year?)

### Options Considered

| Convention | Example | Pros | Cons |
|------------|---------|------|------|
| **Descriptive** | `prudential_rate_current_week` | Clear | Verbose |
| **Positional** | `rate_t0`, `rate_t1` | Compact, standard | Needs documentation |
| **Mixed** | `prudential_rate_t0` | Balanced | Requires discipline |

### Decision

**Positional lag notation** with semantic prefixes:

- `{source}_{metric}_t{lag}` for lagged features
- `{source}_{metric}_weighted` for aggregated competitor features

### Rationale

1. **Mathematical clarity**: `t0` = current, `t1` = 1 period ago
2. **Loop-friendly**: `for lag in range(0, 18): f"rate_t{lag}"`
3. **Industry standard**: Common in time series econometrics
4. **Grep-friendly**: Easy to find all lag-2 features with `_t2`

### Implementation

```python
# src/features/naming.py
from typing import Literal

def create_feature_name(
    source: str,
    metric: str,
    lag: int | None = None,
    aggregation: Literal["weighted", "top5", "mean"] | None = None
) -> str:
    """Create standardized feature name.

    Args:
        source: Data source (e.g., "prudential", "competitor")
        metric: Metric name (e.g., "rate", "buffer")
        lag: Time lag (0 = current, 1 = 1 period ago)
        aggregation: Aggregation method for competitor features

    Returns:
        Standardized feature name

    Examples:
        >>> create_feature_name("prudential", "rate", lag=0)
        'prudential_rate_t0'
        >>> create_feature_name("competitor", "rate", aggregation="weighted")
        'competitor_rate_weighted'
    """
    base = f"{source}_{metric}"

    if lag is not None:
        return f"{base}_t{lag}"
    elif aggregation is not None:
        return f"{base}_{aggregation}"
    else:
        return base

# Migration mapping for backward compatibility
LEGACY_TO_NEW = {
    "prudential_rate_current": "prudential_rate_t0",
    "competitor_mid": "competitor_rate_weighted",
    "rate_week_1": "rate_t1",
}

def migrate_feature_name(legacy_name: str) -> str:
    """Convert legacy feature name to new convention."""
    return LEGACY_TO_NEW.get(legacy_name, legacy_name)
```

### Validation

```python
# tests/unit/features/test_naming.py
import re

def test_feature_names_follow_convention(feature_df):
    """All feature names must follow naming convention."""
    pattern = re.compile(
        r"^[a-z]+_[a-z]+(_t\d+|_weighted|_top\d+|_mean)?$"
    )

    invalid = [col for col in feature_df.columns if not pattern.match(col)]

    assert len(invalid) == 0, f"Invalid feature names: {invalid}"
```

---

## Decision 4: Causal Framework Adoption

### Context

**Problem**: Initial models used predictive ML approach (maximize R²), leading to:
- Overfit models that didn't generalize
- Features with no causal interpretation
- Inability to answer "what if" pricing questions

### The Fundamental Question

> "If Prudential increases cap rates by 25bp, how much will sales increase?"

This is a **causal question**, not a prediction question.

### Options Considered

| Approach | Can Answer Causal Question? | Interpretability |
|----------|----------------------------|------------------|
| **XGBoost** | No | Low |
| **Neural Net** | No | Very Low |
| **Linear Regression** | Yes (with proper specification) | High |
| **Diff-in-Diff** | Yes | High |
| **RDD** | Yes (at discontinuity) | High |

### Decision

**Linear regression with causal specification**:
- Log-linear elasticity model
- Proper temporal ordering (no lag-0)
- Lagged dependent variable for autocorrelation
- Economic sign constraints

### Rationale

1. **Interpretability**: Coefficients are elasticities
2. **Causal identification**: Can isolate rate effect with proper controls
3. **Business utility**: Directly answers pricing questions
4. **Validation**: Can test against economic theory

### Model Specification

```python
# The causal model
#
# ln(sales_t) = β₀ + β₁·ln(own_rate_t) + β₂·ln(competitor_rate_{t-2})
#             + β₃·ln(sales_{t-1}) + γ·controls + ε_t
#
# Where:
#   β₁ = own-rate elasticity (expected positive)
#   β₂ = cross-rate elasticity (expected negative)
#   β₃ = persistence (autocorrelation control)

def specify_causal_model(df: pd.DataFrame) -> FormulaSpec:
    """Specify causal elasticity model."""
    return FormulaSpec(
        dependent="log_sales",
        own_rate="log_prudential_rate_t0",  # Current own rate is OK
        competitor_rate="log_competitor_weighted_t2",  # Lagged!
        controls=[
            "log_sales_t1",  # Autocorrelation control
            "log_vix_t0",    # Market conditions
            "quarter_dummies",
        ],
        expected_signs={
            "log_prudential_rate_t0": "positive",
            "log_competitor_weighted_t2": "negative",
        }
    )
```

### Identification Strategy

**Key insight**: Competitor rates are "exogenous" to Prudential's sales because:
1. Competitors set rates independently
2. Market share too small for reverse causality
3. 2-week lag ensures temporal separation

This allows causal interpretation of β₂, which combined with market structure assumptions, allows causal interpretation of β₁.

---

## Decision 5: Leakage Prevention Checklist

### Context

**Problem**: Multiple instances of data leakage discovered during development, each requiring significant rework.

**Past leakage incidents**:
1. Lag-0 competitor rates (R² jumped to 0.95)
2. Future sales in rolling averages
3. Test data in training set
4. Post-hoc feature engineering

### Decision

**Mandatory pre-deployment checklist** with automated gates

### Implementation

```markdown
# knowledge/practices/LEAKAGE_CHECKLIST.md

## Pre-Deployment Leakage Gate

### Automated Checks (make leakage-audit)
- [ ] No lag-0 competitor features
- [ ] Shuffled target test FAILS (model shouldn't work on random targets)
- [ ] Temporal train/test split (no future data in training)
- [ ] Feature timestamp validation (all features use past data only)

### Manual Checks
- [ ] Coefficient signs match economic theory
- [ ] R² is plausible (< 0.90 for behavioral data)
- [ ] Feature importance explainable
- [ ] Business stakeholder review of top features

### Sign-Off Required
- [ ] Data scientist: _______________
- [ ] Reviewer: _______________
- [ ] Date: _______________
```

### Automated Gate

```python
# tests/leakage/test_leakage_gates.py

@pytest.mark.leakage
class TestLeakageGates:
    """Mandatory leakage prevention tests."""

    def test_shuffled_target_fails(self, fitted_model, test_data):
        """Model should fail on shuffled targets."""
        shuffled = test_data.copy()
        shuffled["sales"] = np.random.permutation(shuffled["sales"])

        shuffled_r2 = fitted_model.score(shuffled)

        # If model works on random data, something is wrong
        assert shuffled_r2 < 0.1, (
            f"Model achieved R²={shuffled_r2:.2f} on shuffled data. "
            "This indicates data leakage."
        )

    def test_no_future_in_features(self, feature_df):
        """Features must only use past data."""
        for col in feature_df.columns:
            if "_t" in col:
                lag = int(col.split("_t")[-1])
                assert lag >= 0, f"Future data in {col}"

            if "competitor" in col.lower():
                if "_t" in col:
                    lag = int(col.split("_t")[-1])
                    assert lag >= 2, f"Insufficient lag in {col}"

    def test_temporal_train_test_split(self, train_df, test_df):
        """Test data must be strictly after training data."""
        train_max = train_df["date"].max()
        test_min = test_df["date"].min()

        assert test_min > train_max, (
            f"Test data ({test_min}) overlaps with training ({train_max})"
        )
```

---

## Summary Table

| Decision | Domain Concept | Key Constraint |
|----------|---------------|----------------|
| Lag-0 Prohibition | Causal identification | min_lag >= 2 |
| Coefficient Signs | Yield economics | Own rate positive |
| Feature Naming | Time series convention | `_t{lag}` notation |
| Causal Framework | Econometric identification | Proper specification |
| Leakage Prevention | Model validity | Automated gates |

---

## Related Documentation

- [../AI_COLLABORATION.md](../AI_COLLABORATION.md) - Overview
- [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) - Architecture patterns
- [../../domain-knowledge/RILA_ECONOMICS.md](../../domain-knowledge/RILA_ECONOMICS.md) - Product economics
- [../../practices/LEAKAGE_CHECKLIST.md](../../practices/LEAKAGE_CHECKLIST.md) - Pre-deployment gate

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-30 | Initial creation | Claude |
