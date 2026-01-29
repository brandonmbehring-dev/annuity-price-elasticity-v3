# Unified Coding & Notebook Standards for Multi-Product Annuity Price Elasticity Analysis (v2)

**Document Version**: 2.0 (Multi-Product Architecture)
**Last Updated**: 2025-01-24
**Status**: Authoritative - Single Source of Truth
**Based On**: RILA Price Elasticity template (3.0) adapted for v2 architecture

## Executive Summary

This document consolidates and unifies all coding standards for the Multi-Product Annuity Price Elasticity Analysis v2 project. It establishes clear guidelines for all development work and provides a single authoritative reference for coding practices across RILA, FIA, and MYGA products.

**Key Features:**
- Unified function design standards with evidence-based metrics
- Mathematical equivalence validation for refactoring (mandatory)
- Fail-fast error handling requirements (critical priority)
- Multi-product architecture patterns with dependency injection
- Comprehensive documentation and testing frameworks
- v2 architecture entry points for adapter, aggregation, and strategy patterns

## Table of Contents

1. [Evidence-Based Standards Specification](#evidence-based-standards)
2. [Core Architecture Principles](#core-principles)
3. [Function Design Standards](#function-design)
4. [Mathematical Equivalence Testing](#mathematical-equivalence)
5. [Configuration Management](#configuration)
6. [Notebook Writing Standards](#notebook-standards)
7. [Testing & Validation Framework](#testing)
   - 7.4 [Mandatory Unit Testing Requirements](#mandatory-unit-testing-requirements)
8. [Visual and Communication Standards](#visual-standards)
9. [Implementation Checklist](#implementation-checklist)

---

## 1. Evidence-Based Standards Specification {#evidence-based-standards}

**Standards based on comprehensive codebase audit. Current compliance status with improvement roadmap:**

### 1.1 Function Design Standards

**Length Limits (Strict Enforcement):**
- **Simple transforms**: 10-30 lines maximum
- **Atomic operations**: 30-50 lines maximum (NO multi-step operations - maintain single responsibility)
- **Exceptions allowed ONLY for**:
  - Configuration builders (like config_builder.py functions)
  - Complex business logic with mathematical formulas (when breaking apart would hurt clarity)

**Current Status**: V2 refactoring phase - 92/100 compliance audit (improved from baseline)
**Target**: Maintain high compliance during multi-product feature development
**Architecture Pattern**: Dependency injection throughout data adapters, strategies, and methodologies

**Type Hints**: Mandatory for ALL functions - complete parameter, return value, and TypedDict configuration typing

**Error Handling**: Always comprehensive with business context - fail-fast patterns required

### 1.2 Fail-Fast Requirements (MANDATORY)

**Status**: MANDATORY (immediate implementation required)
**Priority**: CRITICAL (addresses error visibility and work preservation)
**Rationale**: Maximum error visibility essential for debugging and development tracking

**ABSOLUTE PROHIBITIONS** (Never implement these patterns):
```python
# PROHIBITED: Synthetic data fallbacks masking real issues
try:
    adapter = get_adapter("6Y20B", config)
except Exception as e:
    logger.warning(f"Using fixture data: {e}")  # NEVER DO THIS
    return create_synthetic_data()

# PROHIBITED: Silent failures with warning logs only
try:
    validate_product_constraints(df, product_code)
except ValidationError as e:
    logger.warning(f"Validation failed: {e}")  # NEVER LOG AND CONTINUE
    continue

# PROHIBITED: None returns on critical operations without exceptions
def aggregate_competitor_data(df, strategy):
    try:
        return strategy.aggregate(df)
    except Exception:
        return None  # NEVER RETURN None ON CRITICAL FAILURES
```

**REQUIRED PATTERNS** (Always implement these):
```python
# REQUIRED: Immediate exception raising with comprehensive business context
try:
    adapter = get_adapter("6Y20B", config)
except Exception as e:
    raise AdapterInitializationError(
        f"CRITICAL: Failed to initialize data adapter for product 6Y20B. "
        f"Pipeline cannot continue. "
        f"Business impact: Cannot load annuity pricing data. "
        f"Required action: Verify adapter configuration and data source availability. "
        f"Original error: {e}"
    ) from e

# REQUIRED: Explicit business context in all error messages
def validate_annuity_rates(df, product_code):
    null_count = df['competitor_rate'].isna().sum()
    if null_count > 0:
        raise DataValidationError(
            f"CRITICAL: Competitor rate data contains {null_count} null values for product {product_code}. "
            f"Business context: All annuity products require complete competitor rate data for elasticity analysis. "
            f"Impact: Price elasticity estimates would be biased. "
            f"Required action: Fix data source before proceeding."
        )
```

### 1.3 Import Resilience Standards

**Status**: PROVEN (95 defensive imports, 100% success across execution contexts)
**Current Implementation**: Comprehensive defensive imports ensuring cross-context compatibility

**Proven Pattern**:
```python
try:
    from ..data.adapters import get_adapter
except ImportError:
    try:
        from src.data.adapters import get_adapter
    except ImportError:
        from data.adapters import get_adapter
```

### 1.4 Documentation Standards

**Status**: EXEMPLARY (97.06% coverage in comparable projects)
**Current Implementation**: Industry-leading documentation coverage with business context

**Required Standards**:
- Business context explanation (95% implementation achieved)
- Parameter validation examples (systematic usage)
- Mathematical formulas with Unicode symbols (where applicable)
- Usage examples with real data shapes (90% of complex functions)

---

## 2. Core Architecture Principles {#core-principles}

### 2.1 Dependency Injection Pattern (V2 Architecture)

The v2 architecture uses explicit dependency injection for clean testing and multi-product support:

```python
from src.notebooks.interface import UnifiedNotebookInterface
from src.data.adapters import get_adapter
from src.features.aggregation import get_strategy
from src.config import product_config

# Pattern: Inject dependencies through initialization
interface = UnifiedNotebookInterface(
    product_code="6Y20B",
    adapter=get_adapter("aws", config=aws_config),
    aggregation_strategy=get_strategy("weighted_aggregation", product="RILA"),
    config=product_config["6Y20B"]
)

# Alternatively, use factory pattern for simplified initialization
interface = create_interface(
    product_code="6Y20B",
    environment="aws",
    adapter_kwargs={"config": aws_config}
)
```

### 2.2 Multi-Product Architecture

```
Notebooks → UnifiedNotebookInterface → Adapters/Strategies → Results

┌─────────────────┐
│ Notebook        │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│ UnifiedNotebookInterface                             │
│  ├─ DataSourceAdapter (DI)                          │
│  │   ├─ S3Adapter (production/AWS)                  │
│  │   ├─ LocalAdapter (development)                  │
│  │   └─ FixtureAdapter (testing)                    │
│  ├─ AggregationStrategy (Product-Specific)          │
│  │   ├─ WeightedAggregation (RILA - 20%+ buffer)   │
│  │   ├─ TopNAggregation (FIA - index-linked)       │
│  │   └─ FirmLevelAggregation (MYGA - crediting)    │
│  └─ ProductMethodology                              │
│       ├─ RILAMethodology (rate-based constraint)    │
│       ├─ FIAMethodology (index-based constraint)    │
│       └─ MYGAMethodology (crediting-based)          │
└──────────────────────────────────────────────────────┘
```

### 2.3 Entry Points for V2 Architecture

| Component | Entry Point | Purpose |
|-----------|-------------|---------|
| **Notebook Interface** | `src.notebooks.interface.UnifiedNotebookInterface` | Multi-product workflow orchestration |
| **Data Adapters** | `src.data.adapters.get_adapter()` | Abstract data sources (S3/Local/Fixture) |
| **Aggregation** | `src.features.aggregation.get_strategy()` | Product-specific competitor aggregation |
| **Configuration** | `src.config.product_config` | TypedDict product configurations |
| **Methodology** | `src.products.get_methodology()` | Product-specific constraint rules |

### 2.4 DRY Principles

- **Single Responsibility**: Each function performs ONE atomic operation
- **No Multi-Step Operations**: Break complex workflows into composable atomic functions
- **Configuration-Driven**: All business parameters externalized to configuration
- **Immutable Operations**: All data transformations return new objects
- **Adapter Pattern**: Use dependency injection to swap implementations without code changes

---

## 3. Function Design Standards {#function-design}

### 3.1 Atomic Function Requirements

**Mandatory Elements:**
1. **Type Hints**: Complete typing for all parameters and return values
2. **Docstrings**: NumPy-style with business context
3. **Error Handling**: Comprehensive with business impact explanation
4. **Input Validation**: Check all required columns and data constraints
5. **Immutability**: Always return new DataFrame, never modify in-place
6. **Unit Tests**: Comprehensive test coverage for all new functions (see Section 7.4)
7. **Edge Case Testing**: Explicit tests for boundary conditions and error paths

**Length Guidelines:**
- **Target Range**: 30-50 lines for most atomic operations
- **Minimum Viable**: 10-15 lines for simple transforms
- **Exception Cases**: Configuration builders and mathematical formulations may exceed if justified

**Testing Requirements:**
- Every new function must have corresponding unit tests before merge
- Tests must cover: happy path, edge cases, error conditions, and boundary values
- See Section 7.4 for comprehensive testing requirements

### 3.2 Function Composition Patterns

```python
# Preferred: Functional composition
def create_elasticity_pipeline(config):
    """Create composable pipeline from atomic functions."""
    return compose(
        validate_rate_data,
        apply_product_filters,
        create_lag_features,
        aggregate_competitor_rates
    )

# Avoid: Monolithic multi-step functions
def process_rate_data_monolithic(df):  # AVOID THIS PATTERN
    # 100+ lines of mixed concerns
    df = validate_data(df)
    df = filter_data(df)
    df = create_features(df)
    return aggregate_data(df)
```

### 3.3 Adapter Pattern Example

```python
def process_annuity_data(
    product_code: str,
    adapter_factory: Callable[..., DataSourceAdapter],
    config: ProductConfig
) -> pd.DataFrame:
    """Process annuity data using injected adapter.

    Demonstrates dependency injection for flexible data source handling
    across production (S3), development (local), and testing (fixture) environments.

    Parameters
    ----------
    product_code : str
        Annuity product code (e.g., "6Y20B", "10Y20B")
    adapter_factory : Callable
        Factory function returning DataSourceAdapter instance
    config : ProductConfig
        TypedDict product configuration

    Returns
    -------
    pd.DataFrame
        Processed annuity data ready for elasticity analysis
    """
    # Inject adapter - allows testing without AWS access
    adapter = adapter_factory(product_code, config)

    # Use adapter to load data from any source
    df = adapter.load_historical_rates()

    # Apply product-specific business logic
    df = apply_business_filters(df, config)

    return df
```

---

## 4. Mathematical Equivalence Testing for Refactoring {#mathematical-equivalence}

**When Required**: MANDATORY during active refactoring, optional after validation
**Status**: Framework available (`src/data/mathematical_equivalence_validator.py`)
**Priority**: CRITICAL during refactoring (work preservation)

### 4.1 Quick Reference

| Standard | Value | Use Case |
|----------|-------|----------|
| **Tolerance** | 1e-12 | Deterministic transforms |
| **Target** | 0.00e+00 | Perfect equivalence goal |
| **Scope** | ALL data operations | Transforms, aggregations, calculations |

### 4.2 Detailed Documentation

The complete mathematical equivalence methodology is documented in modular form:

| Document | Purpose |
|----------|---------|
| EQUIVALENCE_WORKFLOW.md | Step-by-step refactoring process |
| TOLERANCE_REFERENCE.md | When to use which tolerance |
| FAILURE_INVESTIGATION.md | Debugging when tests fail |
| VALIDATOR_SELECTION.md | Which validator for which scenario |

### 4.3 Framework Location

- **Data Pipeline**: `src/data/mathematical_equivalence_validator.py`
- **Feature Selection**: `src/features/selection/mathematical_equivalence_validator.py`

### 4.4 CI/CD Integration

| Branch | Equivalence Tests | Behavior |
|--------|-------------------|----------|
| `main` | **Mandatory** | PR blocked if tests fail |
| `feature/*` | Advisory | Tests run, report only |
| `experiment/*` | Skip | No equivalence enforcement |

---

## 5. Configuration Management {#configuration}

### 5.1 Priority Order

**Configuration Resolution Order:**
1. Function parameters (highest priority)
2. params.yaml configuration file
3. Hardcoded defaults (lowest priority)

### 5.2 ProductConfig TypedDict Pattern

```python
from typing_extensions import TypedDict, Required, NotRequired
from typing import Dict, List, Any

class ProductConfig(TypedDict):
    """Type-safe product-specific configuration."""
    # Required product parameters
    product_code: Required[str]
    product_type: Required[str]  # "RILA", "FIA", or "MYGA"
    date_range: Required[Dict[str, str]]
    feature_config: Required[Dict[str, Any]]

    # Product-specific constraints
    buffer_percentage: NotRequired[float]
    index_participation: NotRequired[float]
    crediting_rate: NotRequired[float]

    # Optional parameters
    validation_rules: NotRequired[Dict[str, Any]]
    experiment_name: NotRequired[str]

# Usage example
def create_rila_config() -> ProductConfig:
    return ProductConfig(
        product_code="6Y20B",
        product_type="RILA",
        date_range={"start": "2022-09-01", "end": "2025-08-17"},
        feature_config={"lag_periods": [1, 2, 3, 6]},
        buffer_percentage=0.20,
        validation_rules={"positive_rates": True}
    )
```

### 5.3 Configuration Factory Pattern

```python
def build_product_configuration(
    product_code: str,
    params_file: str = "params.yaml"
) -> Dict[str, Any]:
    """Build type-safe configuration from params.yaml file."""
    import yaml

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    # Create type-safe configurations for product
    product_params = params.get(product_code, {})

    return create_product_config(product_code, product_params)
```

### 5.4 V2 Architecture Configuration Pattern

**Configuration-Based Function Interface:**
```python
def run_elasticity_analysis(
    data: pd.DataFrame,
    config: ProductConfig,
    target_variable: str = "sales_target_current"
) -> ElasticityResults:
    """
    Multi-product elasticity analysis following v2 interface pattern.

    Args:
        data: Dataset for analysis
        config: Product configuration object (TypedDict)
        target_variable: Name of target variable column

    Returns:
        NamedTuple with comprehensive results and metadata
    """
    if not config.get('enabled', True):
        raise ValueError("Analysis must be enabled in config")

    # Extract parameters from config with defaults
    lag_periods = config.get('feature_config', {}).get('lag_periods', [1, 2, 3])

    # Use adapter to get appropriate strategy
    strategy = get_strategy(config['product_type'], product=config['product_code'])

    # Apply aggregation strategy
    result = strategy.aggregate(data, lag_periods)

    return ElasticityResults(...)
```

---

## 6. Notebook Writing Standards {#notebook-standards}

### 6.1 Cell Organization

**One Pipeline Step Per Cell:**
- Each cell should represent one atomic pipeline operation
- No line limits for cells (apply to individual functions within cells)
- Clear separation between configuration, execution, and validation

### 6.2 Markdown Documentation

**Required Sections:**
```markdown
# Notebook Title - Pipeline Stage Name

## Executive Summary
Brief description of business objective and key results achieved.

## Business Context
Detailed business context and methodology explanation.
Include product-specific considerations (RILA/FIA/MYGA).

## Configuration & Setup
- Import statements with defensive imports
- Configuration loading
- Environment validation
- Multi-product initialization

## Implementation Sections
### Section 1: [Operation Name]
Business explanation of the operation.

## Results & Validation
- Performance metrics achieved
- Data quality validation results
- Output verification
- Product-specific validation rules

## Next Steps
Clear transition to next pipeline stage.
```

### 6.3 Code Cell Standards

```python
# Cell: Configuration Setup with Defensive Imports
import sys
import os
from pathlib import Path

# Add project root for clean imports
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

# Defensive imports for cross-context compatibility
try:
    from src.notebooks.interface import UnifiedNotebookInterface
    from src.config import product_config
except ImportError:
    from notebooks.interface import UnifiedNotebookInterface
    from config import product_config

# Load configuration for target product
product_code = "6Y20B"
config = product_config[product_code]
print(f"Configuration loaded for product {product_code} ({config['product_type']})")

# Initialize multi-product interface
interface = create_interface(product_code, environment="local")
```

### 6.4 Multi-Product Notebook Pattern

```python
# Cell: Multi-Product Analysis
products = ["6Y20B", "6Y10B", "10Y20B"]

results = {}
for product_code in products:
    print(f"\nProcessing {product_code}...")

    # Initialize interface for each product
    interface = create_interface(
        product_code,
        environment="local"
    )

    # Load and analyze
    df = interface.load_data()
    elasticity = interface.run_elasticity_analysis(df)

    results[product_code] = elasticity

    print(f"Elasticity for {product_code}: {elasticity.own_rate_elasticity:.4f}")

# Comparative analysis across products
comparison_df = pd.DataFrame({
    product: [results[product].own_rate_elasticity for product in products]
})
print(comparison_df)
```

---

## 7. Testing & Validation Framework {#testing}

### 7.1 Testing Approach

**Testing Philosophy:**
- **Mandatory Coverage**: All new functions require unit tests before merge
- **Quality AND Quantity**: Comprehensive coverage with meaningful assertions
- **Test-Driven Mindset**: Write tests during or immediately after function implementation
- **Edge Case Focus**: Explicit tests for boundary conditions and error paths
- **Business Logic Validation**: Thorough testing of all business rules
- **Mathematical Precision**: Validate numerical operations with high precision

### 7.2 Validation Framework

```python
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class UnifiedValidationFramework:
    """Comprehensive validation framework for business operations."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.results: List[Dict[str, Any]] = []

    def validate_dataframe_shape(self, df: pd.DataFrame, expected_shape: tuple, test_name: str) -> bool:
        """Validate DataFrame shape with detailed reporting."""
        passed = df.shape == expected_shape

        result = {
            'test_name': test_name,
            'passed': passed,
            'expected': expected_shape,
            'actual': df.shape,
            'message': f"Shape validation: expected {expected_shape}, got {df.shape}"
        }

        self.results.append(result)
        return passed

    def validate_business_logic(self, df: pd.DataFrame, rules: Dict[str, Any], test_name: str) -> bool:
        """Validate business logic constraints."""
        violations = []

        for rule_name, rule_config in rules.items():
            if rule_name == "positive_rates":
                if (df['competitor_rate'] < 0).any():
                    violations.append("Found negative rate values")

            elif rule_name == "date_continuity":
                date_gaps = df['date'].diff().dt.days
                max_gap = rule_config.get('max_gap_days', 14)
                if (date_gaps > max_gap).any():
                    violations.append(f"Date gaps exceed {max_gap} days")

        passed = len(violations) == 0

        result = {
            'test_name': test_name,
            'passed': passed,
            'violations': violations,
            'message': f"Business validation: {'PASSED' if passed else 'FAILED'}"
        }

        if violations:
            result['message'] += f". Violations: {'; '.join(violations)}"

        self.results.append(result)
        return passed

    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])

        report = f"""
=== VALIDATION REPORT ===

Summary:
  Total Tests: {total_tests}
  Passed: {passed_tests}
  Failed: {total_tests - passed_tests}
  Success Rate: {passed_tests/total_tests*100:.1f}%

Detailed Results:
"""

        for result in self.results:
            status = "PASS" if result['passed'] else "FAIL"
            report += f"  {status} {result['test_name']}: {result['message']}\n"

        return report
```

### 7.3 pytest Integration

```python
class TestBusinessLogic:
    """Test suite for business logic validation."""

    def setup_method(self):
        """Setup test environment."""
        self.validator = UnifiedValidationFramework()

    def test_rate_data_integrity(self):
        """Test rate data meets business requirements."""
        # Load test data
        df = load_test_rate_data()

        # Define business rules
        rules = {
            "positive_rates": True,
            "date_continuity": {"max_gap_days": 7}
        }

        # Validate
        assert self.validator.validate_business_logic(df, rules, "rate_integrity")

    def test_elasticity_calculation_output(self):
        """Test elasticity calculation produces expected results."""
        # Test implementation
        input_df = create_test_dataframe()
        config = create_test_config()

        result_df = calculate_elasticity(input_df, config)

        # Validate output shape and content
        assert 'own_rate_elasticity' in result_df.columns
        assert self.validator.validate_dataframe_shape(result_df, (100, 15), "elasticity_output")
```

### 7.4 Mandatory Unit Testing Requirements {#mandatory-unit-testing-requirements}

**CRITICAL: All new functions must include comprehensive unit tests before merge.**

#### 7.4.1 Test Coverage Requirements

**Minimum Test Coverage for Every Function:**

1. **Happy Path Test**: Test the primary use case with valid, typical inputs
2. **Edge Cases**: Test boundary conditions (empty data, single row, maximum values, etc.)
3. **Error Conditions**: Test all expected error paths and exceptions
4. **Type Validation**: Test with incorrect types to verify type hints are enforced
5. **Business Logic**: Test all business rules and constraints
6. **Immutability**: Verify function does not modify input objects

**Coverage Targets:**
- **New Functions**: 100% line coverage required before merge
- **Modified Functions**: Must maintain or improve existing coverage
- **Critical Path Functions**: Must include property-based tests (hypothesis) when applicable

#### 7.4.2 Unit Test Template

```python
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import the function under test
from src.features.elasticity import calculate_elasticity

class TestElasticityCalculation:
    """Comprehensive test suite for calculate_elasticity."""

    # --- Setup and Fixtures ---

    @pytest.fixture
    def valid_rate_dataframe(self) -> pd.DataFrame:
        """Create valid test DataFrame for happy path tests."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'own_rate': [2.5, 2.7, 2.9, 3.1, 3.0, 3.2, 3.1, 3.3, 3.2, 3.4],
            'competitor_rate': [2.4, 2.6, 2.8, 3.0, 2.9, 3.1, 3.0, 3.2, 3.1, 3.3],
            'annuity_sales': [100, 150, 200, 175, 225, 250, 300, 275, 325, 350]
        })

    @pytest.fixture
    def valid_config(self) -> Dict[str, Any]:
        """Create valid configuration for tests."""
        return {
            'product_code': '6Y20B',
            'lag_periods': [1, 2, 3],
            'required_columns': ['date', 'own_rate', 'competitor_rate', 'annuity_sales']
        }

    # --- Happy Path Tests ---

    def test_happy_path_typical_inputs(self, valid_rate_dataframe, valid_config):
        """Test function with valid, typical inputs."""
        result = calculate_elasticity(valid_rate_dataframe, valid_config)

        # Verify output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(valid_rate_dataframe)

        # Verify expected columns exist
        expected_cols = ['own_rate_elasticity', 'competitor_elasticity']
        assert all(col in result.columns for col in expected_cols)

        # Verify elasticity magnitude is reasonable
        assert -2.0 < result['own_rate_elasticity'].mean() < 0.0

    # --- Edge Case Tests ---

    def test_edge_case_empty_dataframe(self, valid_config):
        """Test function handles empty DataFrame gracefully."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Cannot process empty DataFrame"):
            calculate_elasticity(empty_df, valid_config)

    def test_edge_case_single_row(self, valid_config):
        """Test function with minimum viable data (single row)."""
        single_row_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-01')],
            'own_rate': [2.5],
            'competitor_rate': [2.4],
            'annuity_sales': [100]
        })

        result = calculate_elasticity(single_row_df, valid_config)

        # Verify function handles single row appropriately
        assert len(result) == 1
        # Elasticity should be NaN for first row (no lag)
        assert pd.isna(result['own_rate_elasticity'].iloc[0])

    # --- Error Condition Tests ---

    def test_error_missing_required_columns(self, valid_config):
        """Test function raises appropriate error for missing columns."""
        incomplete_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'own_rate': range(10)
            # Missing 'competitor_rate' and 'annuity_sales'
        })

        with pytest.raises(ValueError, match="Missing required.*columns"):
            calculate_elasticity(incomplete_df, valid_config)

    # --- Business Logic Tests ---

    def test_business_rule_positive_rates_only(self, valid_config):
        """Test business rule: rates must be non-negative."""
        negative_rate_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'own_rate': [-2.5, 2.7, 2.9, 3.1, 3.0, 3.2, 3.1, 3.3, 3.2, 3.4],
            'competitor_rate': [2.4, 2.6, 2.8, 3.0, 2.9, 3.1, 3.0, 3.2, 3.1, 3.3],
            'annuity_sales': list(range(100, 1100, 100))
        })

        with pytest.raises(ValueError, match="rates.*negative|positive"):
            calculate_elasticity(negative_rate_df, valid_config)

    # --- Immutability Tests ---

    def test_immutability_input_not_modified(self, valid_rate_dataframe, valid_config):
        """Test function does not modify input DataFrame."""
        original_df = valid_rate_dataframe.copy()
        original_cols = list(original_df.columns)
        original_len = len(original_df)

        result = calculate_elasticity(valid_rate_dataframe, valid_config)

        # Verify input DataFrame unchanged
        assert list(valid_rate_dataframe.columns) == original_cols
        assert len(valid_rate_dataframe) == original_len
        pd.testing.assert_frame_equal(valid_rate_dataframe, original_df)

        # Verify result is a new object
        assert result is not valid_rate_dataframe

    # --- Mathematical Precision Tests ---

    def test_mathematical_precision_elasticity_calculations(self, valid_rate_dataframe, valid_config):
        """Test mathematical correctness of elasticity calculations."""
        result = calculate_elasticity(valid_rate_dataframe, valid_config)

        # Verify elasticity calculations are mathematically precise
        for i in range(3, len(result)):
            # Verify own-rate elasticity magnitude
            own_elasticity = result['own_rate_elasticity'].iloc[i]
            assert abs(own_elasticity) < 10.0, f"Elasticity unreasonably large at row {i}"
```

#### 7.4.3 Edge Case Catalog

**Common Edge Cases to Test for All Functions:**

1. **Data Size Edge Cases:**
   - Empty DataFrame (0 rows)
   - Single row DataFrame
   - Very large DataFrame (performance test)
   - Single column DataFrame

2. **Value Edge Cases:**
   - All zeros
   - All same values (no variance)
   - Missing values (NaN, None, pd.NA)
   - Negative values (if applicable)
   - Maximum/minimum numerical values
   - Infinity and negative infinity
   - Very small values (underflow risk)
   - Very large values (overflow risk)

3. **Type Edge Cases:**
   - Wrong data types
   - Mixed types in columns
   - String representations of numbers
   - Unicode and special characters

4. **Configuration Edge Cases:**
   - Empty configuration
   - Missing required keys
   - Invalid parameter values
   - Extreme parameter values (very large/small)

5. **Business Logic Edge Cases:**
   - Data violating business rules
   - Date gaps or discontinuities
   - Duplicate records
   - Missing categorical values

#### 7.4.4 pytest Best Practices

**Test Organization:**
```python
# tests/test_elasticity.py
"""
Comprehensive test suite for elasticity module.

Test organization:
- Setup and fixtures
- Happy path tests
- Edge case tests
- Error condition tests
- Business logic tests
- Immutability tests
- Mathematical precision tests
- Integration tests
"""
```

**Test Naming Convention:**
- `test_happy_path_<scenario>`: Tests for expected, typical usage
- `test_edge_case_<condition>`: Tests for boundary conditions
- `test_error_<error_type>`: Tests for error handling
- `test_business_rule_<rule_name>`: Tests for specific business rules
- `test_immutability_<aspect>`: Tests for immutability guarantees
- `test_mathematical_precision_<calculation>`: Tests for numerical accuracy
- `test_integration_<integration_point>`: Tests for cross-module interactions

**Assertion Best Practices:**
- Use descriptive assertion messages
- Test one concept per test function
- Use `pytest.raises()` for exception testing with message matching
- Use `pytest.approx()` for floating-point comparisons
- Use `pd.testing.assert_frame_equal()` for DataFrame comparisons

#### 7.4.5 Test Execution Requirements

**Pre-Merge Requirements:**
```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Minimum coverage thresholds
# - New functions: 100% line coverage
# - Modified functions: No coverage regression
# - Overall project: Progressive improvement toward 90%+
```

**Continuous Integration:**
- All tests must pass before merge
- Coverage reports generated automatically
- Regression tests run on every commit
- Mathematical equivalence tests run during refactoring

---

## 8. Visual and Communication Standards {#visual-standards}

### 8.1 Emoji Policy

**Strict No-Emoji Policy:**
- **PROHIBITED**: All emojis in code, comments, docstrings, markdown, **and git commit messages**
- **ONLY EXCEPTION**: Red circle character (🔴) for error status indicators when errors actually occur
- **Rationale**: Professional presentation, cross-platform compatibility, and commit message parseability

**Git Commit Message Standards:**
- Use clear, descriptive language without decorative elements or emojis
- Focus on "what changed" and "why it changed"
- Format: Present tense, imperative mood (e.g., "Add feature X" not "Added feature X")
- Length: Subject line ≤72 characters, detailed body if needed
- Examples: "Add elasticity calculation for RILA products", "Fix rate validation error in adapter"

### 8.2 Error Communication

```python
# Correct error reporting with 🔴 (only when error occurs)
try:
    result = elasticity_analysis(data)
except Exception as e:
    logger.error(f"🔴 Elasticity analysis failed: {e}")
    raise

# Correct success communication (no emojis)
print(f"Analysis completed successfully. Processed {len(df)} annuity records.")

# INCORRECT: Decorative emojis (never use)
# print(f"✨ Data processed successfully! 🎉")  # NEVER DO THIS
```

### 8.3 Professional Communication

**Written Communication:**
- Clear, direct language focused on business impact
- Technical accuracy without unnecessary complexity
- Comprehensive context for error messages
- Specific actionable guidance for problem resolution

### 8.4 Git Commit Message Format

**Subject Line Requirements:**
- Maximum 72 characters for git log readability
- Imperative mood: "Fix bug" not "Fixed bug" or "Fixes bug"
- Capitalize first word: "Add feature" not "add feature"
- No trailing period or punctuation
- Be specific and descriptive, avoid vague terms

**Multi-line Commit Body Format:**
```
Subject line (≤72 chars, imperative mood, capitalized)

Blank line separating subject from body (mandatory)

Body paragraphs explaining the what, why, and how:
- Wrap all lines at 72 characters for consistent formatting
- Use bullet points for multiple changes or considerations
- Explain motivation and context, not just what changed
- Include business impact when relevant
- Reference issues/tickets if applicable

Breaking changes should be clearly noted with rationale.
Migration steps should be provided for compatibility.

Closes #123
Fixes #456

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Examples:**

**Good Single-line Commit:**
```
Add elasticity calculation for RILA products
```

**Good Multi-line Commit:**
```
Add multi-product adapter pattern to UnifiedNotebookInterface

Implement dependency injection pattern for data adapters to support
RILA, FIA, and MYGA products. Allows runtime selection of S3, Local,
or Fixture data sources without code changes.

Key improvements:
- Add DataSourceAdapter protocol in src/core/protocols.py
- Implement S3Adapter, LocalAdapter, FixtureAdapter in src/data/adapters/
- Update UnifiedNotebookInterface to accept injected adapter
- Add comprehensive adapter tests with 100% coverage

This addresses the need for multi-product testing support and
improves code reusability across product lines.

Breaking change: Existing notebook initialization must specify
environment parameter. See MIGRATION.md for upgrade instructions.

Closes #456
```

---

## 9. Implementation Checklist {#implementation-checklist}

### 9.1 Immediate Actions for V2 Architecture

**Function Standards:**
- [ ] Enforce 30-50 line limit for atomic functions (NO multi-step operations)
- [ ] Allow exceptions ONLY for config builders and complex business logic with math
- [ ] Mandate complete type hints for ALL functions
- [ ] Use comprehensive error handling with business context everywhere
- [ ] Implement fail-fast error patterns (no silent failures)

**Multi-Product Architecture:**
- [ ] Implement dependency injection across all data adapters
- [ ] Create ProductConfig TypedDict for all products (RILA, FIA, MYGA)
- [ ] Build aggregation strategy factory for product-specific competitor handling
- [ ] Ensure UnifiedNotebookInterface supports all three product types
- [ ] Add adapter selection logic to notebook initialization

**Mathematical Equivalence:**
- [ ] Implement mathematical equivalence validation framework
- [ ] Create pre-commit hooks for refactoring validation
- [ ] Establish 1e-12 precision requirement for all data operations
- [ ] Target 0.00e+00 (perfect equivalence) achievement

**Visual Standards:**
- [ ] Remove all emojis from code, comments, docstrings, markdown
- [ ] Keep ONLY 🔴 for error status when errors occur
- [ ] Update all existing notebooks to remove emojis

**Configuration:**
- [ ] Implement Function parameters → params.yaml → defaults priority order
- [ ] Create params.yaml for all product configurations
- [ ] Design all config loading with TypedDict type safety
- [ ] Add defensive import patterns to all modules

**Notebook Standards:**
- [ ] One pipeline step per cell
- [ ] Markdown-heavy documentation (business + technical details)
- [ ] Create real table of contents with hyperlinks
- [ ] Apply same function standards as .py files
- [ ] Remove all emojis

**Testing:**
- [ ] **MANDATORY**: Write unit tests for ALL new functions before merge
- [ ] Implement comprehensive edge case testing (see Section 7.4.3)
- [ ] Test all error conditions and exception paths
- [ ] Verify immutability of all data transformations
- [ ] Achieve 100% line coverage for new functions
- [ ] Maintain or improve coverage for modified functions
- [ ] Implement mathematical equivalence testing during refactoring
- [ ] Create comprehensive business logic validation
- [ ] Run full test suite before merge: `pytest tests/ --cov=src`
- [ ] Verify all tests pass in CI/CD pipeline

### 9.2 Success Criteria

**Technical Excellence:**
- **Zero Ambiguity**: All standards questions resolved with clear decisions
- **DRY Compliance**: Single responsibility functions, no multi-step operations
- **Mathematical Precision**: 1e-12 tolerance achieved for all refactoring
- **Type Safety**: 100% TypedDict usage for all configurations
- **Multi-Product Support**: Seamless operation across RILA, FIA, MYGA products

**Professional Presentation:**
- **Clean Communication**: No decorative emojis, comprehensive documentation
- **Business Context**: All errors include business impact and required actions
- **Comprehensive Testing**: Critical paths and error conditions thoroughly validated

**Future-Ready Architecture:**
- **MLOps Compatibility**: Easy integration when ready, no refactoring needed
- **Research-Optimized**: Configuration and workflow designed for research iteration
- **Maintainability**: Clear patterns and comprehensive documentation throughout
- **Multi-Product Extensibility**: Adapter pattern enables new products without core changes

---

**Document Provenance:**
- **Based On**: RILA Price Elasticity CODING_STANDARDS.md (v3.0)
- **Adapted For**: Multi-Product Annuity Price Elasticity v2
- **Key Adaptations**:
  - Entry points updated to v2 architecture
  - Product-specific configuration examples added
  - Dependency injection patterns emphasized
  - Multi-product testing scenarios included
  - Audit compliance updated to 92/100 v2 baseline
- **Status**: Authoritative single source of truth for v2 development

This document provides crystal-clear, conflict-free guidelines that maintain all RILA standards while establishing v2 architecture principles for multi-product support with dependency injection and comprehensive testing.
