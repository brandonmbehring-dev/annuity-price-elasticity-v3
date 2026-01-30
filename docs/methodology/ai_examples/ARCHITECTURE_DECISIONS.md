# Architecture Decision Examples

**Last Updated:** 2026-01-30
**Purpose:** Worked examples of AI-assisted architecture decisions

---

## Overview

This document provides detailed examples of architecture decisions made during AI-assisted development. Each example shows:
- The context and constraints
- Options considered
- Decision rationale
- Implementation approach

---

## Decision 1: Dependency Injection Pattern

### Context

**Problem**: Need to support multiple data sources (AWS S3, local files, test fixtures) without code duplication.

**Constraints**:
- Production uses AWS S3 with STS authentication
- Development should work offline
- Tests need deterministic data
- All sources must produce identical DataFrame schemas

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A: If/else branching** | Simple | Violates OCP, hard to test |
| **B: Factory pattern** | Extensible | Still couples code to implementations |
| **C: Dependency Injection** | Testable, extensible | More complex initial setup |
| **D: Strategy pattern** | Flexible | Overkill for data sources |

### Decision

**Option C: Dependency Injection** with Protocol-based adapters

### Rationale

1. **Testability**: Can inject fixture adapter in tests
2. **Extensibility**: New adapters don't modify existing code
3. **Type Safety**: Protocol ensures consistent interface
4. **Separation of Concerns**: Interface knows nothing about data sources

### Implementation

```python
# src/core/protocols.py
from typing import Protocol
import pandas as pd

class DataSourceAdapter(Protocol):
    """Protocol for data source adapters."""

    def load_sales_data(self) -> pd.DataFrame:
        """Load sales data from source."""
        ...

    def load_competitive_rates(self) -> pd.DataFrame:
        """Load competitive rate data from source."""
        ...

# src/data/adapters/s3_adapter.py
class S3Adapter:
    """AWS S3 data source adapter."""

    def __init__(self, config: S3Config):
        self.config = config
        self._client = None

    def load_sales_data(self) -> pd.DataFrame:
        return self._read_parquet(self.config.sales_path)

    def load_competitive_rates(self) -> pd.DataFrame:
        return self._read_parquet(self.config.rates_path)

# src/data/adapters/fixture_adapter.py
class FixtureAdapter:
    """Test fixture data source adapter."""

    def __init__(self, fixture_dir: Path):
        self.fixture_dir = fixture_dir

    def load_sales_data(self) -> pd.DataFrame:
        return pd.read_parquet(self.fixture_dir / "sales.parquet")

    def load_competitive_rates(self) -> pd.DataFrame:
        return pd.read_parquet(self.fixture_dir / "rates.parquet")

# src/notebooks/interface.py
class UnifiedNotebookInterface:
    """Main interface using dependency injection."""

    def __init__(self, adapter: DataSourceAdapter):
        self._adapter = adapter

    def load_data(self) -> pd.DataFrame:
        return self._adapter.load_sales_data()
```

### Usage Pattern

```python
# Production
interface = create_interface("6Y20B", environment="aws")

# Development
interface = create_interface("6Y20B", environment="fixture")

# Custom adapter
custom_adapter = MyCustomAdapter(config)
interface = UnifiedNotebookInterface(adapter=custom_adapter)
```

---

## Decision 2: Product Registry Pattern

### Context

**Problem**: Product-specific configurations (coefficients, constraints, validation rules) scattered across codebase.

**Constraints**:
- Multiple products: RILA 6Y20B, 6Y10B, 10Y20B, FIA variants
- Each product has unique constraint signs
- Need centralized lookup
- Must be extensible for new products

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A: Hardcoded dicts** | Simple | No validation, scattered |
| **B: Config files (YAML)** | Declarative | Loses type safety |
| **C: Registry pattern** | Centralized, typed | More code |
| **D: Database** | Dynamic | Overkill, latency |

### Decision

**Option C: Registry pattern** with TypedDict configurations

### Rationale

1. **Single Source of Truth**: All product configs in one place
2. **Type Safety**: TypedDict catches errors at lint time
3. **Extensibility**: Register new products without modifying core code
4. **Validation**: Registry validates on registration

### Implementation

```python
# src/core/types.py
from typing import TypedDict

class ProductConfig(TypedDict):
    """Product configuration structure."""
    product_code: str
    product_type: str  # RILA, FIA, MYGA
    buffer_level: float
    term_years: int
    own_rate_sign: int  # +1 or -1
    competitor_sign: int  # +1 or -1
    min_lag_weeks: int

# src/core/registry.py
class ProductRegistry:
    """Centralized product configuration registry."""

    _products: dict[str, ProductConfig] = {}

    @classmethod
    def register(cls, config: ProductConfig) -> None:
        """Register a product configuration."""
        cls._validate(config)
        cls._products[config["product_code"]] = config

    @classmethod
    def get(cls, product_code: str) -> ProductConfig:
        """Get product configuration."""
        if product_code not in cls._products:
            raise UnknownProductError(
                f"Product {product_code} not registered. "
                f"Available: {list(cls._products.keys())}"
            )
        return cls._products[product_code]

    @classmethod
    def _validate(cls, config: ProductConfig) -> None:
        """Validate configuration constraints."""
        if config["own_rate_sign"] not in (1, -1):
            raise ConfigValidationError("own_rate_sign must be +1 or -1")
        if config["min_lag_weeks"] < 1:
            raise ConfigValidationError("min_lag_weeks must be >= 1")

# Registration (in src/products/__init__.py)
ProductRegistry.register({
    "product_code": "6Y20B",
    "product_type": "RILA",
    "buffer_level": 0.20,
    "term_years": 6,
    "own_rate_sign": 1,
    "competitor_sign": -1,
    "min_lag_weeks": 2,
})
```

---

## Decision 3: TypedDict vs Dataclass

### Context

**Problem**: Need typed configuration structures for product configs, adapter configs, validation rules.

**Constraints**:
- Configurations often come from JSON/YAML
- Need type checking at lint time
- Some configs are immutable, some mutable
- Performance matters for hot paths

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A: Plain dict** | Simple, flexible | No type safety |
| **B: TypedDict** | Type safe, dict-compatible | No runtime validation |
| **C: dataclass** | Runtime validation possible | Serialization overhead |
| **D: Pydantic** | Full validation | Heavy dependency |
| **E: attrs** | Modern, validated | Another dependency |

### Decision

**Hybrid approach**:
- **TypedDict** for static configurations (JSON-serializable, immutable)
- **dataclass** for runtime objects (need methods, mutable state)

### Rationale

1. **TypedDict**: Perfect for configs that come from files
   - Direct JSON compatibility
   - Zero runtime overhead
   - Mypy catches type errors

2. **dataclass**: Better for runtime objects
   - Can have methods
   - `frozen=True` for immutability
   - Better repr and equality

### Implementation Examples

```python
# TypedDict for JSON-sourced configs
from typing import TypedDict

class S3Config(TypedDict):
    """S3 configuration from JSON."""
    bucket: str
    prefix: str
    region: str

# Usage: directly from JSON
config: S3Config = json.load(f)
adapter = S3Adapter(config)

# dataclass for runtime objects
from dataclasses import dataclass

@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result."""
    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def has_errors(self) -> bool:
        return len(self.errors) > 0

# Usage: constructed at runtime
result = ValidationResult(
    is_valid=False,
    errors=("Lag-0 competitor detected",),
    warnings=()
)
```

---

## Decision 4: Exception Hierarchy

### Context

**Problem**: Errors need business context for effective debugging and user communication.

**Constraints**:
- Different error types (data, model, config, validation)
- All errors need business impact
- All errors need required actions
- Must be catchable at appropriate levels

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A: Built-in exceptions** | Simple | No business context |
| **B: String messages** | Flexible | No structure |
| **C: Exception hierarchy** | Structured, typed | More code |
| **D: Error codes** | Numeric tracking | Hard to read |

### Decision

**Option C: Exception hierarchy** with mandatory business context

### Rationale

1. **Structured errors**: Can catch by category
2. **Business context**: Every error explains impact
3. **Required actions**: User knows what to do
4. **Type safety**: IDE support for handling

### Implementation

```python
# src/core/exceptions.py
class ElasticityBaseError(Exception):
    """Base exception with business context."""

    def __init__(
        self,
        message: str,
        business_impact: str,
        required_action: str
    ):
        self.message = message
        self.business_impact = business_impact
        self.required_action = required_action
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return (
            f"{self.message}\n"
            f"Business Impact: {self.business_impact}\n"
            f"Required Action: {self.required_action}"
        )

class DataLoadError(ElasticityBaseError):
    """Errors during data loading."""
    pass

class ConstraintViolationError(ElasticityBaseError):
    """Economic constraint violations."""
    pass

class ConfigValidationError(ElasticityBaseError):
    """Configuration validation failures."""
    pass

# Usage
raise DataLoadError(
    message="Failed to load sales data from S3",
    business_impact="Cannot run elasticity analysis",
    required_action="Check AWS credentials and bucket permissions"
)

# Catching by category
try:
    result = run_analysis()
except DataLoadError:
    # Handle data issues
    log_and_retry()
except ConstraintViolationError:
    # Handle model issues
    alert_team()
except ElasticityBaseError:
    # Handle any business error
    show_user_message()
```

---

## Decision 5: Aggregation Strategy Pattern

### Context

**Problem**: Different products require different competitor aggregation methods.

**Constraints**:
- RILA: Market-share weighted aggregation
- FIA: Top-N competitors by rate
- MYGA: Firm-level aggregation
- Must be extensible for new products
- Must be testable in isolation

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A: If/else in code** | Simple | Violates OCP |
| **B: Config-driven** | Declarative | Limited flexibility |
| **C: Strategy pattern** | Extensible, testable | More classes |
| **D: Function registry** | Simple, functional | No state |

### Decision

**Option C: Strategy pattern** with Protocol interface

### Rationale

1. **Extensibility**: New strategies don't modify existing code
2. **Testability**: Each strategy testable in isolation
3. **Type Safety**: Protocol ensures consistent interface
4. **Stateful**: Strategies can maintain configuration

### Implementation

```python
# src/core/protocols.py
class AggregationStrategy(Protocol):
    """Protocol for competitor aggregation strategies."""

    def aggregate(
        self,
        competitor_rates: pd.DataFrame,
        market_shares: pd.DataFrame | None = None
    ) -> pd.Series:
        """Aggregate competitor rates to single value."""
        ...

# src/features/aggregation/weighted.py
class WeightedAggregation:
    """Market-share weighted aggregation for RILA."""

    def aggregate(
        self,
        competitor_rates: pd.DataFrame,
        market_shares: pd.DataFrame | None = None
    ) -> pd.Series:
        if market_shares is None:
            raise ValueError("WeightedAggregation requires market_shares")

        weights = market_shares / market_shares.sum()
        return (competitor_rates * weights).sum(axis=1)

# src/features/aggregation/top_n.py
class TopNAggregation:
    """Top-N competitor aggregation for FIA."""

    def __init__(self, n: int = 5):
        self.n = n

    def aggregate(
        self,
        competitor_rates: pd.DataFrame,
        market_shares: pd.DataFrame | None = None
    ) -> pd.Series:
        # For each row, average top N rates
        return competitor_rates.apply(
            lambda row: row.nlargest(self.n).mean(),
            axis=1
        )

# Usage via factory
def get_strategy(product_type: str) -> AggregationStrategy:
    strategies = {
        "RILA": WeightedAggregation(),
        "FIA": TopNAggregation(n=5),
        "MYGA": FirmLevelAggregation(),
    }
    return strategies[product_type]
```

---

## Summary Table

| Decision | Pattern | Key Benefit |
|----------|---------|-------------|
| Data Sources | Dependency Injection | Testability |
| Product Configs | Registry | Single source of truth |
| Config Types | TypedDict + dataclass | Type safety + flexibility |
| Errors | Exception Hierarchy | Business context |
| Aggregation | Strategy | Extensibility |

---

## Related Documentation

- [../AI_COLLABORATION.md](../AI_COLLABORATION.md) - Overview
- [DOMAIN_DECISIONS.md](DOMAIN_DECISIONS.md) - Domain-specific decisions
- [../../development/CLAUDE.md](../../development/CLAUDE.md) - Development guidance
- [../../../.tracking/decisions.md](../../../.tracking/decisions.md) - Decision log

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-30 | Initial creation | Claude |
