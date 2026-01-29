# Product Extension Guide

**Version**: 1.0
**Date**: 2026-01-24
**Status**: Production-Ready for RILA

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Step 1: Define Product Methodology](#3-step-1-define-product-methodology)
4. [Step 2: Register in BusinessRulesRegistry](#4-step-2-register-in-businessrulesregistry)
5. [Step 3: Add to PRODUCT_REGISTRY](#5-step-3-add-to-product_registry)
6. [Step 4: Configure Aggregation Strategy](#6-step-4-configure-aggregation-strategy)
7. [Step 5: Testing Requirements](#7-step-5-testing-requirements)
8. [Appendix A: MYGA Implementation Checklist](#8-appendix-a-myga-implementation-checklist)
9. [Appendix B: Extension Points Reference](#9-appendix-b-extension-points-reference)

---

## 1. Introduction

This guide provides step-by-step instructions for extending the RILA Price Elasticity V2 system to support new product types. The architecture uses dependency injection and protocol-based abstractions to enable clean multi-product support.

### Current Product Support

| Product Type | Status | Notes |
|--------------|--------|-------|
| RILA | Production | 6Y20B, 6Y10B, 10Y20B fully supported |
| FIA | Partial | Methodology defined, needs end-to-end testing |
| MYGA | Stub | Fail-fast guardrails, needs domain implementation |

### Prerequisites

Before extending the system, ensure you understand:
- Economic constraints for the product type
- Expected coefficient signs and their rationale
- Data leakage patterns specific to the product
- Aggregation strategy for competitor rates

---

## 2. Architecture Overview

### Core Abstractions

```
┌─────────────────────────────────────────────────────────────┐
│                  UnifiedNotebookInterface                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ DataSourceAdapter│  │AggregationStrat │  │ProductMethod │ │
│  │   (Protocol)     │  │   (Protocol)    │  │  (Protocol)  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬───────┘ │
│           │                    │                   │         │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌──────▼───────┐ │
│  │ S3Adapter       │  │ WeightedAggr    │  │ RILAMethod   │ │
│  │ LocalAdapter    │  │ TopNAggr        │  │ FIAMethod    │ │
│  │ FixtureAdapter  │  │ FirmLevelAggr   │  │ MYGAMethod   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Files

| Purpose | Location |
|---------|----------|
| Product Config | `src/config/product_config.py` |
| Methodology Base | `src/products/base.py` |
| RILA Methodology | `src/products/rila_methodology.py` |
| Aggregation Strategies | `src/features/aggregation/` |
| Interface | `src/notebooks/interface.py` |

---

## 3. Step 1: Define Product Methodology

Create a new methodology class that defines economic constraints and coefficient expectations.

### Template: `src/products/new_methodology.py`

```python
"""
New Product Methodology Implementation.

Defines economic constraint rules and coefficient expectations for
[Product Type] products.

[Product Type] products have different dynamics than RILA:
- [Key difference 1]
- [Key difference 2]
- [Key difference 3]

Usage:
    from src.products.new_methodology import NewProductMethodology

    methodology = NewProductMethodology()
    rules = methodology.get_constraint_rules()
"""

from typing import List, Dict
from src.products.base import ConstraintRule


class NewProductMethodology:
    """[Product Type]-specific methodology implementation.

    [Detailed description of product economics and modeling approach]

    Attributes
    ----------
    product_type : str
        Always "[product_type]" for this implementation
    """

    @property
    def product_type(self) -> str:
        return "[product_type]"  # e.g., "fia", "myga"

    def get_constraint_rules(self) -> List[ConstraintRule]:
        """Get [Product Type]-specific economic constraint rules.

        Returns
        -------
        List[ConstraintRule]
            Constraint rules for model validation

        Notes
        -----
        Economic rationale for each constraint:
        1. [Constraint 1]: [Rationale]
        2. [Constraint 2]: [Rationale]
        """
        return [
            ConstraintRule(
                name="no_lag_zero_competitor",
                description="Competitor rates cannot use lag-0 (causal identification)",
                constraint_type="NO_LAG_ZERO_COMPETITOR",
                parameters={"allowed_lags": ["t1", "t2", "t3", "t4", "t5"]},
            ),
            ConstraintRule(
                name="own_rate_positive",
                description="Own rate should have positive coefficient",
                constraint_type="SIGN_CONSTRAINT",
                parameters={"pattern": "own_rate", "expected_sign": "positive"},
            ),
            # Add product-specific constraints here
        ]

    def get_coefficient_signs(self) -> Dict[str, str]:
        """Get expected coefficient signs by feature pattern.

        Returns
        -------
        Dict[str, str]
            Mapping of feature pattern to expected sign

        Notes
        -----
        Sign expectations based on [economic theory/empirical evidence]:
        - own_rate: [positive/negative] because [reason]
        - competitor: [positive/negative] because [reason]
        """
        return {
            "own_rate": "positive",      # Higher own rates attract customers
            "competitor": "negative",     # Higher competitor rates reduce share
            # Add product-specific patterns
        }

    def supports_regime_detection(self) -> bool:
        """Whether this product type benefits from regime detection.

        Returns
        -------
        bool
            True if product behavior varies by market regime
        """
        return False  # Override if product has regime-dependent behavior

    def get_leakage_patterns(self) -> List[str]:
        """Get patterns that indicate potential data leakage.

        Returns
        -------
        List[str]
            Regex patterns for leakage-prone features
        """
        return [
            r".*_t0$",           # Lag-0 features
            r".*_current$",      # Current-period features
            r".*_forward.*",     # Forward-looking features
            r".*_future.*",      # Future features
        ]


__all__ = ["NewProductMethodology"]
```

### Required Methods

| Method | Purpose | Return Type |
|--------|---------|-------------|
| `product_type` | Product identifier | `str` |
| `get_constraint_rules()` | Economic validation rules | `List[ConstraintRule]` |
| `get_coefficient_signs()` | Expected coefficient signs | `Dict[str, str]` |
| `get_leakage_patterns()` | Data leakage detection | `List[str]` |

### Optional Methods

| Method | Purpose | Default |
|--------|---------|---------|
| `supports_regime_detection()` | Enable regime analysis | `False` |

---

## 4. Step 2: Register in BusinessRulesRegistry

Add the new methodology to the registry for automatic discovery.

### Update `src/products/__init__.py`

```python
"""Product methodologies package."""

from src.products.rila_methodology import RILAMethodology
from src.products.fia_methodology import FIAMethodology
from src.products.new_methodology import NewProductMethodology  # Add import

# Registry mapping product types to methodology classes
_METHODOLOGY_REGISTRY = {
    "rila": RILAMethodology,
    "fia": FIAMethodology,
    "[product_type]": NewProductMethodology,  # Add registration
}


def get_methodology(product_type: str):
    """Get methodology for a product type.

    Parameters
    ----------
    product_type : str
        Product type identifier ("rila", "fia", "[product_type]")

    Returns
    -------
    ProductMethodology
        Methodology instance for the product type

    Raises
    ------
    KeyError
        If product type not registered
    """
    if product_type not in _METHODOLOGY_REGISTRY:
        available = ", ".join(_METHODOLOGY_REGISTRY.keys())
        raise KeyError(
            f"Unknown product type: {product_type}. Available: {available}"
        )
    return _METHODOLOGY_REGISTRY[product_type]()
```

---

## 5. Step 3: Add to PRODUCT_REGISTRY

Register specific product configurations.

### Update `src/config/product_config.py`

```python
# Add to PRODUCT_REGISTRY dict
PRODUCT_REGISTRY: Dict[str, ProductConfig] = {
    # Existing RILA products
    "6Y20B": ProductConfig(
        name="FlexGuard 6Y20B",
        product_code="6Y20B",
        product_type="rila",
        buffer_level=0.20,
        term_years=6,
    ),
    # ... other existing products ...

    # NEW: Add your product
    "[PRODUCT_CODE]": ProductConfig(
        name="[Product Name]",
        product_code="[PRODUCT_CODE]",
        product_type="[product_type]",  # Must match methodology registration
        buffer_level=None,  # None for FIA/MYGA, float for RILA
        term_years=6,
        own_rate_column="[ColumnName]",  # Column name in WINK data
    ),
}
```

### ProductConfig Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | str | Yes | Human-readable product name |
| `product_code` | str | Yes | Short identifier for data/exports |
| `product_type` | str | Yes | Must match methodology registry |
| `buffer_level` | Optional[float] | RILA only | Buffer percentage (0.20 = 20%) |
| `term_years` | int | Yes | Product term in years |
| `own_rate_column` | str | Yes | WINK rate column name |
| `primary_index` | str | No | Index (default: "SP500") |
| `max_lag` | int | No | Max lag periods (default: 8) |
| `competitor_count` | int | No | Competitors to include (default: 7) |

---

## 6. Step 4: Configure Aggregation Strategy

Choose or implement an aggregation strategy for competitor rates.

### Available Strategies

| Strategy | Use Case | Description |
|----------|----------|-------------|
| `WeightedAggregation` | RILA | Market-share weighted competitor rates |
| `TopNAggregation` | FIA | Average of top-N competitors |
| `FirmLevelAggregation` | MYGA | Individual firm rate tracking |

### Selecting a Strategy

```python
from src.features.aggregation import get_strategy

# Get strategy by name
strategy = get_strategy("weighted", min_companies=3)
strategy = get_strategy("top_n", n=5)
strategy = get_strategy("firm_level")
```

### Creating a Custom Strategy

```python
# src/features/aggregation/custom_aggregation.py

from src.core.protocols import AggregationStrategy
import pandas as pd


class CustomAggregation(AggregationStrategy):
    """Custom aggregation strategy for [Product Type]."""

    def __init__(self, custom_param: float = 1.0):
        self.custom_param = custom_param

    def aggregate(
        self,
        competitive_rates: pd.DataFrame,
        date_column: str,
        rate_column: str,
    ) -> pd.DataFrame:
        """Aggregate competitor rates using custom logic.

        Parameters
        ----------
        competitive_rates : pd.DataFrame
            Raw competitor rate data
        date_column : str
            Date column name
        rate_column : str
            Rate column name

        Returns
        -------
        pd.DataFrame
            Aggregated rates indexed by date
        """
        # Implement aggregation logic
        pass
```

---

## 7. Step 5: Testing Requirements

### Minimum Test Coverage

| Test Category | Requirement | Location |
|---------------|-------------|----------|
| Unit Tests | 80% coverage | `tests/test_[product]_methodology.py` |
| Integration Tests | End-to-end pipeline | `tests/test_[product]_pipeline.py` |
| Baseline Tests | Mathematical equivalence | `tests/baselines/[product]/` |
| Leakage Tests | Advisory | `tests/test_leakage_prevention.py` |

### Test Template

```python
# tests/test_new_methodology.py

import pytest
from src.products.new_methodology import NewProductMethodology


class TestNewProductMethodology:
    """Tests for NewProductMethodology."""

    @pytest.fixture
    def methodology(self):
        return NewProductMethodology()

    def test_product_type(self, methodology):
        assert methodology.product_type == "[product_type]"

    def test_constraint_rules_not_empty(self, methodology):
        rules = methodology.get_constraint_rules()
        assert len(rules) > 0

    def test_coefficient_signs_defined(self, methodology):
        signs = methodology.get_coefficient_signs()
        assert "own_rate" in signs or "[pattern]" in signs

    def test_leakage_patterns_defined(self, methodology):
        patterns = methodology.get_leakage_patterns()
        assert len(patterns) > 0
        assert any("_t0" in p for p in patterns)
```

### Baseline Capture

```bash
# Capture baselines for new product
python scripts/capture_baselines.py \
    --product [PRODUCT_CODE] \
    --output tests/baselines/[product_type]/

# Verify equivalence
python scripts/equivalence_guard.py \
    --baseline tests/baselines/[product_type]/ \
    --current outputs/
```

---

## 8. Appendix A: MYGA Implementation Checklist

MYGA (Multi-Year Guaranteed Annuity) is currently a stub. Implementation requires:

### Domain Research

- [ ] Document MYGA pricing economics
- [ ] Identify key rate sensitivity factors
- [ ] Determine appropriate competitor definition
- [ ] Assess need for regime detection (rising vs. falling rates)

### Implementation Tasks

- [ ] Define `MYGAMethodology.get_constraint_rules()`
- [ ] Define `MYGAMethodology.get_coefficient_signs()`
- [ ] Choose/implement aggregation strategy
- [ ] Add MYGA products to `PRODUCT_REGISTRY`
- [ ] Create test fixtures
- [ ] Capture baseline outputs

### Current Status

```python
# src/products/myga_methodology.py

class MYGAMethodology:
    def get_constraint_rules(self) -> List[ConstraintRule]:
        raise NotImplementedError(
            "MYGA constraint rules not yet implemented. "
            "See docs/architecture/PRODUCT_EXTENSION_GUIDE.md"
        )
```

**Note**: MYGA currently raises `NotImplementedError` as a fail-fast guardrail.

---

## 9. Appendix B: Extension Points Reference

### Protocol Definitions

| Protocol | Location | Purpose |
|----------|----------|---------|
| `DataSourceAdapter` | `src/core/protocols.py` | Data loading abstraction |
| `AggregationStrategy` | `src/core/protocols.py` | Competitor rate aggregation |
| `ProductMethodology` | `src/core/protocols.py` | Economic constraint rules |

### Registry Locations

| Registry | Location | Purpose |
|----------|----------|---------|
| `PRODUCT_REGISTRY` | `src/config/product_config.py` | Product configurations |
| `_METHODOLOGY_REGISTRY` | `src/products/__init__.py` | Methodology classes |
| `_ADAPTER_REGISTRY` | `src/data/adapters/__init__.py` | Data adapters |
| `_STRATEGY_REGISTRY` | `src/features/aggregation/__init__.py` | Aggregation strategies |

### Configuration Fields

```python
@dataclass(frozen=True)
class ProductConfig:
    # Required
    name: str                    # "FlexGuard 6Y20B"
    product_code: str            # "6Y20B"
    product_type: str            # "rila", "fia", "myga"
    term_years: int              # 6

    # Product-type specific
    buffer_level: Optional[float] # 0.20 for RILA, None for FIA/MYGA

    # Rate configuration
    rate_column: str = "capRate"
    own_rate_prefix: str = "P"
    own_rate_column: str = "Prudential"
    competitor_rate_prefix: str = "C"

    # Feature engineering
    primary_index: str = "SP500"
    max_lag: int = 8
    competitor_count: int = 7
```

### WINK Product IDs

```python
# Add to src/config/product_config.py

_WINK_PIPELINE_IDS: Dict[str, Tuple[int, ...]] = {
    "Prudential": (2979,),
    "Allianz": (2162, 3699),
    # ... add company mappings for new products
}
```

---

## Questions?

- **Technical**: See `TECHNICAL_DEBT.md` for resolved issues
- **Architecture**: See `MODULE_HIERARCHY.md` for codebase structure
- **Migration**: See `docs/migration/V2_INTERFACE_WIRING.md` for interface changes
