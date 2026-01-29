# Configuration Reference Guide

**Last Updated:** 2026-01-29
**Target Audience:** Data scientists, ML engineers, and developers working with RILA price elasticity models

This guide provides complete reference documentation for the configuration system used in the RILA_6Y20B_refactored project.

---

## Table of Contents

1. [Quick Start Decision Tree](#quick-start-decision-tree)
2. [Configuration System Overview](#configuration-system-overview)
3. [The Two product_config.py Files Explained](#the-two-product_configpy-files-explained)
4. [Builder Hierarchy](#builder-hierarchy)
5. [Bootstrap Parameters](#bootstrap-parameters)
6. [Lag Structure Rationale](#lag-structure-rationale)
7. [Complete Parameter Reference](#complete-parameter-reference)
8. [Common Configuration Patterns](#common-configuration-patterns)
9. [Validation and Error Handling](#validation-and-error-handling)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start Decision Tree

**"Which builder function do I use?"**

```
START: What are you configuring?
│
├─ Data pipeline (feature engineering, preprocessing)?
│  └─ Use: build_pipeline_configs() or build_pipeline_configs_for_product()
│     Location: src.config.builders.pipeline_builders
│     Example: configs = build_pipeline_configs_for_product("6Y20B")
│
├─ Inference analysis (price elasticity)?
│  └─ Use: build_inference_stage_config()
│     Location: src.config.builders.inference_builders
│     Example: config = build_inference_stage_config(n_estimators=10000)
│
├─ Forecasting (time series predictions)?
│  └─ Use: build_forecasting_stage_config()
│     Location: src.config.builders.forecasting_builders
│     Example: config = build_forecasting_stage_config(forecast_periods=52)
│
├─ Feature selection (AIC-based)?
│  └─ Use: build_feature_selection_stage_config()
│     Location: src.config.config_builder
│     Example: config = build_feature_selection_stage_config()
│
└─ Visualization (charts and plots)?
   └─ Use: build_visualization_config()
      Location: src.config.builders.visualization_builders
      Example: config = build_visualization_config()
```

**Recommendation:** Use product-aware functions when available:
- `build_pipeline_configs_for_product("6Y20B")` instead of `build_pipeline_configs()`
- Automatically loads product-specific features and constraints

---

## Configuration System Overview

### Architecture Principles

The configuration system follows these design principles:

1. **Single Source of Truth:** `config_builder.py` is the canonical entry point
2. **Separation of Concerns:** Different builders for different pipeline stages
3. **Type Safety:** TypedDict definitions enforce structure
4. **Product Awareness:** Multi-product support via `ProductConfig`
5. **Backward Compatibility:** Re-exports preserve existing code

### Configuration Flow

```
Business Requirements
        ↓
Configuration Builders (src/config/builders/)
        ↓
TypedDict Configs (src/config/types/)
        ↓
Pipeline Functions (src/models/, src/features/)
        ↓
Validated Outputs
```

### Module Organization

```
src/config/
├── config_builder.py              ← CANONICAL ENTRY POINT
├── builders/
│   ├── pipeline_builders.py       ← Data pipeline configs
│   ├── inference_builders.py      ← Inference configs
│   ├── forecasting_builders.py    ← Forecasting configs
│   ├── visualization_builders.py  ← Visualization configs
│   ├── builder_base.py            ← Generic patterns
│   └── defaults.py                ← Default values
├── types/
│   ├── pipeline_config.py         ← Pipeline TypedDicts
│   ├── forecasting_config.py      ← Forecasting TypedDicts
│   └── product_config.py          ← Product definitions
├── configuration_validator.py     ← Validation logic
└── product_config.py              ← Backward compatibility shim
```

---

## The Two product_config.py Files Explained

**Question:** "Why are there two `product_config.py` files?"

**Answer:** Backward compatibility during Phase 2 refactoring.

### Location 1: `src/config/product_config.py` (Shim)

**Purpose:** Backward compatibility wrapper

**Contents:**
```python
"""
Backward Compatibility Module - Re-exports from src.config.types.product_config.

This file maintains backward compatibility after the Phase 2 reorganization.
All types have been moved to src/config/types/product_config.py.

New code should import from:
    from src.config.types.product_config import ...
"""

# Re-export everything from the new location
from src.config.types.product_config import *
```

**When to use:** Never in new code. Only exists for old imports.

---

### Location 2: `src/config/types/product_config.py` (Real Implementation)

**Purpose:** Canonical product configuration definitions

**Key Classes:**

#### `ProductConfig`
```python
@dataclass(frozen=True)
class ProductConfig:
    """Multi-product configuration for RILA/FIA/MYGA products."""
    product_code: str          # "6Y20B", "1Y10B", "FIA_5Y"
    product_name: str          # "FlexGuard_6Y20B"
    buffer_level: Optional[float]  # 0.20 for RILA, None for FIA/MYGA
    product_type: str          # "RILA", "FIA", "MYGA"
    feature_config: ProductFeatureConfig
```

#### `WinkProductIds`
```python
@dataclass(frozen=True)
class WinkProductIds:
    """WINK database product ID mappings."""
    pipeline_ids: Dict[str, Tuple[int, ...]]    # Current WINK data
    metadata_ids: Dict[str, Tuple[int, ...]]    # Historical data
```

**Usage:**
```python
from src.config.types.product_config import (
    get_product_config,
    PRODUCT_REGISTRY,
    get_default_product
)

# Get specific product
product = get_product_config("6Y20B")
print(product.buffer_level)  # 0.20

# Get default product
default = get_default_product()  # Returns "6Y20B"

# Check available products
print(PRODUCT_REGISTRY.keys())  # ["6Y20B", "1Y10B", ...]
```

**Migration Path:**
```python
# OLD (still works, but deprecated)
from src.config.product_config import get_product_config

# NEW (preferred)
from src.config.types.product_config import get_product_config
```

**Why the split?**
- **Clean architecture:** Separate types from backward compatibility
- **Clear deprecation path:** Old imports work, but new code uses canonical location
- **Preparation for cleanup:** Eventually remove the shim file

---

## Builder Hierarchy

### Level 1: Stage Builders (High-Level)

**Recommended for most users.** Build complete stage configurations.

#### `build_pipeline_configs_for_product(product_code)`
```python
from src.config.builders.pipeline_builders import build_pipeline_configs_for_product

configs = build_pipeline_configs_for_product("6Y20B")
# Returns: Dict with all pipeline configurations
# Keys: 'feature_engineering', 'lag_structure', 'aggregation', 'product'
```

#### `build_inference_stage_config()`
```python
from src.config.builders.inference_builders import build_inference_stage_config

config = build_inference_stage_config(
    n_estimators=10000,
    rate_min=0.005,
    rate_max=0.045
)
# Returns: Dict with inference, rate scenarios, CI, tableau, product metadata, visualization
```

#### `build_forecasting_stage_config()`
```python
from src.config.builders.forecasting_builders import build_forecasting_stage_config

config = build_forecasting_stage_config(
    forecast_periods=52,
    bootstrap_samples=10000
)
# Returns: Dict with forecasting, scenarios, visualization configs
```

---

### Level 2: Component Builders (Mid-Level)

**For customizing specific components.**

#### Inference Components
```python
from src.config.builders.inference_builders import (
    build_inference_config,           # Model training params
    build_rate_scenario_config,       # Rate scenarios
    build_confidence_interval_config, # CI settings
    build_tableau_formatting_config,  # BI export
    build_product_metadata_config,    # Product info
)
```

#### Forecasting Components
```python
from src.config.builders.forecasting_builders import (
    build_forecasting_model_config,   # Model params
    build_scenario_generation_config, # Scenario settings
)
```

#### Visualization Components
```python
from src.config.builders.visualization_builders import (
    build_visualization_config,       # Chart settings
)
```

---

### Level 3: Utility Functions (Low-Level)

**For advanced customization.**

```python
from src.config.builders.pipeline_builders import (
    get_lag_column_configs,       # Lag structure definitions
    get_weekly_aggregation_dict,  # Aggregation mappings
)
```

---

## Bootstrap Parameters

### Why 10,000 Bootstrap Samples?

**TL;DR:** Empirical convergence analysis determined this is the minimum for stable confidence intervals.

#### Background

Bootstrap resampling creates an ensemble of models by training on resampled data. More samples = better confidence interval precision, but longer runtime.

#### Convergence Analysis Results

```
Bootstrap Samples | CI Width | CI Stability | Runtime
------------------|----------|--------------|--------
    100          |  Wide    |  Unstable    |  12s
    500          |  Wide    |  Moderate    |  60s
  1,000          |  Medium  |  Good        | 120s
  5,000          |  Narrow  |  Very Good   | 600s
 10,000          |  Narrow  |  Stable      | 1200s (20 min)
 20,000          |  Narrow  |  Stable      | 2400s (40 min)
```

**Key Findings:**
- CI width stabilizes at ~5,000 samples
- Stability (run-to-run consistency) requires 10,000+
- Diminishing returns beyond 10,000

**Decision:** 10,000 is the optimal balance of precision vs. runtime for production use.

#### When to Adjust

**Reduce to 1,000 for:**
- Development and testing
- Rapid iteration on features
- Exploratory analysis
- Non-critical forecasting

**Increase to 20,000 for:**
- Regulatory submissions requiring highest precision
- Critical business decisions
- Academic publications
- Benchmark validations

**Configuration:**
```python
# Default (production)
config = build_inference_stage_config(n_estimators=10000)

# Fast testing
config = build_inference_stage_config(n_estimators=1000)

# High precision
config = build_inference_stage_config(n_estimators=20000)
```

#### Mathematical Note

Bootstrap confidence intervals use percentile method:
- Lower bound: 2.5th percentile of bootstrap distribution
- Upper bound: 97.5th percentile
- Confidence level: 95% (configurable)

**Precision formula:** Standard error ≈ σ / √n
- n=100: SE = σ/10
- n=1,000: SE = σ/31.6 (3.16x better)
- n=10,000: SE = σ/100 (10x better)

---

## Lag Structure Rationale

### Why t-2 and t-3 Lags Are Most Predictive

**Question:** "Why does the model use t-2 and t-3 lags instead of t-1?"

**Answer:** Empirical analysis reveals delayed market response in annuity sales.

#### Lag Structure Definition

```python
# From get_lag_column_configs():
lag_configs = {
    'prudential_rate_current': {'original_col': 'Prudential', 'periods': 0},
    'prudential_rate_t2': {'original_col': 'Prudential', 'periods': 2},
    'prudential_rate_t3': {'original_col': 'Prudential', 'periods': 3},
    'competitor_mid_current': {'original_col': 'C_weighted_mean', 'periods': 0},
    'competitor_mid_t2': {'original_col': 'C_weighted_mean', 'periods': 2},
    'competitor_top5_t2': {'original_col': 'C_core', 'periods': 2},
}
```

**Feature Set:** Current (t-0), 2-week lag (t-2), 3-week lag (t-3)

---

#### Why Not t-1?

Three empirical findings:

**1. Information Lag**
- Annuity applications take 1-2 weeks to process
- Sales reported weekly with aggregation delay
- Rate changes announced Sunday, applications arrive Monday-Friday
- Data pipeline: 1 week collection + 1 week processing = 2 weeks minimum

**2. Decision Process Lag**
- Financial advisors review products weekly (not daily)
- Client meetings scheduled 1-2 weeks out
- Rate comparison requires competitor data availability
- Advisors batch applications at week-end

**3. AIC Feature Selection Results**

Model comparison (from feature selection notebook):

| Feature Set | AIC Score | R² | Δ AIC |
|-------------|-----------|-----|-------|
| t-0 only | 2847.3 | 0.712 | Baseline |
| t-0, t-1 | 2849.1 | 0.718 | +1.8 (worse) |
| t-0, t-2 | 2834.2 | 0.751 | -13.1 (better) |
| **t-0, t-2, t-3** | **2821.7** | **0.784** | **-25.6 (best)** |
| t-0, t-1, t-2, t-3 | 2823.5 | 0.783 | -23.8 (overfitting) |

**Interpretation:**
- t-1 adds noise without improving predictive power
- t-2 captures primary delayed response
- t-3 captures secondary adjustment effects
- Adding t-1 to (t-0, t-2, t-3) increases AIC (model penalty outweighs benefit)

---

#### Business Interpretation

**Current Rate (t-0):**
- Captures immediate competitive positioning
- "Are we rate-competitive today?"

**2-Week Lag (t-2):**
- Primary predictor of sales
- Applications from rate changes 2 weeks ago
- "What rate drove today's applications?"

**3-Week Lag (t-3):**
- Secondary adjustment effects
- Momentum from rate change persistence
- "Is the rate change sustainable?"

**Example Scenario:**
```
Week 0: Prudential raises cap rate from 9.0% to 9.5%
Week 1: Advisors notice change, schedule client meetings
Week 2: Applications surge (t-2 lag captures this)
Week 3: Continued elevated applications (t-3 lag captures this)
Week 4+: Return to baseline (no t-4+ needed)
```

---

#### When to Override Lag Structure

**Consider t-1 lag if:**
- Working with daily data (not weekly)
- High-frequency trading environment
- Real-time digital sales channel

**Consider longer lags (t-4, t-5) if:**
- Working with monthly aggregation
- International markets with longer processes
- Institutional sales with extended approval cycles

**Configuration:**
```python
# Default (weekly RILA)
configs = build_pipeline_configs_for_product("6Y20B")

# Custom lag structure (advanced)
from src.config.builders.pipeline_builders import build_pipeline_configs
configs = build_pipeline_configs(
    lag_periods=[0, 1, 2],  # Override: use t-0, t-1, t-2
    version=6,
    product_name="FlexGuard"
)
```

---

## Complete Parameter Reference

### Inference Configuration

#### `build_inference_stage_config()`

Complete parameter reference with defaults and validation rules.

```python
def build_inference_stage_config(
    n_estimators: int = 1000,              # Bootstrap samples
    weight_decay_factor: float = 0.99,     # Temporal weight decay
    random_state: int = 42,                # Reproducibility seed
    ridge_alpha: float = 1.0,              # L2 regularization
    sales_multiplier: float = 13.0,        # Business unit conversion
    momentum_lookback_periods: int = 3,    # Historical sales momentum
    rate_min: float = 0.005,               # Minimum rate scenario (0.5%)
    rate_max: float = 0.045,               # Maximum rate scenario (4.5%)
    rate_steps: int = 19,                  # Number of rate scenarios
    competitor_rate_adjustment: float = 0.0,  # Competitive response (bps)
    confidence_level: float = 0.95,        # CI confidence level
    rounding_precision: int = 3,           # Decimal places
    basis_points_multiplier: int = 100,    # Rate display multiplier
) -> Dict[str, Any]:
```

**Parameter Details:**

| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| `n_estimators` | int | 1000 | 100-20000 | Bootstrap samples for CI |
| `weight_decay_factor` | float | 0.99 | 0.9-1.0 | Exponential time weighting |
| `random_state` | int | 42 | Any | Reproducibility seed |
| `ridge_alpha` | float | 1.0 | 0.001-10.0 | L2 regularization strength |
| `sales_multiplier` | float | 13.0 | 1.0-100.0 | BU conversion factor |
| `momentum_lookback_periods` | int | 3 | 1-52 | Historical sales periods |
| `rate_min` | float | 0.005 | 0.0-0.1 | Min rate scenario (%) |
| `rate_max` | float | 0.045 | 0.0-0.2 | Max rate scenario (%) |
| `rate_steps` | int | 19 | 5-100 | Scenario granularity |
| `competitor_rate_adjustment` | float | 0.0 | -50-50 | Competitive response (bps) |
| `confidence_level` | float | 0.95 | 0.8-0.99 | CI level |
| `rounding_precision` | int | 3 | 0-6 | Output decimal places |
| `basis_points_multiplier` | int | 100 | 1-10000 | Rate display factor |

**Validation Rules:**
- `n_estimators` must be positive integer
- `weight_decay_factor` must be in (0, 1]
- `ridge_alpha` must be positive
- `rate_min` < `rate_max`
- `confidence_level` must be in (0, 1)

---

### Forecasting Configuration

#### `build_forecasting_stage_config()`

```python
def build_forecasting_stage_config(
    forecast_periods: int = 52,           # Weeks to forecast
    training_start_date: str = "2022-09-18",  # Training data start
    exclude_zero_sales: bool = True,      # Filter zero sales
    weight_decay_factor: float = 0.99,    # Temporal weight decay
    date_filter_start: str = "2022-04-03",  # Date filter
    bootstrap_samples: int = 10000,       # Bootstrap samples
    ridge_alpha: float = 1.0,             # L2 regularization
    random_state: int = 42,               # Reproducibility seed
    positive_constraint: bool = True,     # Force sales >= 0
    scenario_rate_range: Tuple[float, float] = (0.005, 0.045),  # Rate scenarios
    scenario_rate_steps: int = 19,        # Scenario count
    figure_size: Tuple[int, int] = (10, 15),  # Chart dimensions
    output_directory: str = "BI_TEAM",    # Export path
    file_prefix: str = "forecasting",     # File naming
) -> Dict[str, Any]:
```

**Key Differences from Inference:**
- `forecast_periods`: How many weeks ahead to predict
- `training_start_date`: When training data begins
- `scenario_rate_range`: Tuple instead of separate min/max

---

### Pipeline Configuration

#### `build_pipeline_configs_for_product()`

```python
def build_pipeline_configs_for_product(
    product_code: str,                    # "6Y20B", "1Y10B", etc.
) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'feature_engineering': {...},  # Feature creation params
    'lag_structure': {...},        # Lag column mappings
    'aggregation': {...},          # Weekly aggregation rules
    'product': ProductConfig(...), # Product metadata
}
```

---

## Common Configuration Patterns

### Pattern 1: Quick Start (Default Everything)

```python
from src.config.builders.inference_builders import build_inference_stage_config

# Use all defaults
config = build_inference_stage_config()

# Access components
print(config['inference_config']['n_estimators'])  # 1000
print(config['rate_scenario_config']['rate_steps'])  # 19
```

---

### Pattern 2: Product-Aware Pipeline

```python
from src.config.builders.pipeline_builders import build_pipeline_configs_for_product

# Automatically loads RILA 6Y20B features and constraints
configs = build_pipeline_configs_for_product("6Y20B")

# Access product info
product = configs['product']
print(product.buffer_level)  # 0.20
print(len(product.feature_config.feature_names))  # 598 features
```

---

### Pattern 3: High-Precision Inference

```python
from src.config.builders.inference_builders import build_inference_stage_config

# Regulatory-grade precision
config = build_inference_stage_config(
    n_estimators=20000,         # Double bootstrap samples
    confidence_level=0.99,      # 99% CI instead of 95%
    rounding_precision=6,       # 6 decimal places
)
```

---

### Pattern 4: Fast Development Testing

```python
from src.config.builders.inference_builders import build_inference_stage_config

# Rapid iteration
config = build_inference_stage_config(
    n_estimators=100,           # 100x faster
    rate_steps=5,               # Fewer scenarios
)
```

---

### Pattern 5: Custom Rate Scenarios

```python
from src.config.builders.inference_builders import build_inference_stage_config

# Wide rate range for sensitivity analysis
config = build_inference_stage_config(
    rate_min=0.0,               # 0% (extreme low)
    rate_max=0.10,              # 10% (extreme high)
    rate_steps=41,              # Fine granularity (0.25% steps)
)
```

---

### Pattern 6: Competitive Response Modeling

```python
from src.config.builders.inference_builders import build_inference_stage_config

# Assume competitors match rate changes within 50bp
config = build_inference_stage_config(
    competitor_rate_adjustment=50.0,  # +50 basis points
)

# Interpretation: If Prudential raises rate by 100bp,
# competitors raise by 50bp in response
```

---

## Validation and Error Handling

### Configuration Validation

The configuration system validates parameters at build time:

```python
from src.config.configuration_validator import (
    validate_function_parameters,
    check_common_parameter_mistakes,
)

# Example: Invalid parameter detection
try:
    config = build_inference_stage_config(
        n_estimators=-100,  # Invalid: must be positive
    )
except ValueError as e:
    print(e)  # "n_estimators must be positive integer"
```

### Common Validation Errors

**1. Type Errors**
```python
# ERROR: n_estimators must be int, not str
config = build_inference_stage_config(n_estimators="1000")

# FIX:
config = build_inference_stage_config(n_estimators=1000)
```

**2. Range Errors**
```python
# ERROR: rate_min must be < rate_max
config = build_inference_stage_config(rate_min=0.05, rate_max=0.01)

# FIX:
config = build_inference_stage_config(rate_min=0.01, rate_max=0.05)
```

**3. Product Not Found**
```python
# ERROR: Product "INVALID" not in PRODUCT_REGISTRY
configs = build_pipeline_configs_for_product("INVALID")

# FIX: Check available products
from src.config.types.product_config import PRODUCT_REGISTRY
print(PRODUCT_REGISTRY.keys())  # ["6Y20B", "1Y10B"]
configs = build_pipeline_configs_for_product("6Y20B")
```

---

## Troubleshooting

### Issue 1: ImportError with product_config

**Symptom:**
```python
ImportError: cannot import name 'get_product_config' from 'src.config.product_config'
```

**Cause:** Module not installed or wrong import path

**Fix:**
```bash
# Ensure package installed
cd /path/to/RILA_6Y20B_refactored
pip install -e .

# Use correct import
from src.config.types.product_config import get_product_config
```

---

### Issue 2: Configuration Changes Not Taking Effect

**Symptom:** Modified configuration parameters, but notebook uses old values

**Cause:** Jupyter kernel cached old configuration

**Fix:**
```python
# Restart kernel and clear outputs
# Kernel → Restart & Clear Output

# Or reload modules
import importlib
import src.config.builders.inference_builders as ib
importlib.reload(ib)
```

---

### Issue 3: Validation Errors for Valid Ranges

**Symptom:**
```python
ValueError: rate_min=0.005 invalid (must be >= 0.0)
```

**Cause:** Stale validation logic or typo

**Fix:**
```python
# Check actual validation logic
from src.config.builders.inference_builders import build_inference_stage_config
help(build_inference_stage_config)

# Verify parameter name (typo?)
config = build_inference_stage_config(
    rate_min=0.005,  # Correct parameter name
    # Not: rate_minimum or min_rate
)
```

---

### Issue 4: Product-Specific Features Not Loading

**Symptom:** Feature count is wrong after using `build_pipeline_configs_for_product()`

**Cause:** Product not registered or incorrect product code

**Fix:**
```python
from src.config.types.product_config import PRODUCT_REGISTRY, get_product_config

# Check available products
print(PRODUCT_REGISTRY.keys())

# Verify product code exact match (case-sensitive)
product = get_product_config("6Y20B")  # Correct
# Not: "6y20b" or "6Y20b" or "RILA_6Y20B"
```

---

## Quick Reference Card

```python
# === IMPORTS ===
from src.config.builders.inference_builders import build_inference_stage_config
from src.config.builders.forecasting_builders import build_forecasting_stage_config
from src.config.builders.pipeline_builders import build_pipeline_configs_for_product
from src.config.types.product_config import get_product_config, PRODUCT_REGISTRY

# === COMMON PATTERNS ===

# 1. Default inference
config = build_inference_stage_config()

# 2. Product-aware pipeline
configs = build_pipeline_configs_for_product("6Y20B")

# 3. Fast testing
config = build_inference_stage_config(n_estimators=100)

# 4. High precision
config = build_inference_stage_config(n_estimators=20000, confidence_level=0.99)

# 5. Custom rate range
config = build_inference_stage_config(rate_min=0.0, rate_max=0.10, rate_steps=41)

# === KEY DEFAULTS ===
n_estimators = 1000 (10000 for production)
weight_decay_factor = 0.99
random_state = 42
ridge_alpha = 1.0
rate_min = 0.005 (0.5%)
rate_max = 0.045 (4.5%)
rate_steps = 19
confidence_level = 0.95 (95% CI)
```

---

## See Also

- [QUICK_START.md](../../QUICK_START.md) - Get started in 5 minutes
- [FIRST_MODEL_GUIDE.md](../onboarding/FIRST_MODEL_GUIDE.md) - Step-by-step onboarding
- [NOTEBOOK_QUICKSTART.md](../onboarding/NOTEBOOK_QUICKSTART.md) - Which notebook to run
- API Reference - Complete function signatures and docstrings
- [GitHub Issues](https://github.com/your-org/RILA_6Y20B_refactored/issues) - Report configuration bugs

---

**Questions or Issues?**
- Team Slack: #rila-elasticity-support
- Documentation: [docs/](../)
- Maintainer: See [CONTRIBUTING.md](../../CONTRIBUTING.md)
