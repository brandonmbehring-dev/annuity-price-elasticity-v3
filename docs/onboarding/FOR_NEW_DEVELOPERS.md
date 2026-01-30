# For New Developers (Technical)

**Last Updated:** 2026-01-30
**Audience:** Developers with Python and statistics background
**Time to First Inference:** ~30 minutes

---

## Why This Project Exists

**Business Question:** "If we increase cap rates by 25bp, how much will sales change?"

This is a **causal question**, not a prediction question. We use econometric methods to estimate price elasticity of demand for RILA (Registered Index-Linked Annuity) products.

**Key Insight:** Cap rate IS the yield for these products. Higher cap rates attract more customers (positive elasticity). This differs from bonds where higher yield = lower price.

---

## Quick Start (5 minutes)

```bash
# 1. Clone and setup
git clone <repo>
cd annuity-price-elasticity-v3

# 2. Create environment
conda env create -f environment.yml
conda activate annuity-price-elasticity-v3

# 3. Verify installation
python -c "from src.notebooks import create_interface; print('OK')"

# 4. Run tests to confirm everything works
make quick-check
```

---

## Architecture Overview (10 minutes)

### Dependency Injection Pattern

```
Notebook → UnifiedNotebookInterface → Adapters/Strategies → Results
                    │
                    ├── DataSourceAdapter (S3 | Local | Fixture)
                    ├── AggregationStrategy (Weighted | TopN | FirmLevel)
                    └── ProductMethodology (RILA | FIA)
```

### Key Entry Point

```python
from src.notebooks import create_interface

# Development (no AWS needed)
interface = create_interface("6Y20B", environment="fixture")

# Production
interface = create_interface("6Y20B", environment="aws",
                            adapter_kwargs={"config": aws_config})

# Usage
df = interface.load_data()
results = interface.run_inference(df)
```

### Directory Structure

```
src/
├── core/           # Protocols, types, exceptions
├── notebooks/      # UnifiedNotebookInterface
├── data/adapters/  # S3, Local, Fixture adapters
├── features/       # Engineering, aggregation, selection
├── products/       # RILA/FIA methodologies
├── models/         # Inference models
└── validation/     # Validators
```

---

## Critical Traps [WARN]

**READ THIS BEFORE WRITING CODE**

### 1. No Lag-0 Competitor Features

```python
# FORBIDDEN - violates causal identification
competitor_rate_t0 = df["competitor_rate"]

# CORRECT - use lagged values
competitor_rate_t2 = df["competitor_rate"].shift(2)
```

### 2. Own Rate Coefficient Must Be Positive

Higher cap rates attract customers. If your model shows negative own-rate elasticity, you have a bug (likely data leakage).

### 3. Never Fail Silently

```python
# WRONG
try:
    data = load_aws_data()
except:
    return synthetic_data()  # Silently corrupt

# CORRECT
try:
    data = load_aws_data()
except AWSError as e:
    raise DataLoadError(
        "Cannot load data",
        business_impact="Analysis invalid",
        required_action="Check AWS credentials"
    ) from e
```

### 4. Mathematical Equivalence at 1e-12

When refactoring, outputs must match at 1e-12 precision. Use:
```bash
make validate
```

### 5. Always Run Leakage Audit Before Deployment

```bash
make leakage-audit
```

---

## Running Your First Model (15 minutes)

### Option A: Using Fixtures (Recommended for Development)

```python
from src.notebooks import create_interface

# Create interface with fixture data
interface = create_interface("6Y20B", environment="fixture")

# Load data
df = interface.load_data()
print(f"Loaded {len(df)} rows")

# Run inference
results = interface.run_inference(df)

# Check results
print(f"R²: {results['metrics']['r_squared']:.2%}")
print(f"Own rate coefficient: {results['coefficients']['prudential_rate_t0']:.2f}")
```

### Option B: Using AWS (Production)

```python
from src.notebooks import create_interface
from src.config import aws_config

interface = create_interface("6Y20B", environment="aws",
                            adapter_kwargs={"config": aws_config})
# ... same as above
```

---

## Testing

```bash
# Quick smoke test (~30s)
make quick-check

# Full test suite (~5 min)
make test

# Leakage detection (MANDATORY before deployment)
make leakage-audit

# Coverage report
make coverage
```

---

## Code Quality Standards

| Requirement | Standard |
|-------------|----------|
| Function length | 30-50 lines max |
| Type hints | Mandatory |
| Tests | 100% coverage for new code |
| Error handling | Fail-fast, never silent |
| Precision | 1e-12 for refactoring |

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | AI assistant guidance |
| `MODULE_HIERARCHY.md` | Complete architecture |
| `knowledge/practices/LEAKAGE_CHECKLIST.md` | Pre-deployment gate |
| `knowledge/integration/LESSONS_LEARNED.md` | 5 critical traps |

---

## Getting Help

1. Check `knowledge/domain/` for product economics
2. Check `.tracking/decisions.md` for WHY decisions were made
3. Run `make quick-check` to validate patterns
4. Ask in Slack #annuity-elasticity

---

## Next Steps

1. [DONE] Environment setup
2. [DONE] Run first inference
3. [ ] Read `knowledge/domain/RILA_ECONOMICS.md`
4. [ ] Review `knowledge/practices/LEAKAGE_CHECKLIST.md`
5. [ ] Run `make test` and understand test structure
