# For New Developers (Comprehensive)

**Last Updated:** 2026-01-30
**Audience:** Developers new to annuity products or causal inference
**Time to Productive Work:** ~60 minutes with this guide

---

## Part 1: Why This Project Exists

### The Business Problem

Prudential sells RILA (Registered Index-Linked Annuity) products. A key pricing question:

> "If we increase our cap rate by 25 basis points, how much will sales increase?"

This question seems simple but is actually quite tricky because:
- We can't run A/B tests (regulatory and practical constraints)
- Historical data has confounding factors
- Correlation ≠ causation

### What is a RILA?

A RILA is a retirement product with:
- **Cap Rate**: Maximum return you can earn (e.g., 8%)
- **Buffer**: Protection against losses (e.g., 20% buffer means Prudential absorbs first 20% of losses)
- **Term**: Investment period (e.g., 6 years)

**Key Insight**: Cap rate IS the yield. Unlike bonds (where higher yield = lower price), for RILAs:
- Higher cap rate = more attractive product
- More attractive = higher sales
- Therefore: Own-rate elasticity should be **positive**

### Why Causal Inference?

Standard machine learning would just find patterns in data. But we need to answer:
- "What WOULD happen if we change rates?" (counterfactual)
- Not "What DOES the data show?" (correlation)

This requires causal econometric methods that properly handle:
- Temporal ordering (can't use future data)
- Confounding variables (market conditions affect everyone)
- Endogeneity (competitors respond to same signals)

---

## Part 2: Understanding the Code

### The Architecture Pattern

The codebase uses **Dependency Injection** to separate concerns:

```
┌─────────────────────────────────────────────────────────────┐
│ User Code (Notebooks)                                        │
│   interface = create_interface("6Y20B", environment="aws")   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ UnifiedNotebookInterface                                     │
│   - Coordinates all operations                              │
│   - Doesn't know WHERE data comes from                      │
│   - Doesn't know HOW to aggregate competitors               │
└─────────────────────────────┬───────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    DataSourceAdapter   AggregationStrategy   ProductMethodology
    (S3, Local, Fixture)  (Weighted, TopN)    (RILA, FIA)
```

**Why this pattern?**
- **Testing**: Can inject fake data for tests
- **Flexibility**: Can swap AWS for local without code changes
- **Clarity**: Each component has single responsibility

### Key Modules Explained

```
src/
├── core/
│   ├── protocols.py      # Interfaces (what adapters must implement)
│   ├── types.py          # TypedDict configurations
│   ├── exceptions.py     # Business-context errors
│   └── registry.py       # Product registry
│
├── notebooks/
│   └── interface.py      # UnifiedNotebookInterface
│
├── data/
│   ├── adapters/         # S3, Local, Fixture implementations
│   ├── extraction.py     # AWS data loading
│   └── preprocessing.py  # DataFrame transformations
│
├── features/
│   ├── engineering/      # Lag features, rolling averages
│   ├── aggregation/      # Competitor aggregation strategies
│   └── selection/        # Feature selection algorithms
│
├── products/
│   ├── rila.py           # RILA-specific constraints
│   └── fia.py            # FIA-specific constraints
│
└── models/
    └── inference.py      # Ridge regression with bootstrap
```

### Data Flow

```
Raw Sales Data (S3)
    │
    ▼
DataSourceAdapter.load_sales_data()
    │
    ▼
Preprocessing (date parsing, type conversion)
    │
    ▼
Feature Engineering (lags, rolling averages)
    │
    ▼
Competitor Aggregation (weighted by market share)
    │
    ▼
Model Training (Ridge regression)
    │
    ▼
Coefficient Validation (sign constraints)
    │
    ▼
Results (elasticity estimates)
```

---

## Part 3: The Five Critical Traps

These mistakes have caused significant rework. Learn from our pain.

### Trap 1: Lag-0 Competitor Features

**The Bug**: Including competitor rates at time t to predict sales at time t.

**Why It's Wrong**:
- Violates temporal ordering (cause must precede effect)
- Competitors might be responding to same market signals
- Creates "too good to be true" R² values

**The Tell**: R² suddenly jumps to >0.90

```python
# FORBIDDEN
features["competitor_rate_t0"] = df["competitor_rate"]

# CORRECT (minimum 2-week lag)
features["competitor_rate_t2"] = df["competitor_rate"].shift(2)
```

### Trap 2: Wrong Coefficient Signs

**The Bug**: Model shows negative own-rate coefficient.

**Why It's Wrong**: Economics says higher rates attract customers. Negative coefficient means your model is broken.

**Common Causes**:
- Data leakage (lag-0 features)
- Multicollinearity
- Coding errors (sign flip in data)

### Trap 3: Silent Failures

**The Bug**: Catching exceptions and returning fake data.

```python
# CATASTROPHICALLY WRONG
try:
    data = load_from_aws()
except:
    return pd.DataFrame()  # Silent failure!
```

**Why It's Wrong**: Analysis proceeds with garbage data, results are meaningless, nobody notices until production.

### Trap 4: Future Data in Features

**The Bug**: Using rolling averages calculated with future data.

```python
# WRONG - centered window uses future data
df["rolling_mean"] = df["value"].rolling(4, center=True).mean()

# CORRECT - trailing window only uses past
df["rolling_mean"] = df["value"].rolling(4).mean()
```

### Trap 5: Test Data Contamination

**The Bug**: Using random train/test split instead of temporal split.

```python
# WRONG - leaks future into training
train, test = train_test_split(df, test_size=0.2, random_state=42)

# CORRECT - test data strictly after training
train = df[df["date"] < "2024-01-01"]
test = df[df["date"] >= "2024-01-01"]
```

---

## Part 4: Setting Up Your Environment

### Step 1: Clone Repository

```bash
git clone <repo-url>
cd annuity-price-elasticity-v3
```

### Step 2: Create Conda Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate
conda activate annuity-price-elasticity-v3

# Verify
python --version  # Should be 3.10+
```

### Step 3: Verify Installation

```bash
# Quick import check
python -c "from src.notebooks import create_interface; print('OK')"

# Run pattern validator
make quick-check
```

### Step 4: Run Tests

```bash
# Fast smoke test
make quick-check

# Full test suite (takes ~5 minutes)
make test

# Specific test file
pytest tests/unit/data/test_extraction.py -v
```

---

## Part 5: Running Your First Analysis

### Using Fixture Data (Recommended for Learning)

Fixture data is pre-loaded test data that doesn't require AWS credentials.

```python
# In a Jupyter notebook or Python script
from src.notebooks import create_interface

# Create interface with fixture data
interface = create_interface(
    product_code="6Y20B",      # 6-year, 20% buffer RILA
    environment="fixture"      # Use local fixture data
)

# Load the data
df = interface.load_data()
print(f"Loaded {len(df)} observations")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Run inference
results = interface.run_inference(df)

# Examine results
print("\n=== Model Results ===")
print(f"R-squared: {results['metrics']['r_squared']:.2%}")
print(f"\nKey Coefficients:")
for coef, value in results['coefficients'].items():
    if 'prudential' in coef or 'competitor' in coef:
        print(f"  {coef}: {value:,.2f}")
```

### Interpreting Results

```
=== Model Results ===
R-squared: 62.3%

Key Coefficients:
  prudential_rate_t0: 8,500.00   # Positive ✓ (higher rate → more sales)
  prudential_rate_t1: 4,200.00   # Positive ✓
  competitor_weighted_t2: -6,300.00  # Negative ✓ (higher competitor → less sales)
```

**What the coefficients mean**:
- `prudential_rate_t0 = 8500`: A 1 percentage point increase in cap rate increases weekly sales by ~$8,500
- `competitor_weighted_t2 = -6300`: A 1 percentage point increase in competitor rates (lagged 2 weeks) decreases our sales by ~$6,300

---

## Part 6: Common Workflows

### Making a Code Change

1. **Find the right module**: Check `MODULE_HIERARCHY.md`
2. **Read existing tests**: Understand expected behavior
3. **Write your change**: Follow patterns in existing code
4. **Write tests first**: 100% coverage for new code
5. **Validate patterns**: `make quick-check`
6. **Run full tests**: `make test`
7. **If refactoring**: `make validate` (1e-12 precision)

### Debugging an Issue

1. **Check error message**: Our errors include business context
2. **Trace through DI**: Notebook → Interface → Adapter
3. **Check for anti-patterns**: `make pattern-check`
4. **Check for leakage**: `make leakage-audit`

### Deploying to Production

1. **All tests pass**: `make test`
2. **Leakage audit passes**: `make leakage-audit`
3. **Review checklist**: `knowledge/practices/LEAKAGE_CHECKLIST.md`
4. **Get sign-off**: Data scientist + reviewer

---

## Part 7: Testing Philosophy

### The 6-Layer Validation Architecture

```
Layer 6: Property-Based Tests (Hypothesis)
    "Coefficient signs should ALWAYS match theory"

Layer 5: End-to-End Tests
    "Full pipeline from raw data to inference"

Layer 4: Integration Tests
    "Multiple modules working together"

Layer 3: Unit Tests
    "Individual functions in isolation"

Layer 2: Input Validation
    "Fail-fast at function entry"

Layer 1: Type Safety
    "Catch errors at lint time"
```

### Key Test Commands

```bash
make quick-check      # Fast smoke test
make test            # Full suite
make test-unit       # Unit tests only
make test-leakage    # Leakage detection
make coverage        # Coverage report
```

---

## Part 8: Getting Help

### Documentation Hierarchy

| Priority | Document | When to Read |
|----------|----------|--------------|
| 1 | This file | First day |
| 2 | `CLAUDE.md` | Before writing code |
| 3 | `MODULE_HIERARCHY.md` | Finding where code lives |
| 4 | `knowledge/integration/LESSONS_LEARNED.md` | Before major changes |
| 5 | `knowledge/domain/RILA_ECONOMICS.md` | Understanding the business |

### Asking Questions

Before asking:
1. Check the documentation hierarchy above
2. Search codebase: `grep -r "your_topic" src/`
3. Check `.tracking/decisions.md` for past decisions

When asking:
- Include what you tried
- Include error messages
- Reference specific files/lines

---

## Part 9: Checklist for Your First Week

### Day 1
- [ ] Clone repo and set up environment
- [ ] Run `make quick-check` successfully
- [ ] Run first inference with fixture data
- [ ] Read this document completely

### Day 2-3
- [ ] Read `knowledge/domain/RILA_ECONOMICS.md`
- [ ] Understand the 5 critical traps
- [ ] Run full test suite: `make test`
- [ ] Explore test files to understand expected behavior

### Day 4-5
- [ ] Read `CLAUDE.md` and `MODULE_HIERARCHY.md`
- [ ] Make a small change and write tests for it
- [ ] Run `make leakage-audit` and understand what it checks
- [ ] Review `knowledge/practices/LEAKAGE_CHECKLIST.md`

### Week 2
- [ ] Understand the DI architecture deeply
- [ ] Review `.tracking/decisions.md` for context
- [ ] Pair with senior developer on a real task
- [ ] Present your understanding to team (solidifies learning)

---

## Glossary

| Term | Definition |
|------|------------|
| **RILA** | Registered Index-Linked Annuity |
| **Cap Rate** | Maximum return customer can earn |
| **Buffer** | Loss protection (e.g., 20% buffer = issuer absorbs first 20% of losses) |
| **Lag-0** | Using contemporaneous (same time period) data |
| **Elasticity** | % change in sales per % change in rate |
| **DI** | Dependency Injection (design pattern) |
| **Adapter** | Component that implements an interface for data access |
| **Leakage** | Using information that wouldn't be available at prediction time |

---

## Appendix: Useful Commands

```bash
# Environment
conda activate annuity-price-elasticity-v3
conda deactivate

# Testing
make quick-check       # 30-second smoke test
make test             # Full suite
make test-rila        # RILA tests only
make leakage-audit    # Pre-deployment gate

# Code Quality
make lint             # Run linters
make format           # Auto-format
make pattern-check    # Validate patterns

# Validation
make validate         # Mathematical equivalence (1e-12)
make stub-hunter      # Find placeholder code
make hardcode-scan    # Find hardcoded values
```
