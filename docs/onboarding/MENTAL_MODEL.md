# Mental Model: How It All Fits Together

**The big picture of RILA price elasticity modeling.**

This document explains the system architecture in plain English—no code required. Read this before diving into the codebase.

---

## The Business Question

> **"How does Prudential's cap rate affect RILA sales, controlling for competitor rates and market conditions?"**

This is a **causal** question, not a prediction question. We want to know:
- If we raise our cap rate by 50 basis points, how much do sales increase?
- What's the **effect** of our pricing decisions?

---

## The Data Story

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW (30,000 ft view)                        │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌────────────────┐                 ┌────────────────┐
    │   TDE (Sales)  │                 │  WINK (Rates)  │
    │                │                 │                │
    │ Weekly premium │                 │ Daily cap rates│
    │ by product     │                 │ by company     │
    └───────┬────────┘                 └───────┬────────┘
            │                                  │
            │         ┌─────────────┐          │
            └────────►│   MERGE     │◄─────────┘
                      │ (by week)   │
                      └──────┬──────┘
                             │
                             ▼
                      ┌─────────────┐
                      │  FEATURE    │
                      │ ENGINEERING │
                      │             │
                      │ • Lag features (t-1, t-2, ...)     │
                      │ • Competitor aggregation           │
                      │ • Seasonality indicators           │
                      └──────┬──────┘
                             │
                             ▼
                      ┌─────────────┐
                      │  VALIDATE   │
                      │ (No lag-0!) │
                      │             │
                      │ Check for simultaneity violations  │
                      └──────┬──────┘
                             │
                             ▼
                      ┌─────────────┐
                      │  INFERENCE  │
                      │ (OLS model) │
                      │             │
                      │ Constrained regression             │
                      │ P_coef > 0, C_coef < 0            │
                      └──────┬──────┘
                             │
                             ▼
                      ┌─────────────┐
                      │   RESULTS   │
                      │             │
                      │ Elasticity = X.XX                  │
                      │ CI: [lower, upper]                 │
                      └─────────────┘
```

---

## Key Abstractions

### 1. The Product System

Each product type (RILA, FIA, MYGA) has different economic rules.

```
Product Types:
├── RILA (Registered Index-Linked Annuity)
│   ├── Has buffers (10%, 20%, etc.)
│   ├── Own rate: POSITIVE coefficient
│   ├── Competitor rates: NEGATIVE coefficient
│   └── Aggregation: Market-share weighted
│
├── FIA (Fixed Indexed Annuity)
│   ├── Has floors (0% = full protection)
│   ├── Same coefficient signs as RILA
│   └── Aggregation: Top-N mean
│
└── MYGA (Multi-Year Guaranteed Annuity)
    ├── Fixed rate (like a CD)
    ├── Simpler product
    └── Different elasticity magnitude
```

### 2. The Dependency Injection Pattern

The codebase uses "dependency injection" (DI) for flexibility. Think of it as pluggable components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    UnifiedNotebookInterface                      │
│                                                                  │
│   "I orchestrate the analysis. Give me components and I'll      │
│    wire them together."                                          │
│                                                                  │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│   │  DataAdapter    │  │   Aggregation   │  │   Methodology   │ │
│   │  (where data?)  │  │ (how combine?)  │  │ (what rules?)   │ │
│   │                 │  │                 │  │                 │ │
│   │ • AWS (prod)    │  │ • Weighted      │  │ • RILA rules    │ │
│   │ • Local (dev)   │  │ • Top-N         │  │ • FIA rules     │ │
│   │ • Fixture (test)│  │ • Firm-level    │  │ • MYGA rules    │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Why DI matters**:
- **Testing**: Swap real AWS data for fixtures without changing analysis code
- **Development**: Work offline with local data
- **Extension**: Add new products by implementing new components

### 3. The Leakage Prevention System

Data leakage is the biggest technical risk. The system has multiple guards:

```
LEAKAGE GUARDS
═════════════════════════════════════════════════════════════

1. Feature Validation
   └── Rejects any feature matching "competitor.*_t0" or "_current"

2. Methodology Constraints
   └── Product-specific rules enforced at inference time

3. Automated Gates
   └── CI pipeline checks for leakage patterns

4. Human Checklist
   └── Pre-deployment LEAKAGE_CHECKLIST.md review
```

---

## The Causal Identification Strategy

### Why Lag-0 Competitors Are Forbidden

```
THE SIMULTANEITY PROBLEM
════════════════════════════════════════════════════════════

        Market Conditions (volatility, rates, sentiment)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Competitor Rates           Our Sales
       (at time t)                (at time t)

If both are driven by the same market conditions,
using C_t to predict Sales_t captures spurious correlation,
not causal effect.

SOLUTION: Use C_{t-1} (lagged competitor rates)
═════════════════════════════════════════════════════════════

       Competitor Rates           Market Conditions
       (at time t-1)              (at time t)
              │                         │
              │    ┌────────────────────┘
              ▼    ▼
           Our Sales (at time t)

Now C_{t-1} precedes Sales_t in time.
Competitor rates from LAST week can't be caused by THIS week's conditions.
```

### Why Own Rate (P_lag_0) IS Allowed

```
WE CONTROL OUR OWN RATE
═════════════════════════════════════════════════════════════

  VNB Targets    Option Costs    Earned Rates
       │              │              │
       └──────────────┼──────────────┘
                      ▼
             Prudential Rate (P_t)
                      │
                      │    (we set this BEFORE seeing sales)
                      ▼
              Our Sales (at time t)

The rate is set BEFORE we observe application-date sales.
Even if we later adjust based on sales, there's a lag:
- We see contract-issue-date sales (19-76 days after application)
- This creates an "identification window"
```

---

## Directory Map

```
annuity-price-elasticity-v3/
│
├── src/                          # Source code
│   ├── core/                     # Foundational abstractions
│   │   ├── protocols.py          # Interface definitions (what components must do)
│   │   ├── types.py              # Type definitions (data structures)
│   │   ├── exceptions.py         # Custom exceptions with business context
│   │   └── registry.py           # Component registration
│   │
│   ├── notebooks/                # Entry point for analysis
│   │   └── interface.py          # UnifiedNotebookInterface (start here!)
│   │
│   ├── data/                     # Data layer
│   │   ├── adapters/             # S3, Local, Fixture data sources
│   │   ├── extraction.py         # AWS S3 loading
│   │   └── preprocessing.py      # DataFrame transformations
│   │
│   ├── features/                 # Feature engineering
│   │   ├── aggregation/          # Competitor aggregation strategies
│   │   └── selection/            # Feature selection algorithms
│   │
│   ├── products/                 # Product-specific logic
│   │   ├── rila_methodology.py   # RILA constraint rules
│   │   └── fia_methodology.py    # FIA constraint rules
│   │
│   ├── models/                   # Inference models
│   │   └── inference_scenarios.py # OLS with constraints
│   │
│   └── validation/               # Validation utilities
│
├── knowledge/                    # Domain documentation
│   ├── domain/                   # Product economics
│   │   ├── RILA_ECONOMICS.md     #  Start here for product understanding
│   │   └── GLOSSARY.md           # Term definitions
│   ├── analysis/                 # Modeling framework
│   │   └── CAUSAL_FRAMEWORK.md   #  Start here for causal understanding
│   └── practices/                # Best practices
│       └── LEAKAGE_CHECKLIST.md  # Pre-deployment validation
│
├── docs/                         # Architecture documentation
│   ├── onboarding/               # You are here
│   └── architecture/             # Design decisions
│
├── notebooks/                    # Jupyter notebooks
│   ├── rila/                     # RILA analysis notebooks
│   ├── fia/                      # FIA analysis notebooks
│   └── onboarding/               # Tutorial notebooks
│
└── tests/                        # Test suite
    ├── fixtures/                 # Test data (74MB RILA, 14MB FIA)
    └── baselines/                # Reference outputs for equivalence testing
```

---

## How Components Connect

### Running Inference (Simplified)

```python
# 1. Create interface (picks components based on product)
interface = create_interface("6Y20B", environment="fixture")
#                              │              │
#                              │              └── DataAdapter: FixtureAdapter
#                              └── ProductConfig: RILA 6-year 20% buffer
#                                   → Methodology: RILAMethodology
#                                   → Aggregation: WeightedAggregation

# 2. Load data (adapter handles source)
df = interface.load_data()
#     └── FixtureAdapter reads tests/fixtures/rila/*.parquet

# 3. Run inference (methodology enforces constraints)
results = interface.run_inference(df)
#          └── Validates: no lag-0 competitors
#          └── Runs: constrained OLS
#          └── Returns: coefficients, confidence intervals

# 4. Validate results (check economic constraints)
validation = interface.validate_coefficients(results["coefficients"])
#             └── P_coef > 0? [PASS]
#             └── C_coef < 0? [PASS]
```

### Adding a New Product

```
To add a new product variant (e.g., 6Y15B):

1. Add config in src/config/product_registry.py
   └── ProductConfig(code="6Y15B", buffer=0.15, ...)

2. Create fixtures in tests/fixtures/rila/6Y15B/
   └── sales.parquet, rates.parquet

3. (Optional) If new methodology needed, implement in src/products/
   └── Usually RILA variants share RILAMethodology

4. Run tests
   └── pytest tests/test_multiproduct_equivalence.py -v
```

---

## Common Mental Traps

### Trap 1: "Higher Price = Lower Demand"

**Wrong**: Applying traditional price elasticity intuition.

**Right**: Cap rate is a **yield**. Higher yield = more attractive = more sales.

```
Traditional:    Price ↑  →  Demand ↓  →  Negative coefficient
RILA:           Cap ↑    →  Demand ↑  →  POSITIVE coefficient
```

### Trap 2: "Use All Available Data"

**Wrong**: Including lag-0 competitor rates for maximum information.

**Right**: Lag-0 creates simultaneity bias. Use lag-1+ only.

```
Wrong:  model(y ~ P_t + C_t)     # C_t correlated with noise in y
Right:  model(y ~ P_t + C_{t-1}) # C_{t-1} predetermined
```

### Trap 3: "All RILA Products Are the Same"

**Wrong**: Treating 10% buffer and 20% buffer identically.

**Right**: Different buffers attract different buyer segments with potentially different elasticity.

```
Buffer matters:
- 10% buffer: Higher risk tolerance buyers, possibly more rate-sensitive
- 20% buffer: Moderate risk tolerance, baseline behavior
```

---

## When to Read What

| You Want To... | Read This |
|----------------|-----------|
| Understand the product | `knowledge/domain/RILA_ECONOMICS.md` |
| Understand the causal model | `knowledge/analysis/CAUSAL_FRAMEWORK.md` |
| Run inference code | `src/notebooks/interface.py` |
| Add a new product | `docs/architecture/PRODUCT_EXTENSION_GUIDE.md` |
| Debug constraint violations | `knowledge/integration/LESSONS_LEARNED.md` |
| Prepare for deployment | `knowledge/practices/LEAKAGE_CHECKLIST.md` |
| Look up a term | `knowledge/domain/GLOSSARY.md` |

---

## Key Equations (Reference)

### Estimand (What We're Estimating)

```
E[Sales_t | do(P_t = p), C_{t-1}, DGS5_t, Season_t, Buffer]
      - E[Sales_t | do(P_t = p'), C_{t-1}, DGS5_t, Season_t, Buffer]
```

### OLS Specification

```
sales_logit_t = β₀ + β₁·P_lag_0 + β₂·C_weighted_lag_k + β₃·DGS5_lag_k
                   + γ·Season + δ·Buffer + ε

Constraints:
  β₁ > 0  (own rate positive)
  β₂ < 0  (competitor rate negative)
```

### Logit Transform

```python
sales_scaled = 0.95 * sales / max(sales)
sales_logit = log(sales_scaled / (1 - sales_scaled))
```

---

## Summary

1. **Business Question**: How does our cap rate affect sales?
2. **Data Flow**: TDE sales + WINK rates → features → constrained OLS → elasticity
3. **Key Pattern**: Dependency injection for flexibility and testability
4. **Critical Rule**: No lag-0 competitor features (causal identification)
5. **Economic Signs**: Own rate positive, competitor rates negative

Now you understand the architecture. Time to run some code—proceed to `notebooks/onboarding/architecture_walkthrough.ipynb`.
