# Getting Started: RILA Price Elasticity

**For new data scientists joining the team.**

This guide takes you from zero to running your first model. Follow the sections in order—each builds on the previous.

---

## Prerequisites

### Technical Skills

| Skill | Level Required | Quick Refresh |
|-------|----------------|---------------|
| Python | Intermediate | `pandas`, `numpy`, `sklearn` |
| Statistics | Basic regression | OLS, confidence intervals |
| Git | Basic workflow | clone, branch, commit |

### Don't Have Econometrics Background?

That's OK. Read `docs/fundamentals/ECONOMETRICS_PRIMER.md` alongside this guide. The key concepts you'll need:
- Causality vs correlation
- Omitted variable bias
- Why lagged features matter

---

## Day 1: Orientation (2 hours)

### Hour 1: Understand the Product

**What are we modeling?** Price elasticity of RILA (Registered Index-Linked Annuity) products—how changes in cap rates affect sales.

#### Read These (45 min total)

1. **`knowledge/domain/RILA_ECONOMICS.md`** (20 min)
   - Key insight: Cap rate is a **yield** (customer benefit), NOT a price
   - Higher cap rate → more sales (positive coefficient, not negative)
   - This is the opposite of traditional price elasticity!

2. **`knowledge/domain/GLOSSARY.md`** (10 min)
   - Skim for key terms: buffer, cap rate, RILA vs FIA
   - Bookmark this—you'll reference it often

3. **`knowledge/domain/FIXED_DEFERRED_ANNUITY_TAXONOMY.md`** (15 min)
   - How RILA fits in the annuity product family
   - Key differences: RILA has buffers (partial protection), FIA has floors (full protection)

#### Key Takeaways

```
RILA Products:
- Higher cap rate = better yield = more customer demand
- Buffer absorbs first X% of loss (10%, 20%, etc.)
- Product codes: 6Y20B = 6-year term, 20% buffer

Economic Sign Expectations:
- Own rate (Prudential): POSITIVE coefficient
- Competitor rates: NEGATIVE coefficient (substitution effect)
```

### Hour 2: Understand the Model

**Why is this causal, not just predictive?** We're estimating the *effect* of changing rates, not just predicting sales.

#### Read These (35 min)

1. **`knowledge/analysis/CAUSAL_FRAMEWORK.md`** (25 min)
   - The causal question we're answering
   - Why lag-0 competitor features are FORBIDDEN
   - The DAG (causal diagram) showing variable relationships

2. **`knowledge/integration/LESSONS_LEARNED.md`** (10 min)
   - **Five critical traps** that have caused problems
   - Read the "CRITICAL TRAPS" section thoroughly

#### Key Takeaways

```
Causal Identification:
- Own rate (P_lag_0): OK to use current—we set it before observing sales
- Competitor rates (C_lag_1+): MUST be lagged to avoid simultaneity bias
- C_lag_0 is FORBIDDEN: competitors and sales respond to same market conditions

Why This Matters:
- Using C_lag_0 creates spurious correlation (not causal)
- Model will reject features with lag-0 competitor patterns
```

### Hands-On: Run Your First Model (25 min)

Now let's make it concrete. Run the architecture walkthrough notebook.

```bash
# Activate environment
conda activate annuity-price-elasticity-v2

# Start Jupyter
jupyter lab notebooks/onboarding/architecture_walkthrough.ipynb
```

This notebook will:
- Load test data (fixtures—no AWS needed)
- Create an interface for the 6Y20B product
- Run inference
- Interpret the coefficients

If you don't have the environment set up yet, see [Installation](#installation) below.

---

## Day 2: Deep Dive (2 hours)

### Hour 1: Data & Features

1. **`knowledge/domain/WINK_SCHEMA.md`** (15 min)
   - Rate data structure (competitor cap rates)
   - Key fields: `ratePct`, `company`, `productCode`

2. **`knowledge/domain/TDE_SCHEMA.md`** (15 min)
   - Sales data structure
   - Key fields: `week_start_date`, `total_premium`

3. **`knowledge/analysis/FEATURE_RATIONALE.md`** (30 min)
   - Why we chose specific features
   - Lag structure rationale (t-2, t-3 most predictive)
   - Holiday mask necessity

### Hour 2: Architecture & Code

1. **`docs/onboarding/MENTAL_MODEL.md`** (20 min)
   - High-level system architecture
   - Dependency injection pattern
   - Where to find things

2. **`docs/architecture/MULTI_PRODUCT_DESIGN.md`** (20 min)
   - Design decisions and tradeoffs
   - How to add new products

3. **`MODULE_HIERARCHY.md`** (20 min)
   - Complete code navigation
   - Module responsibilities

---

## Day 3: Practice Tasks

### Task 1: Run Inference with Different Products

```python
from src.notebooks import create_interface

# Try each supported product
for code in ["6Y20B", "6Y10B", "10Y20B"]:
    interface = create_interface(code, environment="fixture")
    df = interface.load_data()
    results = interface.run_inference(df)
    print(f"{code}: elasticity = {results['elasticity_point']:.4f}")
```

### Task 2: Validate Economic Constraints

```python
interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()
results = interface.run_inference(df)

# Check coefficient signs
validation = interface.validate_coefficients(results["coefficients"])
print("Passed:", validation["passed"])
print("Violated:", validation["violated"])
```

### Task 3: Explore the Causal Framework

Open `knowledge/analysis/CAUSAL_FRAMEWORK.md` and:
1. Trace through the DAG (causal diagram)
2. Identify which variables are confounders
3. Explain why DGS5 (Treasury rate) is controlled for

---

## Installation

### Option A: Conda (Recommended)

```bash
# Clone repository
git clone <repo-url>
cd annuity-price-elasticity-v2

# Create environment
conda env create -f environment.yml
conda activate annuity-price-elasticity-v2

# Verify installation
python -c "from src.notebooks import create_interface; print('OK')"
```

### Option B: pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.notebooks import create_interface; print('OK')"
```

### Verify Tests Pass

```bash
# Quick smoke test (30 seconds)
make quick-check

# Full test suite (2-3 minutes)
make test
```

---

## Using Claude Code

This repository is optimized for AI-assisted development with Claude Code.

**Read**: `docs/onboarding/USING_CLAUDE_CODE.md` for:
- How `CLAUDE.md` guides Claude's behavior
- Effective prompting patterns
- What Claude knows about this codebase

---

## Reference Reading Order

```
START HERE
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ docs/onboarding/GETTING_STARTED.md  (this file)          │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ knowledge/domain/RILA_ECONOMICS.md    (product basics)   │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ knowledge/analysis/CAUSAL_FRAMEWORK.md  (why we model)   │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ knowledge/integration/LESSONS_LEARNED.md (critical traps)│
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ notebooks/onboarding/architecture_walkthrough.ipynb      │
└──────────────────────────────────────────────────────────┘
    │
    ▼
REFERENCE AS NEEDED:
  • GLOSSARY.md           — Terms and definitions
  • MODULE_HIERARCHY.md   — Code navigation
  • LEAKAGE_CHECKLIST.md  — Pre-deployment validation
  • COMMON_TASKS.md       — How to do common tasks
  • TROUBLESHOOTING.md    — Error solutions
```

---

## Quick Reference: Economic Intuition

| Coefficient | Expected Sign | Why |
|-------------|---------------|-----|
| Own rate (P) | **Positive** | Higher cap = better yield = more sales |
| Competitor (C) | **Negative** | Higher competitor rates = substitution |
| DGS5 | Under review | Treasury affects both rates and alternatives |

### Critical Rules

1. **Never use lag-0 competitors** — Simultaneity bias
2. **Expect positive own-rate coefficient** — Cap rate is yield, not price
3. **Use market-share weighted competitor means** — RILA market is concentrated

---

## Success Milestones

| Milestone | Target Time | How to Verify |
|-----------|-------------|---------------|
| Explain what RILA is | 30 min | Can describe buffer vs floor |
| Run inference on fixtures | 1 hour | Gets coefficient results |
| Explain causal identification | 2 hours | Knows why lag-0 is forbidden |
| Add new product variant | Day 2 | PR with tests passing |
| Debug sign violation | Day 3 | Knows diagnostic steps |

---

## Getting Help

- **Claude Code**: Use AI-assisted development (see `docs/onboarding/USING_CLAUDE_CODE.md`)
- **GLOSSARY.md**: When you encounter unfamiliar terms
- **TROUBLESHOOTING.md**: When you hit errors
- **LESSONS_LEARNED.md**: When something seems wrong

---

## Next Steps

After completing this guide:

1. **Week 1**: Work through `docs/onboarding/COMMON_TASKS.md`
2. **Week 2**: Read fundamentals primers in `docs/fundamentals/`
3. **Week 3**: Attempt your first feature addition (with mentorship)

Welcome to the team!
