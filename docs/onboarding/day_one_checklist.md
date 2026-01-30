# Day One Checklist

Your interactive guide to a productive first day with the RILA Price Elasticity model.

**Goal:** By end of day, you'll have a running environment and understand the basics of RILAs, price elasticity, and this codebase.

**Time Required:** 4 hours

---

## Morning Session (2 hours)

### Setup & Environment (30 minutes)

- [ ] Clone repository
  ```bash
  git clone <repo-url>
  cd annuity-price-elasticity-v3
  ```

- [ ] Create conda environment
  ```bash
  conda env create -f environment.yml
  conda activate annuity-price-elasticity-v3
  ```

- [ ] Verify installation
  ```bash
  make test  # Should see tests pass
  python -c "import src; print('✓ Import successful')"
  ```

- [ ] Open QUICK_START.md
  ```bash
  cat QUICK_START.md
  ```

**Checkpoint:** Environment running, tests passing

---

### Orientation Reading (90 minutes)

Read these documents in order to build mental model:

#### 1. Quick Technical Overview (5 minutes)
- [ ] Read [QUICK_START.md](../../QUICK_START.md)
- [ ] Run the code example (copy-paste into Python/notebook)
- [ ] Verify you get predictions output

**What you learned:** Basic model usage

---

#### 2. What is a RILA? (20 minutes)
- [ ] Read [docs/domain-knowledge/RILA_ECONOMICS.md](../domain-knowledge/RILA_ECONOMICS.md)
- [ ] Focus on:
  - What RILAs are (registered index-linked annuities)
  - How cap rates work
  - Why price matters for sales

**What you learned:** The business problem we're solving

**Key takeaway:** Higher cap rates generally increase sales (positive price elasticity)

---

#### 3. Why This Model Works (25 minutes)
- [ ] Read [docs/analysis/CAUSAL_FRAMEWORK.md](../analysis/CAUSAL_FRAMEWORK.md)
- [ ] Focus on:
  - Price elasticity fundamentals
  - Why competitor rates matter
  - Temporal structure (lags)

**What you learned:** The economic theory behind the model

**Key takeaway:** Sales respond to competitive positioning with 2-3 week lags

---

#### 4. Historical Context (10 minutes)
- [ ] Read [docs/integration/LESSONS_LEARNED.md](../integration/LESSONS_LEARNED.md)
- [ ] Focus on:
  - Why we refactored to v2
  - Key design decisions
  - What to avoid

**What you learned:** Why the code is structured this way

---

#### 5. Getting Started Guide - Part 1 (30 minutes)
- [ ] Read sections 1-2 of [GETTING_STARTED.md](GETTING_STARTED.md)
  - Section 1: Prerequisites and Installation
  - Section 2: Core Concepts

**What you learned:** System architecture and key concepts

---

**Morning Checkpoint:**
- [ ] Can explain what a RILA is to a colleague
- [ ] Understand why we model price elasticity
- [ ] Know why competitor rates matter
- [ ] Environment is working

---

## Afternoon Session (2 hours)

### Hands-On Practice (90 minutes)

#### 1. Architecture Walkthrough (30 minutes)
- [ ] Navigate to `notebooks/onboarding/`
- [ ] Open `architecture_walkthrough.ipynb`
- [ ] Execute all cells
- [ ] Read explanations carefully

**What you learned:** How data flows through the system

---

#### 2. Run Inference on Multiple Products (45 minutes)

**RILA 6Y20B (Production model):**
```python
from src.models.inference_models import BootstrapRidgeInferenceModel
from src.utils.fixture_loader import load_fixture_data
from src.config.product_config import ProductConfig

# Load fixture
fixture = load_fixture_data('rila_6y20b')

# Initialize model
config = ProductConfig.get_config('rila_6y20b')
model = BootstrapRidgeInferenceModel(config)

# Predict
pred = model.predict(fixture)
print(f"Point estimate: {pred['point_estimate']:.2f}")
print(f"95% CI: [{pred['ci_lower']:.2f}, {pred['ci_upper']:.2f}]")

# Check coefficient signs
coefficients = model.get_coefficients()
print(f"Own price coefficient (P_lag_0): {coefficients['P_lag_0']:.4f}")
# Should be POSITIVE (higher cap → more sales)
```

- [ ] Run for RILA 6Y20B
- [ ] Run for RILA 6Y10B
- [ ] Run for RILA 10Y20B

**For each product, verify:**
- [ ] Point estimate is reasonable (not NaN, not extreme)
- [ ] Confidence interval contains point estimate
- [ ] P_lag_0 coefficient is POSITIVE
- [ ] No "leakage warning" messages

---

#### 3. Export Results (15 minutes)

```python
# Export predictions to Excel
import pandas as pd

results = {
    'Product': ['6Y20B', '6Y10B', '10Y20B'],
    'Point_Estimate': [pred_6y20b['point_estimate'],
                       pred_6y10b['point_estimate'],
                       pred_10y20b['point_estimate']],
    'CI_Lower': [pred_6y20b['ci_lower'],
                 pred_6y10b['ci_lower'],
                 pred_10y20b['ci_lower']],
    'CI_Upper': [pred_6y20b['ci_upper'],
                 pred_6y10b['ci_upper'],
                 pred_10y20b['ci_upper']]
}

df = pd.DataFrame(results)
output_path = 'notebooks/outputs/rila_6y20b/bi_team/day_one_results.xlsx'
df.to_excel(output_path, index=False)
print(f"✓ Results saved to {output_path}")
```

- [ ] Export results to Excel
- [ ] Open file and verify formatting
- [ ] Locate in `notebooks/outputs/rila_6y20b/bi_team/`

---

**Afternoon Checkpoint:**
- [ ] Successfully ran 3 product models
- [ ] Verified coefficient signs make economic sense
- [ ] Exported results to Excel
- [ ] Understand data flow from input → prediction

---

## Knowledge Check (30 minutes)

Answer these questions (write in notebook or scratch file):

### 1. Domain Knowledge
- [ ] **What is a RILA?**
  - Expected: Registered Index-Linked Annuity with cap rates

- [ ] **Why is the own-price coefficient (P_lag_0) positive?**
  - Expected: Higher cap rates make product more attractive → more sales

- [ ] **Why do we use lag-2 and lag-3 competitor rates instead of lag-0?**
  - Expected: Prevent data leakage (lag-0 = future information)

### 2. System Understanding
- [ ] **Explain the fixture data structure**
  - Expected: Pre-processed features for quick inference testing

- [ ] **Where are production notebook outputs saved?**
  - Expected: `notebooks/outputs/{product}/bi_team/`

- [ ] **What does the bootstrap ensemble do?**
  - Expected: Provides uncertainty quantification (confidence intervals)

### 3. Practical Skills
- [ ] **How do you run inference for a new product?**
  - Expected: Load fixture → Initialize model → Predict

- [ ] **What should you check after getting predictions?**
  - Expected: Coefficient signs, no leakage warnings, reasonable ranges

- [ ] **Where would you find feature engineering details?**
  - Expected: `docs/methodology/feature_engineering_guide.md` (or methodology report)

---

## End of Day Review

### What You Should Have
- [ ] ✓ Running conda environment with passing tests
- [ ] ✓ Inference results for 3 products (6Y20B, 6Y10B, 10Y20B)
- [ ] ✓ Excel export with predictions and confidence intervals
- [ ] ✓ Notes on questions or confusing topics

### What You Should Understand
- [ ] ✓ What RILAs are and why cap rates matter
- [ ] ✓ Basic price elasticity concepts
- [ ] ✓ Why we prevent data leakage (no lag-0 competitor features)
- [ ] ✓ How to run inference on fixture data
- [ ] ✓ Where to find documentation

### Questions to Ask Tomorrow
Write down any questions from today:
- [ ] _____________________________________________
- [ ] _____________________________________________
- [ ] _____________________________________________

---

## Next Steps

### Day 2: Feature Engineering Deep Dive
- [ ] Read [docs/methodology/feature_engineering_guide.md](../methodology/feature_engineering_guide.md)
- [ ] Read [docs/analysis/FEATURE_RATIONALE.md](../analysis/FEATURE_RATIONALE.md)
- [ ] Explore `src/features/` modules
- [ ] Run data pipeline notebook: `notebooks/production/rila_6y20b/00_data_pipeline.ipynb`

### Day 3: Practice Tasks
- [ ] Complete exercises in [COMMON_TASKS.md](COMMON_TASKS.md)
- [ ] Try modifying inference scenarios
- [ ] Practice validation checks from [docs/practices/LEAKAGE_CHECKLIST.md](../practices/LEAKAGE_CHECKLIST.md)

### Week 1: Fundamentals Primers
- [ ] Complete [GETTING_STARTED.md](GETTING_STARTED.md) (full document)
- [ ] Read [docs/architecture/MULTI_PRODUCT_DESIGN.md](../architecture/MULTI_PRODUCT_DESIGN.md)
- [ ] Review [docs/development/MODULE_HIERARCHY.md](../development/MODULE_HIERARCHY.md)
- [ ] Study production notebooks in `notebooks/production/rila_6y20b/`

---

## Resources

### Quick Links
- **Quick help:** [QUICK_START.md](../../QUICK_START.md)
- **Common tasks:** [COMMON_TASKS.md](COMMON_TASKS.md)
- **Documentation index:** [docs/README.md](../README.md)
- **Emergency procedures:** [docs/operations/EMERGENCY_PROCEDURES.md](../operations/EMERGENCY_PROCEDURES.md)

### Key Documentation
- **For business context:** [docs/business/executive_summary.md](../business/executive_summary.md)
- **For technical details:** [docs/business/methodology_report.md](../business/methodology_report.md)
- **For system design:** [docs/architecture/MULTI_PRODUCT_DESIGN.md](../architecture/MULTI_PRODUCT_DESIGN.md)

---

**Congratulations on completing Day One! 🎉**

You now have a working environment and foundational understanding of RILA price elasticity modeling. Tomorrow you'll dive deeper into feature engineering and data pipelines.
