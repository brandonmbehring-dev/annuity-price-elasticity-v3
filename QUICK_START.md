# Quick Start (5 Minutes)

Get the RILA Price Elasticity model running in 5 minutes.

## Prerequisites

- Python 3.9+
- conda or pip

## Installation

```bash
# Clone repository
cd annuity-price-elasticity-v3

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate annuity-price-elasticity-v3

# Install package in editable mode (REQUIRED)
pip install -e .

# Verify installation
python -c "from src.notebooks import create_interface; print('✓ Installation verified - Ready to run models')"
```

## Run Your First Model

```python
from src.notebooks import create_interface

# Create interface for RILA 6Y20B product
interface = create_interface("6Y20B", environment="fixture")

# Load data
df = interface.load_data()

# Run inference
results = interface.run_inference(df)

# Export results
interface.export_results(results, format="excel")

print(f"✓ Model completed successfully")
print(f"✓ Results exported to: {results.get('output_path', 'output/')}")
```

## What Just Happened?

You ran a production-ready price elasticity model that:
- Loaded pre-processed fixture data (598 engineered features)
- Executed Bootstrap Ridge Regression with 10,000 samples
- Generated point estimates with 95% confidence intervals
- Achieved 78.37% R² accuracy on validation data

**Expected Output:**
```
Loading data for RILA 6Y20B (fixture environment)...
Running inference pipeline...
Bootstrap iterations: 10000/10000 [100%]
Model R²: 0.7837
✓ Model completed successfully
✓ Results exported to: output/rila_6y20b_inference_results.xlsx
```

## ⚠️ Critical Traps - Read Before You Proceed

Before diving deeper, **read the 5 critical traps** that have caused problems in annuity elasticity work:
→ [docs/integration/LESSONS_LEARNED.md](docs/integration/LESSONS_LEARNED.md)

Key trap: Cap rate is a **YIELD** (customer benefit), not a price. Expect a **positive** coefficient.

## Next Steps

- **Critical traps:** [docs/integration/LESSONS_LEARNED.md](docs/integration/LESSONS_LEARNED.md) ⚠️ **READ FIRST**
- **Full onboarding:** [docs/onboarding/GETTING_STARTED.md](docs/onboarding/GETTING_STARTED.md) (2 hours)
- **Common tasks:** [docs/onboarding/COMMON_TASKS.md](docs/onboarding/COMMON_TASKS.md)
- **Architecture overview:** [docs/onboarding/MENTAL_MODEL.md](docs/onboarding/MENTAL_MODEL.md)
- **Day one checklist:** [docs/onboarding/day_one_checklist.md](docs/onboarding/day_one_checklist.md)
- **Business context:** [docs/business/executive_summary.md](docs/business/executive_summary.md)
