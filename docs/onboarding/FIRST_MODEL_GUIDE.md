# Your First RILA Model in 5 Minutes

This guide gets you from zero to running a production-ready price elasticity model with minimal friction.

## Prerequisites Checklist

Verify you have these before starting:

- [ ] **Python 3.9 or higher** installed
  - Check: `python --version` (should show 3.9.x, 3.10.x, 3.11.x, or 3.12.x)
  - If not: [Download Python](https://www.python.org/downloads/)

- [ ] **conda or pip** package manager available
  - Check: `conda --version` OR `pip --version`
  - If not: [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html)

- [ ] **Git** for cloning the repository
  - Check: `git --version`
  - If not: [Install Git](https://git-scm.com/downloads)

- [ ] **10 GB free disk space** (for data and model artifacts)
  - Check: `df -h .` (on Linux/Mac) or `dir` (on Windows)

## Step-by-Step Installation

Follow these steps exactly. Each step includes verification.

### Step 1: Clone the Repository

```bash
cd ~
git clone https://github.com/your-org/annuity-price-elasticity-v3.git
cd annuity-price-elasticity-v3
```

**Verify:**
```bash
ls README.md
# Should output: README.md
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
```

This takes 2-5 minutes. You'll see packages being downloaded and installed.

**Verify:**
```bash
conda env list | grep annuity-price-elasticity-v3
# Should show: annuity-price-elasticity-v3  /path/to/envs/annuity-price-elasticity-v3
```

### Step 3: Activate Environment

```bash
conda activate annuity-price-elasticity-v3
```

Your terminal prompt should change to show `(annuity-price-elasticity-v3)`.

**Verify:**
```bash
python -c "import sys; print(sys.prefix)"
# Should show path containing 'annuity-price-elasticity-v3'
```

### Step 4: Install Package

```bash
pip install -e .
```

The `-e` flag installs in "editable" mode so code changes are immediately available.

**Verify:**
```bash
python -c "from src.notebooks import create_interface; print('[PASS] Installation verified')"
# Should output: [PASS] Installation verified
```

If you see `ModuleNotFoundError`, you skipped the `pip install -e .` step!

## Three Working Examples

Copy-paste these examples to verify your installation. Expected output is shown below each.

### Example 1: Basic Inference (Fixture Data)

```python
from src.notebooks import create_interface

# Create interface with pre-loaded test data
interface = create_interface("6Y20B", environment="fixture")

# Load data
df = interface.load_data()
print(f"[PASS] Loaded {len(df)} rows of data")

# Run inference
results = interface.run_inference(df)
print(f"[PASS] Inference completed")
print(f"[PASS] Model R²: {results['metrics']['r_squared']:.4f}")
```

**Expected Output:**
```
Loading data for RILA 6Y20B (fixture environment)...
[PASS] Loaded 1248 rows of data
Running inference pipeline...
Bootstrap iterations: 10000/10000 [100%]
[PASS] Inference completed
[PASS] Model R²: 0.7837
```

**What This Does:** Runs the full inference pipeline on pre-processed test data stored locally. No AWS credentials needed.

---

### Example 2: Inspect Model Configuration

```python
from src.notebooks import create_interface

interface = create_interface("6Y20B", environment="fixture")

# Get configuration
config = interface.get_config()

print(f"Product: {config['product_name']}")
print(f"Bootstrap samples: {config['bootstrap_samples']}")
print(f"Features: {len(config['feature_names'])} engineered features")
print(f"Target variable: {config['target_variable']}")
```

**Expected Output:**
```
Product: RILA 6Y20B
Bootstrap samples: 10000
Features: 598 engineered features
Target variable: weekly_sales_units
```

**What This Does:** Displays the model configuration for RILA 6Y20B product without running inference.

---

### Example 3: Export Results to Excel

```python
from src.notebooks import create_interface

interface = create_interface("6Y20B", environment="fixture")
df = interface.load_data()
results = interface.run_inference(df)

# Export to Excel
output_path = interface.export_results(results, format="excel")
print(f"[PASS] Results saved to: {output_path}")
```

**Expected Output:**
```
Loading data for RILA 6Y20B (fixture environment)...
Running inference pipeline...
Bootstrap iterations: 10000/10000 [100%]
Exporting results to Excel...
[PASS] Results saved to: output/rila_6y20b_inference_2026-01-28.xlsx
```

**What This Does:** Runs inference and exports results to a timestamped Excel workbook with multiple sheets (predictions, metrics, diagnostics).

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'src'`

**Cause:** You didn't run `pip install -e .`

**Fix:**
```bash
cd /path/to/annuity-price-elasticity-v3
pip install -e .
```

---

### Error: `FileNotFoundError: data/fixture/rila_6y20b.parquet`

**Cause:** Fixture data files are missing

**Fix:**
```bash
# Download fixture data (if repository doesn't include it)
python scripts/download_fixtures.py
```

Or contact your team lead for access to fixture data.

---

### Error: `ImportError: cannot import name 'create_interface'`

**Cause:** Wrong Python environment activated

**Fix:**
```bash
conda activate annuity-price-elasticity-v3
python -c "from src.notebooks import create_interface"
```

---

### Error: `KeyError: '6Y20B'`

**Cause:** Invalid product code

**Fix:** Use exact product code `"6Y20B"` (case-sensitive). Available products:
- `"6Y20B"` - RILA 6-year 20% buffer
- `"FIA_5Y"` - Fixed Index Annuity 5-year (if configured)
- `"MYGA_3Y"` - Multi-Year Guaranteed Annuity 3-year (if configured)

---

### Warning: `Bootstrap convergence not achieved`

**Cause:** Insufficient bootstrap samples or data quality issues

**Fix:** This is usually informational. If you need more precision:
```python
interface = create_interface("6Y20B", environment="fixture", bootstrap_samples=20000)
```

---

### Performance: Model takes >5 minutes to run

**Cause:** Large dataset or resource constraints

**Fix:**
- Use fixture data for testing (`environment="fixture"`)
- Reduce bootstrap samples for experimentation: `bootstrap_samples=1000`
- Check available RAM: `free -h` (Linux) or Activity Monitor (Mac)

---

## What to Do Next

**Option 1: Run Notebooks Interactively**
- See [NOTEBOOK_QUICKSTART.md](./NOTEBOOK_QUICKSTART.md) for guided notebook walkthrough
- Start with: `production/rila_6y20b/01_price_elasticity_inference.ipynb`

**Option 2: Explore the Codebase**
- Architecture overview: [docs/onboarding/MENTAL_MODEL.md](./MENTAL_MODEL.md)
- Configuration guide: [docs/reference/CONFIGURATION_REFERENCE.md](../reference/CONFIGURATION_REFERENCE.md)
- API reference: [docs/api/index.html](../api/index.html)

**Option 3: Common Tasks**
- [docs/onboarding/COMMON_TASKS.md](./COMMON_TASKS.md)
  - Change model parameters
  - Add new features
  - Run with AWS production data
  - Debug model output

**Option 4: Understand the Business Context**
- [docs/business/executive_summary.md](../business/executive_summary.md)
- Why price elasticity matters for annuities
- Key stakeholders and decision-making

## Quick Reference

**Run inference:**
```python
from src.notebooks import create_interface
interface = create_interface("6Y20B", environment="fixture")
results = interface.run_inference(interface.load_data())
```

**Run tests:**
```bash
make test
```

**Check coverage:**
```bash
make coverage
```

**Activate environment:**
```bash
conda activate annuity-price-elasticity-v3
```

**Deactivate environment:**
```bash
conda deactivate
```

## Getting Help

- **Common errors:** See Troubleshooting section above
- **API questions:** Check [docs/api/](../api/)
- **Team Slack:** #annuity-price-elasticity-v3-support
- **Bug reports:** [GitHub Issues](https://github.com/your-org/annuity-price-elasticity-v3/issues)
