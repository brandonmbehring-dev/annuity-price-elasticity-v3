# Python Best Practices Primer

**For data scientists new to professional Python development.**

This primer covers the practices we follow in this repository. Good habits here make collaboration easier and bugs rarer.

---

## Virtual Environments

**Never install packages globally.** Use isolated environments.

### Conda (Recommended)

```bash
# Create from environment file
conda env create -f environment.yml
conda activate annuity-price-elasticity-v2

# Or create fresh
conda create -n myenv python=3.11
conda activate myenv

# Install packages
conda install pandas numpy scikit-learn

# Export environment
conda env export > environment.yml
```

### venv (Standard Library)

```bash
# Create
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install
pip install -r requirements.txt
```

**Why environments matter**: Different projects need different package versions. Environments prevent conflicts.

---

## Code Style

### Use Black for Formatting

```bash
# Format a file
black src/module.py

# Format entire project
black .

# Check without modifying
black --check .
```

**This repo uses 100-character line length** (not Black's default 88):

```bash
black --line-length 100 .
```

### Type Hints

**Always use type hints.** They catch bugs and serve as documentation.

```python
# BAD: No types
def calculate_elasticity(coefficients, data):
    return coefficients["own_rate"] * data.mean()

# GOOD: With types
def calculate_elasticity(
    coefficients: dict[str, float],
    data: pd.DataFrame
) -> float:
    """Calculate point elasticity from coefficients and data."""
    return coefficients["own_rate"] * data.mean()
```

Common types:
```python
from typing import Optional, Dict, List, Any, Tuple

def process(
    data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    features: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ...
```

### Docstrings (NumPy Style)

```python
def load_sales_data(
    product_filter: Optional[str] = None,
    start_date: str = "2022-01-01",
) -> pd.DataFrame:
    """Load and filter sales data.

    Parameters
    ----------
    product_filter : Optional[str]
        Filter by product name. If None, load all products.
    start_date : str
        Earliest date to include (YYYY-MM-DD format).

    Returns
    -------
    pd.DataFrame
        Sales data with columns: date, product, premium, units.

    Raises
    ------
    ValueError
        If start_date is invalid format.
    FileNotFoundError
        If data file not found.

    Examples
    --------
    >>> df = load_sales_data(product_filter="6Y20B")
    >>> len(df)
    150
    """
```

---

## Function Design

### Keep Functions Small

**Target: 20-50 lines.** If longer, consider splitting.

```python
# BAD: Giant function doing everything
def analyze_data(df):
    # 200 lines of mixed logic...

# GOOD: Composed small functions
def analyze_data(df: pd.DataFrame) -> AnalysisResults:
    """Orchestrate the analysis pipeline."""
    df_clean = clean_data(df)
    features = engineer_features(df_clean)
    model = fit_model(features)
    return package_results(model)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove nulls and outliers."""
    # 15 lines

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and interaction features."""
    # 25 lines
```

### Single Responsibility

Each function should do **one thing well**.

```python
# BAD: Doing too much
def load_and_process_and_model(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df["feature"] = df["a"] * df["b"]
    model = fit(df)
    return model

# GOOD: Separate concerns
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["feature"] = df["a"] * df["b"]
    return df

def train_model(df: pd.DataFrame) -> Model:
    return fit(df)
```

### Return New Objects (Immutability)

**Never modify inputs in place** unless explicitly documented.

```python
# BAD: Modifies input
def add_feature(df):
    df["new_col"] = df["a"] + df["b"]  # Mutates df!
    return df

# GOOD: Returns new object
def add_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Work on copy
    df["new_col"] = df["a"] + df["b"]
    return df
```

---

## Error Handling

### Fail Fast with Clear Messages

**Never fail silently.** Raise exceptions with context.

```python
# BAD: Silent failure
def load_data(path):
    try:
        return pd.read_csv(path)
    except:
        return None  # Caller has no idea what went wrong!

# GOOD: Explicit failure
def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV file.

    Raises
    ------
    FileNotFoundError
        If path doesn't exist.
    ValueError
        If file is empty or malformed.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. "
            f"Required action: Check path or create file."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Data file is empty: {path}. "
            f"Business impact: Cannot proceed with analysis."
        )

    return df
```

### Exception Hierarchy

This repo uses custom exceptions in `src/core/exceptions.py`:

```python
from src.core.exceptions import DataLoadError, ConstraintViolationError

try:
    df = load_sales_data()
except DataLoadError as e:
    # Handle data loading issues
    logger.error(f"Failed to load data: {e}")
    raise
```

---

## Testing

### Use pytest

```bash
# Run all tests
pytest

# Run specific file
pytest tests/unit/test_interface.py

# Run with coverage
pytest --cov=src

# Run only tests matching pattern
pytest -k "test_validation"
```

### Test Structure

```python
# tests/unit/test_interface.py

import pytest
from src.notebooks.interface import create_interface

class TestCreateInterface:
    """Tests for create_interface factory function."""

    def test_creates_fixture_interface(self):
        """Fixture environment creates FixtureAdapter."""
        interface = create_interface("6Y20B", environment="fixture")
        assert interface.adapter.source_type == "fixture"

    def test_raises_for_unknown_product(self):
        """Unknown product code raises ValueError."""
        with pytest.raises(ValueError, match="Unknown product"):
            create_interface("INVALID")

    @pytest.fixture
    def sample_data(self):
        """Fixture providing test DataFrame."""
        return pd.DataFrame({
            "sales_target_current": [100, 110, 105],
            "prudential_rate_current": [9.5, 9.3, 9.8],
        })

    def test_inference_with_valid_features(self, sample_data):
        """Inference runs with valid features."""
        interface = create_interface("6Y20B", environment="fixture")
        # ... test logic
```

### Test Categories

```
tests/
├── unit/           # Test individual functions in isolation
├── integration/    # Test component interactions
├── property_based/ # Test with generated inputs
└── fixtures/       # Test data (parquet files)
```

---

## Git Workflow

### Branch Naming

```bash
# Feature
git checkout -b feature/add-volatility-control

# Bug fix
git checkout -b fix/lag-zero-detection

# Refactor
git checkout -b refactor/interface-methods
```

### Commit Messages

```bash
# Format: type: description
git commit -m "feat: Add VIX as control variable

Include market volatility in feature set per CAUSAL_FRAMEWORK.md.
Added vix_t1 and vix_t2 lag features.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`

### Before Committing

```bash
# Run checks
make quick-check  # Fast validation
make test         # Full test suite
make lint         # Code quality

# Stage specific files (not everything)
git add src/specific_file.py tests/test_specific.py

# Review changes
git diff --staged
```

---

## Debugging

### Use the Debugger

```python
# Add breakpoint
def complex_function(data):
    result = first_step(data)
    breakpoint()  # Execution pauses here
    final = second_step(result)
    return final
```

Then run normally—Python will drop into debugger at the breakpoint.

### Debugger Commands

```
n (next)     - Execute next line
s (step)     - Step into function
c (continue) - Continue until next breakpoint
p variable   - Print variable value
pp variable  - Pretty-print variable
l (list)     - Show code context
q (quit)     - Exit debugger
```

### Print Debugging (Quick)

```python
# For quick checks
print(f"DEBUG: df shape = {df.shape}")
print(f"DEBUG: columns = {df.columns.tolist()}")
print(f"DEBUG: feature value = {feature_value!r}")
```

### Logging (Production)

```python
import logging

logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")

    df = pd.read_csv(path)
    logger.debug(f"Loaded {len(df)} rows")

    if df.empty:
        logger.warning(f"Empty dataframe from {path}")

    return df
```

---

## Project Structure

```
src/
├── __init__.py          # Package marker
├── core/                # Foundational abstractions
│   ├── __init__.py
│   ├── protocols.py     # Interface definitions
│   ├── types.py         # Type definitions
│   └── exceptions.py    # Custom exceptions
├── module_name/
│   ├── __init__.py      # Exports public API
│   ├── public_api.py    # Functions users call
│   └── _internal.py     # Implementation details (underscore prefix)
```

### Imports

```python
# Absolute imports (preferred)
from src.core.types import ProductConfig
from src.notebooks.interface import create_interface

# Relative imports (within package only)
from .interface import create_interface
from ..core.types import ProductConfig
```

### `__init__.py` Exports

```python
# src/notebooks/__init__.py

from src.notebooks.interface import (
    UnifiedNotebookInterface,
    create_interface,
)

__all__ = [
    "UnifiedNotebookInterface",
    "create_interface",
]
```

---

## Quick Reference

| Practice | Do | Don't |
|----------|-----|-------|
| Environments | Use conda/venv | Install globally |
| Formatting | Run Black | Manual formatting |
| Type hints | Always use | Omit types |
| Functions | 20-50 lines | 200+ line monsters |
| Errors | Raise with context | Return None silently |
| Testing | pytest with fixtures | Manual testing only |
| Git | Specific file staging | `git add -A` |

---

## Further Reading

- **This repo's standards**: `CODING_STANDARDS.md` (if exists)
- **Python style guide**: PEP 8, PEP 484 (type hints)
- **Testing**: pytest documentation
- **Git**: Pro Git book (free online)
