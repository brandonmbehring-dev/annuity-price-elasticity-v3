# Notebook sys.path Detection Fix

**Date**: 2026-01-29
**Issue**: Production notebooks failing with `ModuleNotFoundError: No module named 'src'`
**Root Cause**: sys.path detection logic checked for vestigial directory structure from pre-refactor architecture
**Status**: [DONE] Fixed in all 6 production notebooks

---

## Problem Summary

### Broken Import Pattern

All RILA production notebooks were failing on the first import statement:

```python
from src.data.output_manager import configure_output_mode, get_output_info
# ModuleNotFoundError: No module named 'src'
```

### Root Cause Analysis

The sys.path detection logic in production notebooks was checking for **old directory paths** that no longer exist:

**OLD (Broken) Logic:**
```python
cwd = os.getcwd()
if 'notebooks/rila/production' in cwd or 'notebooks/rila/eda' in cwd:  # [ERROR] WRONG PATH
    project_root = Path(cwd).parents[3]
elif 'notebooks/rila' in cwd:  # [ERROR] WRONG PATH
    project_root = Path(cwd).parents[2]
else:
    project_root = cwd  # [ERROR] Falls back to notebook directory
```

**Actual Directory Structure:**
```
RILA_6Y20B_refactored/
├── notebooks/
│   ├── production/
│   │   ├── rila_6y20b/          # ← Actual location
│   │   │   ├── 00_data_pipeline.ipynb
│   │   │   ├── 01_price_elasticity_inference.ipynb
│   │   │   └── 02_time_series_forecasting.ipynb
│   │   └── rila_1y10b/          # ← Actual location
│   │       ├── 00_data_pipeline.ipynb
│   │       ├── 01_price_elasticity_inference.ipynb
│   │       └── 02_time_series_forecasting.ipynb
│   └── eda/
│       └── rila_6y20b/
└── src/                         # ← Import target
```

**Problem**: When running from `/notebooks/production/rila_6y20b/`:
- Path contains: `notebooks/production/rila_6y20b`
- Old logic checks for: `notebooks/rila/production` [ERROR]
- No conditions match → falls back to `project_root = cwd` (notebook directory)
- Import fails because `src/` is not in notebook directory

---

## Solution

### Fixed sys.path Detection Logic

**NEW (Fixed) Logic:**
```python
from pathlib import Path
import sys
import os

cwd = os.getcwd()

# Check for actual production notebook structure
if 'notebooks/production/rila' in cwd:
    # Running from /notebooks/production/rila_6y20b/ or rila_1y10b/
    project_root = Path(cwd).parents[2]  # Up 2 levels: rila_6y20b → production → notebooks → ROOT
elif 'notebooks/production/fia' in cwd:
    # Running from /notebooks/production/fia/
    project_root = Path(cwd).parents[2]
elif 'notebooks/eda/rila' in cwd:
    # Running from /notebooks/eda/rila_6y20b/
    project_root = Path(cwd).parents[2]
elif 'notebooks/archive' in cwd:
    # Running from /notebooks/archive/*/
    project_root = Path(cwd).parents[2]
elif os.path.basename(cwd) == 'notebooks':
    # Running from /notebooks/
    project_root = Path(cwd).parent
else:
    # Running from project root (fallback)
    project_root = Path(cwd)

project_root = str(project_root)

# IMPORTANT: Add verification to catch future issues early
if not os.path.exists(os.path.join(project_root, 'src')):
    raise RuntimeError(
        f"sys.path setup failed: 'src' package not found at {project_root}/src\n"
        f"Current directory: {cwd}\n"
        "This indicates the sys.path detection logic needs adjustment for your directory structure."
    )

sys.path.insert(0, project_root)
print(f"[PASS] Project root: {project_root}")
```

### Key Changes

1. **Pattern Match Fix**: Changed from `'notebooks/rila/production'` to `'notebooks/production/rila'`
   - Now correctly matches actual directory structure

2. **Path Traversal Fix**: Uses `.parents[2]` instead of `.parents[3]` for production notebooks
   - From `/notebooks/production/rila_6y20b/`:
     - `parents[0]` = `production/`
     - `parents[1]` = `notebooks/`
     - `parents[2]` = project root [DONE]

3. **Added Verification**: Checks that `src/` exists before continuing
   - Fails fast with clear error message if path detection is wrong
   - Prevents confusing "ModuleNotFoundError" later

---

## Directory Structure Mapping

| Notebook Location | cwd Example | Condition Match | Parents Needed | Project Root |
|-------------------|-------------|-----------------|----------------|--------------|
| RILA 6Y20B prod | `/home/.../notebooks/production/rila_6y20b/` | `'notebooks/production/rila' in cwd` | 2 | `/home/.../RILA_6Y20B_refactored/` |
| RILA 1Y10B prod | `/home/.../notebooks/production/rila_1y10b/` | `'notebooks/production/rila' in cwd` | 2 | `/home/.../RILA_6Y20B_refactored/` |
| RILA EDA | `/home/.../notebooks/eda/rila_6y20b/` | `'notebooks/eda/rila' in cwd` | 2 | `/home/.../RILA_6Y20B_refactored/` |
| FIA prod | `/home/.../notebooks/production/fia/` | `'notebooks/production/fia' in cwd` | 2 | `/home/.../RILA_6Y20B_refactored/` |
| Archive | `/home/.../notebooks/archive/*/` | `'notebooks/archive' in cwd` | 2 | `/home/.../RILA_6Y20B_refactored/` |
| Notebooks dir | `/home/.../notebooks/` | `basename == 'notebooks'` | 1 | `/home/.../RILA_6Y20B_refactored/` |

---

## Fixed Notebooks

All 6 production notebooks have been updated with corrected sys.path detection:

### RILA 6Y20B (3 notebooks)
- [DONE] `/notebooks/production/rila_6y20b/00_data_pipeline.ipynb` (Cell 2)
- [DONE] `/notebooks/production/rila_6y20b/01_price_elasticity_inference.ipynb` (Cell 3)
- [DONE] `/notebooks/production/rila_6y20b/02_time_series_forecasting.ipynb` (Cell 3)

### RILA 1Y10B (3 notebooks)
- [DONE] `/notebooks/production/rila_1y10b/00_data_pipeline.ipynb` (Cell 2)
- [DONE] `/notebooks/production/rila_1y10b/01_price_elasticity_inference.ipynb` (Cell 3)
- [DONE] `/notebooks/production/rila_1y10b/02_time_series_forecasting.ipynb` (Cell 3)

---

## Verification

### Test Import from Production Directory

```bash
# Test RILA 6Y20B
cd /home/sagemaker-user/RILA_6Y20B_refactored/notebooks/production/rila_6y20b/
python3 -c "
from pathlib import Path
import os, sys
cwd = os.getcwd()
if 'notebooks/production/rila' in cwd:
    project_root = Path(cwd).parents[2]
project_root_str = str(project_root)
sys.path.insert(0, project_root_str)
from src.data import extraction as ext
print(f'[PASS] Import successful from: {cwd}')
"

# Test RILA 1Y10B
cd /home/sagemaker-user/RILA_6Y20B_refactored/notebooks/production/rila_1y10b/
python3 -c "
from pathlib import Path
import os, sys
cwd = os.getcwd()
if 'notebooks/production/rila' in cwd:
    project_root = Path(cwd).parents[2]
project_root_str = str(project_root)
sys.path.insert(0, project_root_str)
from src.data import extraction as ext
print(f'[PASS] Import successful from: {cwd}')
"
```

**Expected Output:**
```
[PASS] Import successful from: /home/sagemaker-user/RILA_6Y20B_refactored/notebooks/production/rila_6y20b
[PASS] Import successful from: /home/sagemaker-user/RILA_6Y20B_refactored/notebooks/production/rila_1y10b
```

---

## Implementation Notes

### Inline Documentation

Each fixed notebook includes a comment block explaining the fix:

```python
# =============================================================================
# FIXED: 2026-01-29 - Corrected sys.path detection for /notebooks/production/rila_* structure
# OLD: Checked for 'notebooks/rila/production' (wrong - vestigial path from old structure)
# NEW: Checks for 'notebooks/production/rila' (correct - actual directory structure)
# See: docs/development/NOTEBOOK_SYS_PATH_FIX.md for details
# =============================================================================
```

### Verification Check

All fixed notebooks include a runtime verification:

```python
# IMPORTANT: Verify import will work (catches future sys.path issues early)
if not os.path.exists(os.path.join(project_root, 'src')):
    raise RuntimeError(
        f"sys.path setup failed: 'src' package not found at {project_root}/src\n"
        f"Current directory: {cwd}\n"
        "This indicates the sys.path detection logic needs adjustment."
    )
```

This ensures that if directory structure changes in the future, notebooks will fail immediately with a clear error message rather than later with confusing import errors.

---

## Future Prevention

### Guidelines for New Notebooks

When creating new notebooks that need to import from `src/`, use this standard pattern:

```python
from pathlib import Path
import sys
import os

cwd = os.getcwd()

# Check for actual production notebook structure
if 'notebooks/production/rila' in cwd:
    project_root = Path(cwd).parents[2]
elif 'notebooks/production/fia' in cwd:
    project_root = Path(cwd).parents[2]
elif 'notebooks/eda/rila' in cwd:
    project_root = Path(cwd).parents[2]
elif 'notebooks/archive' in cwd:
    project_root = Path(cwd).parents[2]
elif os.path.basename(cwd) == 'notebooks':
    project_root = Path(cwd).parent
else:
    project_root = Path(cwd)

project_root = str(project_root)

# Verify src exists
if not os.path.exists(os.path.join(project_root, 'src')):
    raise RuntimeError(
        f"sys.path setup failed: 'src' package not found at {project_root}/src\n"
        f"Current directory: {cwd}\n"
    )

sys.path.insert(0, project_root)
print(f"[PASS] Project root: {project_root}")
```

### Directory Structure Rules

1. **Never hardcode absolute paths** - always use relative path detection
2. **Test notebooks from their actual directory** - don't run from project root
3. **Add verification checks** - verify `src/` exists before importing
4. **Document path assumptions** - comment why you're going up N levels

---

## Related Documentation

- [MODULE_HIERARCHY.md](../development/MODULE_HIERARCHY.md) - Module organization and import patterns
- [NOTEBOOK_QUICKSTART.md](../onboarding/NOTEBOOK_QUICKSTART.md) - Running notebooks guide
- [CODING_STANDARDS.md](../development/CODING_STANDARDS.md) - Import and path conventions

---

**Status**: [DONE] Complete - All production notebooks fixed and verified
**Impact**: Critical - Unblocks all RILA production notebook execution
**Testing**: Manual verification in both rila_6y20b and rila_1y10b directories
