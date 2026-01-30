#!/usr/bin/env python3
"""
Fix sys.path detection logic in EDA notebooks 01-03

Problem: These notebooks check for 'notebooks/rila/eda' but actual path is 'notebooks/eda/rila_6y20b'
Solution: Replace with production notebook pattern that correctly checks 'notebooks/eda/rila'
"""

import json
from pathlib import Path

# Correct sys.path pattern (from production notebooks)
CORRECT_SYSPATH_BLOCK = '''# Canonical sys.path setup (auto-detect project root)
# Auto-detect project root (handles actual directory structure)
cwd = os.getcwd()

# Check for actual directory structure
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

# IMPORTANT: Verify import will work
if not os.path.exists(os.path.join(project_root, 'src')):
    raise RuntimeError(
        f"sys.path setup failed: 'src' package not found at {project_root}/src\\n"
        f"Current directory: {cwd}\\n"
        "This indicates the sys.path detection logic needs adjustment."
    )

sys.path.insert(0, project_root)
'''


def fix_eda_notebook(notebook_path):
    """Fix sys.path detection in EDA notebooks 01-03."""
    print(f"\n{'='*70}")
    print(f"Processing: {notebook_path}")
    print(f"{'='*70}")

    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Get setup cell (Cell 1, index 1 - index 0 is markdown title)
    setup_cell = nb['cells'][1]
    source = setup_cell['source']

    # Verify it's the setup cell
    cell_text = ''.join(source)
    if 'STANDARD SETUP CELL' not in cell_text:
        print(f"  [FAIL] ERROR: Cell 1 is not the setup cell!")
        return False

    # Path import should already exist
    has_pathlib = any('from pathlib import Path' in line for line in source)
    if not has_pathlib:
        print(f"  [WARN] Warning: Missing 'from pathlib import Path' - should already exist")
        return False

    print(f"  [PASS] Verified: Has 'from pathlib import Path'")

    # Find the sys.path setup section
    syspath_start = None
    syspath_end = None

    for i, line in enumerate(source):
        if 'Canonical sys.path setup' in line or '# Auto-detect project root' in line:
            syspath_start = i
        if syspath_start is not None and 'sys.path.insert' in line:
            # Find end of sys.path block (next blank line or next import)
            syspath_end = i + 1
            break

    if syspath_start is None or syspath_end is None:
        print(f"  [FAIL] ERROR: Could not find sys.path section")
        return False

    print(f"  [PASS] Found sys.path section: lines {syspath_start} to {syspath_end}")

    # Extract old logic for comparison
    old_logic = ''.join(source[syspath_start:syspath_end])
    print(f"\n  OLD LOGIC:")
    for line in source[syspath_start:syspath_end]:
        print(f"    {line.rstrip()}")

    # Replace the section
    before = source[:syspath_start]
    after = source[syspath_end:]

    # Split new block into lines (preserve existing line structure)
    new_syspath_lines = [line + '\n' for line in CORRECT_SYSPATH_BLOCK.split('\n')]

    # Construct new source
    nb['cells'][1]['source'] = before + new_syspath_lines + after

    print(f"\n  NEW LOGIC:")
    for line in new_syspath_lines[:15]:  # Show first 15 lines
        print(f"    {line.rstrip()}")
    print(f"    ... ({len(new_syspath_lines)} lines total)")

    # Save notebook
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n  [PASS] Notebook saved successfully")
    return True


def main():
    """Fix all 3 broken EDA notebooks."""
    print("="*70)
    print("FIXING EDA NOTEBOOKS 01-03: sys.path Detection Logic")
    print("="*70)

    project_root = Path('/home/sagemaker-user/RILA_6Y20B_refactored')

    # Only fix the 3 broken notebooks
    notebooks_to_fix = [
        project_root / 'notebooks/eda/rila_6y20b/01_EDA_sales_RILA.ipynb',
        project_root / 'notebooks/eda/rila_6y20b/02_EDA_rates_RILA.ipynb',
        project_root / 'notebooks/eda/rila_6y20b/03_EDA_RILA_feature_engineering.ipynb',
    ]

    results = []
    for nb_path in notebooks_to_fix:
        if not nb_path.exists():
            print(f"\n[FAIL] ERROR: Notebook not found: {nb_path}")
            results.append(False)
            continue

        success = fix_eda_notebook(nb_path)
        results.append(success)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for nb_path, success in zip(notebooks_to_fix, results):
        status = "[PASS] SUCCESS" if success else "[FAIL] FAILED"
        print(f"{status}: {nb_path.name}")

    total_success = sum(results)
    print(f"\nTotal: {total_success}/{len(notebooks_to_fix)} notebooks fixed successfully")

    if total_success == len(notebooks_to_fix):
        print("\n[PASS][PASS] ALL NOTEBOOKS FIXED SUCCESSFULLY [PASS][PASS]")
        return 0
    else:
        print("\n[FAIL][FAIL] SOME NOTEBOOKS FAILED [FAIL][FAIL]")
        return 1


if __name__ == '__main__':
    exit(main())
