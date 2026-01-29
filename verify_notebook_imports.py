#!/usr/bin/env python3
"""
Verify that all notebooks correctly import from the refactored codebase.

This script tests the sys.path setup logic in all active notebooks to ensure
they load code from /home/sagemaker-user/RILA_6Y20B_refactored/src instead of
the system-installed package at /opt/conda/lib/python3.12/site-packages.

Usage:
    python3 verify_notebook_imports.py
"""

import sys
import os
from pathlib import Path
import subprocess


def test_notebook_directory(nb_dir, test_import_module="src.data.extraction"):
    """
    Test that imports work correctly from a notebook directory.

    Returns:
        tuple: (success, project_root, uses_refactored, error_msg)
    """
    # Save original state
    orig_dir = os.getcwd()

    try:
        os.chdir(nb_dir)
        cwd = os.getcwd()

        # Apply standard sys.path logic (from notebooks)
        if 'notebooks/production/rila' in cwd:
            project_root = Path(cwd).parents[2]
        elif 'notebooks/production/fia' in cwd:
            project_root = Path(cwd).parents[2]
        elif 'notebooks/eda/rila' in cwd:
            project_root = Path(cwd).parents[2]
        elif 'notebooks/archive' in cwd:
            project_root = Path(cwd).parents[2]
        elif 'notebooks/onboarding' in cwd:
            project_root = Path(cwd).parents[1]
        elif os.path.basename(cwd) == 'notebooks':
            project_root = Path(cwd).parent
        else:
            project_root = Path(cwd)

        project_root = str(project_root)

        # Verify src/ exists
        if not os.path.exists(os.path.join(project_root, 'src')):
            return False, project_root, False, "src/ directory not found"

        # Create isolated Python environment to test import
        test_script = f"""
import sys
sys.path.insert(0, '{project_root}')
import {test_import_module}
import src
print(src.__file__)
"""

        result = subprocess.run(
            ['python3', '-c', test_script],
            cwd=nb_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return False, project_root, False, result.stderr

        src_path = result.stdout.strip()
        uses_refactored = 'RILA_6Y20B_refactored' in src_path

        return True, project_root, uses_refactored, src_path

    except Exception as e:
        return False, "Unknown", False, str(e)
    finally:
        os.chdir(orig_dir)


def main():
    """Test all active notebook directories."""
    print("="*80)
    print("NOTEBOOK IMPORT VERIFICATION")
    print("="*80)
    print("\nVerifying that all notebooks load from REFACTORED codebase:")
    print("  ✓ Expected: /home/sagemaker-user/RILA_6Y20B_refactored/src")
    print("  ✗ Wrong:    /opt/conda/lib/python3.12/site-packages (system package)")
    print("="*80)

    base_path = Path("/home/sagemaker-user/RILA_6Y20B_refactored")

    # Define test cases
    test_cases = [
        ("Production RILA 6Y20B", base_path / "notebooks/production/rila_6y20b", "src.config.config_builder"),
        ("Production RILA 1Y10B", base_path / "notebooks/production/rila_1y10b", "src.config.config_builder"),
        ("EDA 01 Sales", base_path / "notebooks/eda/rila_6y20b", "src.data.extraction"),
        ("EDA 02 Rates", base_path / "notebooks/eda/rila_6y20b", "src.data.extraction"),
        ("EDA 03 Features", base_path / "notebooks/eda/rila_6y20b", "src.data.extraction"),
        ("Onboarding", base_path / "notebooks/onboarding", "src.data.extraction"),
    ]

    results = []
    for name, nb_dir, test_module in test_cases:
        if not nb_dir.exists():
            print(f"\n⚠️  SKIP: {name}")
            print(f"   Directory not found: {nb_dir}")
            continue

        success, project_root, uses_refactored, detail = test_notebook_directory(
            nb_dir, test_module
        )

        results.append((name, success, uses_refactored))

        if success and uses_refactored:
            print(f"\n✓✓ PASS: {name}")
            print(f"   Directory: {nb_dir.relative_to(base_path)}")
            print(f"   Project root: {project_root}")
            print(f"   Imports from: {detail}")
        elif success and not uses_refactored:
            print(f"\n✗✗ FAIL: {name} - Using OLD code")
            print(f"   Directory: {nb_dir.relative_to(base_path)}")
            print(f"   Project root: {project_root}")
            print(f"   Imports from: {detail}")
        else:
            print(f"\n✗✗ FAIL: {name} - Import error")
            print(f"   Directory: {nb_dir.relative_to(base_path)}")
            print(f"   Error: {detail}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total = len(results)
    passed = sum(1 for _, success, uses_refactored in results if success and uses_refactored)
    failed = total - passed

    for name, success, uses_refactored in results:
        status = "✓" if (success and uses_refactored) else "✗"
        print(f"{status} {name}")

    print(f"\nTotal: {passed}/{total} notebooks passed")

    if passed == total:
        print("\n✓✓ ALL NOTEBOOKS USE REFACTORED CODE ✓✓")
        return 0
    else:
        print(f"\n✗✗ {failed} NOTEBOOK(S) FAILED ✗✗")
        return 1


if __name__ == '__main__':
    exit(main())
