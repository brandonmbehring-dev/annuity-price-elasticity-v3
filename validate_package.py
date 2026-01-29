#!/usr/bin/env python3
"""
Package Integrity Validation Script

Validates that the refactoring package was extracted correctly and
all required files are present with correct checksums.
"""

import sys
import json
import hashlib
from pathlib import Path


def compute_checksum(file_path):
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_directory_structure():
    """Verify expected directory structure exists."""
    required_dirs = [
        "src",
        "tests",
        "tests/fixtures",
        "tests/fixtures/rila",
        "tests/baselines",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "tests/performance",
        "tests/property_based",
        "docs",
        "notebooks",
    ]

    print("Checking directory structure...")
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            missing.append(dir_path)
            print(f"  ✗ Missing: {dir_path}")
        else:
            print(f"  ✓ Found: {dir_path}")

    return len(missing) == 0, missing


def check_key_files():
    """Verify key files are present."""
    required_files = [
        "requirements.txt",
        "pyproject.toml",
        "README_REFACTORING.md",
        "VALIDATION_GUIDE.md",
        "REINTEGRATION_GUIDE.md",
        "validate_equivalence.py",
        "prepare_reintegration.py",
        "CHANGELOG_REFACTORING.md",
    ]

    optional_files = [
        "pytest.ini",  # May be configured in pyproject.toml
    ]

    print("\nChecking key files...")
    missing = []
    for file_path in required_files:
        if not Path(file_path).is_file():
            missing.append(file_path)
            print(f"  ✗ Missing: {file_path}")
        else:
            print(f"  ✓ Found: {file_path}")

    # Check optional files (informational only)
    for file_path in optional_files:
        if not Path(file_path).is_file():
            print(f"  ⓘ Optional file not present: {file_path}")
        else:
            print(f"  ✓ Found: {file_path}")

    return len(missing) == 0, missing


def check_fixtures():
    """Verify fixture files are present and reasonable size."""
    fixtures_dir = Path("tests/fixtures/rila")

    if not fixtures_dir.exists():
        print("\n✗ Fixtures directory not found")
        return False, []

    print("\nChecking fixtures...")
    fixture_files = list(fixtures_dir.rglob("*.parquet"))

    if len(fixture_files) == 0:
        print("  ✗ No fixture files found")
        return False, []

    total_size = sum(f.stat().st_size for f in fixture_files)
    total_size_mb = total_size / (1024 * 1024)

    print(f"  ✓ Found {len(fixture_files)} fixture files")
    print(f"  ✓ Total size: {total_size_mb:.1f} MB")

    # Expected approximately 74 MB
    if total_size_mb < 50:
        print(f"  ⚠ Warning: Fixture size seems small (expected ~74 MB)")
    elif total_size_mb > 100:
        print(f"  ⚠ Warning: Fixture size seems large (expected ~74 MB)")

    return True, []


def check_baselines():
    """Verify baseline files are present and reasonable size."""
    baselines_dir = Path("tests/baselines")

    if not baselines_dir.exists():
        print("\n✗ Baselines directory not found")
        return False, []

    print("\nChecking baselines...")
    baseline_files = list(baselines_dir.rglob("*.parquet"))

    if len(baseline_files) == 0:
        print("  ✗ No baseline files found")
        return False, []

    total_size = sum(f.stat().st_size for f in baseline_files)
    total_size_mb = total_size / (1024 * 1024)

    print(f"  ✓ Found {len(baseline_files)} baseline files")
    print(f"  ✓ Total size: {total_size_mb:.1f} MB")

    # Expected approximately 144 MB
    if total_size_mb < 100:
        print(f"  ⚠ Warning: Baseline size seems small (expected ~144 MB)")
    elif total_size_mb > 200:
        print(f"  ⚠ Warning: Baseline size seems large (expected ~144 MB)")

    return True, []


def check_manifest():
    """Verify manifest.json exists and validate checksums if present."""
    manifest_path = Path("manifest.json")

    if not manifest_path.exists():
        print("\n⚠ Warning: manifest.json not found (checksum validation skipped)")
        return True, []

    print("\nValidating checksums from manifest...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    if "files" not in manifest:
        print("  ⚠ Warning: No file checksums in manifest")
        return True, []

    mismatches = []
    checked = 0
    for file_path, expected_checksum in manifest["files"].items():
        path = Path(file_path)
        if path.exists():
            actual_checksum = compute_checksum(path)
            if actual_checksum != expected_checksum:
                mismatches.append(file_path)
                print(f"  ✗ Checksum mismatch: {file_path}")
            else:
                checked += 1

    if len(mismatches) == 0:
        print(f"  ✓ All {checked} checksums validated")
        return True, []
    else:
        print(f"  ✗ {len(mismatches)} checksum mismatches found")
        return False, mismatches


def test_imports():
    """Test that key modules can be imported."""
    print("\nTesting imports...")
    print("  Note: Import errors are expected until dependencies are installed")
    print("        and PYTHONPATH is configured. This is informational only.")
    print()

    test_imports = [
        "src.core.price_elasticity_pipeline",
        "src.data.adapters.sales_data_adapter",
        "src.features.feature_engineering",
        "src.models.ridge_regression_wrapper",
    ]

    import_errors = []
    for module_name in test_imports:
        try:
            __import__(module_name)
            print(f"  ✓ Import successful: {module_name}")
        except Exception as e:
            import_errors.append((module_name, str(e)))
            print(f"  ⓘ Import not yet working: {module_name}")

    if len(import_errors) > 0:
        print()
        print("  To fix import errors:")
        print("    1. pip install -r requirements.txt")
        print("    2. pip install -r requirements-dev.txt")
        print("    3. export PYTHONPATH=\"${PYTHONPATH}:$(pwd)\"")

    # Don't fail validation based on imports (expected to fail initially)
    return True, []


def validate_package():
    """Run all validation checks."""
    print("=" * 80)
    print("PACKAGE INTEGRITY VALIDATION")
    print("=" * 80)
    print()

    results = {
        "timestamp": Path.cwd().as_posix(),
        "checks": {}
    }

    # Run all checks
    checks = [
        ("directory_structure", check_directory_structure),
        ("key_files", check_key_files),
        ("fixtures", check_fixtures),
        ("baselines", check_baselines),
        ("manifest", check_manifest),
        ("imports", test_imports),
    ]

    all_passed = True
    for check_name, check_func in checks:
        passed, issues = check_func()
        results["checks"][check_name] = {
            "passed": passed,
            "issues": issues
        }
        if not passed:
            all_passed = False

    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for check_name, check_result in results["checks"].items():
        passed = check_result["passed"]
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {check_name.replace('_', ' ').title()}: {'PASSED' if passed else 'FAILED'}")

    print()

    if all_passed:
        print("RESULT: Package Validation PASSED ✓")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install dev dependencies: pip install -r requirements-dev.txt")
        print("3. Run test suite: pytest -v")
        print("4. Start refactoring!")
        exit_code = 0
    else:
        print("RESULT: Package Validation FAILED ✗")
        print()
        print("Issues found. Review errors above.")
        print("Package may be incomplete or corrupted.")
        exit_code = 1

    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    sys.exit(validate_package())
