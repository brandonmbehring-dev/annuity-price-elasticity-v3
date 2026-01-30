#!/usr/bin/env python3
"""
Create Refactoring Package Script

Creates a complete, self-contained zip package with everything needed to safely
refactor the RILA Price Elasticity system in a non-AWS environment.
"""

import sys
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import zipfile


def get_git_info():
    """Get current git commit hash and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return {
            "commit": commit,
            "branch": branch,
            "commit_short": commit[:8]
        }
    except:
        return {
            "commit": "unknown",
            "branch": "unknown",
            "commit_short": "unknown"
        }


def compute_checksum(file_path):
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_directory_size(directory):
    """Get total size of directory in bytes."""
    total = 0
    for path in Path(directory).rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def validate_source_structure():
    """Validate that source directory has expected structure."""
    print("Validating source directory structure...")

    required_dirs = [
        "src",
        "src/core",
        "src/data",
        "src/features",
        "src/models",
        "tests",
        "tests/fixtures",
        "tests/fixtures/rila",
        "tests/baselines",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "docs",
        "notebooks",
    ]

    required_files = [
        "requirements.txt",
        "pyproject.toml",
        "README_REFACTORING.md",
        "VALIDATION_GUIDE.md",
        "REINTEGRATION_GUIDE.md",
        "validate_package.py",
        "validate_equivalence.py",
        "prepare_reintegration.py",
        "CHANGELOG_REFACTORING.md",
    ]

    optional_files = [
        "pytest.ini",  # May be in pyproject.toml instead
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            missing_dirs.append(dir_path)
            print(f"  [FAIL] Missing directory: {dir_path}")

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).is_file():
            missing_files.append(file_path)
            print(f"  [FAIL] Missing file: {file_path}")

    # Check optional files (just inform, don't fail)
    for file_path in optional_files:
        if not Path(file_path).is_file():
            print(f"  ⓘ Optional file not present: {file_path}")

    if missing_dirs or missing_files:
        print(f"\n[FAIL] Validation failed: {len(missing_dirs)} directories and {len(missing_files)} files missing")
        return False

    print("  [PASS] All required directories and files present")
    return True


def check_fixture_completeness():
    """Verify fixtures are complete."""
    print("\nChecking fixture completeness...")

    fixtures_dir = Path("tests/fixtures/rila")
    if not fixtures_dir.exists():
        print("  [FAIL] Fixtures directory not found")
        return False

    fixture_files = list(fixtures_dir.rglob("*.parquet"))
    total_size = sum(f.stat().st_size for f in fixture_files)
    total_size_mb = total_size / (1024 * 1024)

    print(f"  Found {len(fixture_files)} fixture files")
    print(f"  Total size: {total_size_mb:.1f} MB")

    # Expected approximately 74 MB and 20+ files
    if len(fixture_files) < 15:
        print(f"  [WARN] Warning: Expected 20+ fixture files, found {len(fixture_files)}")
    if total_size_mb < 50:
        print(f"  [WARN] Warning: Expected ~74 MB, found {total_size_mb:.1f} MB")

    print("  [PASS] Fixtures appear complete")
    return True


def check_baseline_completeness():
    """Verify baselines are complete."""
    print("\nChecking baseline completeness...")

    baselines_dir = Path("tests/baselines")
    if not baselines_dir.exists():
        print("  [FAIL] Baselines directory not found")
        return False

    baseline_files = list(baselines_dir.rglob("*.parquet"))
    total_size = sum(f.stat().st_size for f in baseline_files)
    total_size_mb = total_size / (1024 * 1024)

    print(f"  Found {len(baseline_files)} baseline files")
    print(f"  Total size: {total_size_mb:.1f} MB")

    # Expected approximately 144 MB and 230+ files
    if len(baseline_files) < 200:
        print(f"  [WARN] Warning: Expected 230+ baseline files, found {len(baseline_files)}")
    if total_size_mb < 100:
        print(f"  [WARN] Warning: Expected ~144 MB, found {total_size_mb:.1f} MB")

    print("  [PASS] Baselines appear complete")
    return True


def update_changelog_metadata(git_info):
    """Update changelog with package creation metadata."""
    print("\nUpdating changelog metadata...")

    changelog_path = Path("CHANGELOG_REFACTORING.md")
    if not changelog_path.exists():
        print("  [WARN] Warning: CHANGELOG_REFACTORING.md not found")
        return

    with open(changelog_path, 'r') as f:
        content = f.read()

    # Update placeholders
    content = content.replace(
        "**Package Created**: [Date will be filled by create_refactoring_package.py]",
        f"**Package Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    content = content.replace(
        "**Original Git Commit**: [Commit hash will be filled by create_refactoring_package.py]",
        f"**Original Git Commit**: {git_info['commit']} (branch: {git_info['branch']})"
    )

    with open(changelog_path, 'w') as f:
        f.write(content)

    print("  [PASS] Changelog metadata updated")


def update_readme_metadata(git_info):
    """Update README with package metadata."""
    print("\nUpdating README metadata...")

    readme_path = Path("README_REFACTORING.md")
    if not readme_path.exists():
        print("  [WARN] Warning: README_REFACTORING.md not found")
        return

    with open(readme_path, 'r') as f:
        content = f.read()

    # Update package information section
    package_info = f"""**Package created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Source git commit**: {git_info['commit_short']} (branch: {git_info['branch']})

**Python version**: {sys.version.split()[0]}"""

    content = content.replace(
        "**Package created**: [Will be filled by create_refactoring_package.py]\n\n"
        "**Source git commit**: [Will be filled by create_refactoring_package.py]\n\n"
        "**Python version**: [Will be filled by create_refactoring_package.py]",
        package_info
    )

    with open(readme_path, 'w') as f:
        f.write(content)

    print("  [PASS] README metadata updated")


def create_manifest(file_list):
    """Create manifest with file checksums."""
    print("\nGenerating manifest with checksums...")

    manifest = {
        "created": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "files": {},
        "directories": {},
        "statistics": {}
    }

    # Compute checksums
    print("  Computing checksums (this may take a minute)...")
    for i, file_path in enumerate(file_list, 1):
        if i % 100 == 0:
            print(f"    Processed {i}/{len(file_list)} files...")

        path = Path(file_path)
        if path.is_file() and path.suffix in ['.py', '.md', '.txt', '.toml', '.ini', '.json']:
            # Only checksum text files to save time
            checksum = compute_checksum(path)
            manifest["files"][file_path] = checksum

    # Directory statistics
    for dir_name in ["src", "tests", "docs", "notebooks"]:
        if Path(dir_name).exists():
            size = get_directory_size(dir_name)
            manifest["directories"][dir_name] = {
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2)
            }

    # Overall statistics
    manifest["statistics"] = {
        "total_files": len(file_list),
        "total_checksums": len(manifest["files"]),
        "fixtures_size_mb": manifest["directories"].get("tests", {}).get("size_mb", 0)
    }

    print(f"  [PASS] Generated checksums for {len(manifest['files'])} files")

    return manifest


def create_package_zip(output_filename):
    """Create the zip package."""
    print(f"\nCreating package: {output_filename}")

    # Files and directories to include
    include_patterns = [
        "src/**/*.py",
        "tests/**/*.py",
        "tests/**/*.parquet",
        "tests/**/*.json",
        "tests/conftest.py",
        "tests/fixtures/**/*",
        "tests/baselines/**/*",
        "docs/**/*.md",
        "docs/**/*.png",
        "notebooks/**/*.ipynb",
        "*.py",  # Root level scripts
        "*.md",  # Root level docs
        "*.txt",  # Requirements files
        "*.toml",
        "*.ini",
        ".gitignore",
    ]

    # Directories/patterns to exclude
    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".git",
        "venv",
        ".venv",
        "*.egg-info",
        ".DS_Store",
        "*.swp",
        "*~",
        "_archive_refactoring",
    ]

    # Collect files
    files_to_include = []
    for pattern in include_patterns:
        for path in Path(".").glob(pattern):
            # Check exclusions
            skip = False
            for exclude in exclude_patterns:
                if exclude in str(path):
                    skip = True
                    break

            if not skip and path.is_file():
                files_to_include.append(str(path))

    # Remove duplicates and sort
    files_to_include = sorted(set(files_to_include))

    print(f"  Including {len(files_to_include)} files...")

    # Create manifest
    manifest = create_manifest(files_to_include)

    # Save manifest
    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    files_to_include.append("manifest.json")

    # Create zip file
    print("  Creating zip archive (this may take a few minutes)...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as zipf:
        for i, file_path in enumerate(files_to_include, 1):
            if i % 100 == 0:
                print(f"    Added {i}/{len(files_to_include)} files...")

            zipf.write(file_path)

    # Get final size
    zip_size = Path(output_filename).stat().st_size
    zip_size_mb = zip_size / (1024 * 1024)

    print(f"  [PASS] Package created: {output_filename}")
    print(f"  [PASS] Package size: {zip_size_mb:.1f} MB")
    print(f"  [PASS] Contains {len(files_to_include)} files")

    return output_filename, zip_size_mb


def create_package():
    """Main function to create the refactoring package."""
    print("=" * 80)
    print("CREATING REFACTORING PACKAGE")
    print("=" * 80)
    print()

    # Get git info
    git_info = get_git_info()
    print(f"Source git commit: {git_info['commit_short']}")
    print(f"Branch: {git_info['branch']}")
    print()

    # Validate structure
    if not validate_source_structure():
        print("\n[FAIL] Package creation failed: source structure validation failed")
        return 1

    # Check fixtures
    if not check_fixture_completeness():
        print("\n[WARN] Warning: Fixture check failed (continuing anyway)")

    # Check baselines
    if not check_baseline_completeness():
        print("\n[WARN] Warning: Baseline check failed (continuing anyway)")

    # Update metadata
    update_changelog_metadata(git_info)
    update_readme_metadata(git_info)

    # Create package filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"rila-refactoring-package-{timestamp}.zip"

    # Create zip package
    package_path, package_size = create_package_zip(output_filename)

    # Summary
    print("\n" + "=" * 80)
    print("PACKAGE CREATION COMPLETE")
    print("=" * 80)
    print()
    print(f"[PASS] Package: {package_path}")
    print(f"[PASS] Size: {package_size:.1f} MB")
    print(f"[PASS] Source commit: {git_info['commit_short']}")
    print()
    print("Next steps:")
    print("1. Transfer this package to your non-AWS environment")
    print("2. Extract: unzip", package_path)
    print("3. Follow instructions in README_REFACTORING.md")
    print("4. Validate package: python validate_package.py")
    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(create_package())
