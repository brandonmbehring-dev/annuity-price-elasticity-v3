"""
Fixture Validity Tests.

These tests validate that fixture data is:
- Fresh (< 90 days old)
- Complete (all required files exist)
- High quality (schema, size, missing values within acceptable limits)
- Properly documented (metadata present and valid)

These tests should run in CI to detect stale or corrupted fixtures early.

Usage:
    # Run all fixture validation tests
    pytest tests/fixtures/test_fixture_validity.py -v

    # Run only freshness check
    pytest tests/fixtures/test_fixture_validity.py::TestFixtureFreshness -v

    # Run only completeness check
    pytest tests/fixtures/test_fixture_validity.py::TestFixtureCompleteness -v

Fixture Refresh:
    If tests fail due to stale fixtures, run the fixture refresh script:
    python tests/fixtures/refresh_fixtures.py
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime


# Fixture base path
FIXTURE_PATH = Path("tests/fixtures/rila")


class TestFixtureFreshness:
    """
    Tests for fixture freshness validation.

    Fixtures should be refreshed quarterly (every 90 days) to ensure
    they reflect current production data.
    """

    def test_metadata_file_exists(self):
        """Verify fixture metadata file exists."""
        metadata_path = FIXTURE_PATH / "refresh_metadata.json"

        assert metadata_path.exists(), (
            f"Fixture metadata missing: {metadata_path}. "
            f"Run fixture refresh script: python tests/fixtures/refresh_fixtures.py"
        )

        print(f"✓ Fixture metadata exists: {metadata_path}")

    def test_metadata_is_valid_json(self):
        """Verify metadata file is valid JSON."""
        metadata_path = FIXTURE_PATH / "refresh_metadata.json"

        if not metadata_path.exists():
            pytest.skip("Metadata file doesn't exist")

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            assert isinstance(metadata, dict), "Metadata must be a dictionary"
            print(f"✓ Metadata is valid JSON")

        except json.JSONDecodeError as e:
            pytest.fail(f"Metadata is not valid JSON: {e}")

    def test_fixtures_not_too_old(self):
        """
        Verify fixtures are not too old (< 90 days).

        Stale fixtures may diverge from production data and should be
        refreshed to maintain equivalence.
        """
        metadata_path = FIXTURE_PATH / "refresh_metadata.json"

        if not metadata_path.exists():
            pytest.skip("Metadata file doesn't exist")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check if refresh_date exists
        assert 'refresh_date' in metadata, (
            "Metadata missing 'refresh_date' field. "
            "Re-run fixture refresh script."
        )

        # Parse refresh date
        try:
            refresh_date = datetime.fromisoformat(metadata['refresh_date'])
        except ValueError as e:
            pytest.fail(f"Invalid refresh_date format: {e}")

        # Calculate age
        days_old = (datetime.now() - refresh_date).days

        # Warn if > 60 days, fail if > 90 days
        if days_old > 60:
            print(f"⚠ Fixtures are {days_old} days old (consider refreshing)")

        assert days_old < 90, (
            f"Fixtures are {days_old} days old (max: 90 days). "
            f"Refresh fixtures with: python tests/fixtures/refresh_fixtures.py"
        )

        print(f"✓ Fixtures are {days_old} days old (< 90 days)")

    def test_metadata_has_required_fields(self):
        """Verify metadata contains all required fields."""
        metadata_path = FIXTURE_PATH / "refresh_metadata.json"

        if not metadata_path.exists():
            pytest.skip("Metadata file doesn't exist")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        required_fields = ['refresh_date', 'data_shape']
        missing_fields = [f for f in required_fields if f not in metadata]

        assert not missing_fields, (
            f"Metadata missing required fields: {missing_fields}. "
            f"Re-run fixture refresh script."
        )

        print(f"✓ Metadata has all required fields: {required_fields}")


class TestFixtureCompleteness:
    """
    Tests for fixture completeness validation.

    All required fixture files must exist for offline development.
    """

    @pytest.fixture(scope="class")
    def required_files(self):
        """List of required fixture files."""
        return [
            "raw_sales_data.parquet",
            "raw_wink_data.parquet",
            "market_share_weights.parquet",
            "final_weekly_dataset.parquet",
            "filtered_flexguard_6y20.parquet",
            "cleaned_sales_with_processing_days.parquet",
            "daily_sales_timeseries.parquet",
            "contract_sales_timeseries.parquet",
        ]

    def test_all_required_files_exist(self, required_files):
        """Verify all required fixture files exist."""
        missing_files = []

        for filename in required_files:
            filepath = FIXTURE_PATH / filename
            if not filepath.exists():
                missing_files.append(filename)

        assert not missing_files, (
            f"Required fixtures missing: {missing_files}. "
            f"Run fixture refresh: python tests/fixtures/refresh_fixtures.py"
        )

        print(f"✓ All {len(required_files)} required fixture files exist")

    def test_economic_indicators_directory_exists(self):
        """Verify economic indicators directory exists."""
        econ_dir = FIXTURE_PATH / "economic_indicators"

        assert econ_dir.exists(), (
            f"Economic indicators directory missing: {econ_dir}. "
            f"Run fixture refresh."
        )

        assert econ_dir.is_dir(), (
            f"Economic indicators path exists but is not a directory: {econ_dir}"
        )

        print(f"✓ Economic indicators directory exists: {econ_dir}")

    def test_economic_indicators_not_empty(self):
        """Verify economic indicators directory has files."""
        econ_dir = FIXTURE_PATH / "economic_indicators"

        if not econ_dir.exists():
            pytest.skip("Economic indicators directory doesn't exist")

        # List parquet files
        indicator_files = list(econ_dir.glob("*.parquet"))

        assert len(indicator_files) > 0, (
            f"Economic indicators directory is empty: {econ_dir}. "
            f"Run fixture refresh."
        )

        print(f"✓ Economic indicators: {len(indicator_files)} files")

    def test_fixture_files_not_empty(self, required_files):
        """Verify fixture files are not empty (> 0 bytes)."""
        empty_files = []

        for filename in required_files:
            filepath = FIXTURE_PATH / filename
            if filepath.exists() and filepath.stat().st_size == 0:
                empty_files.append(filename)

        assert not empty_files, (
            f"Fixture files are empty (0 bytes): {empty_files}. "
            f"Re-run fixture refresh."
        )

        print(f"✓ All fixture files have data (> 0 bytes)")


class TestFixtureDataQuality:
    """
    Tests for fixture data quality validation.

    Fixtures should pass basic quality checks to ensure they are
    suitable for development and testing.
    """

    def test_sales_data_quality(self):
        """Validate sales data quality."""
        filepath = FIXTURE_PATH / "raw_sales_data.parquet"

        if not filepath.exists():
            pytest.skip("Sales data fixture doesn't exist")

        # Load data
        sales = pd.read_parquet(filepath)

        # Check size
        assert len(sales) > 10000, (
            f"Sales data too small: {len(sales)} rows (expected > 10,000). "
            f"Fixture may be incomplete."
        )

        # Check required columns (basic schema validation)
        expected_columns = [
            'application_signed_date',
            'product_name',
            'contract_initial_premium_amount',
        ]

        missing_cols = [col for col in expected_columns if col not in sales.columns]

        assert not missing_cols, (
            f"Sales data missing expected columns: {missing_cols}"
        )

        # Check missing values (should be < 1% for critical columns)
        premium_col = 'contract_initial_premium_amount'
        if premium_col in sales.columns:
            missing_premium_pct = sales[premium_col].isna().sum() / len(sales) * 100

            assert missing_premium_pct < 1.0, (
                f"Too many missing premiums: {missing_premium_pct:.2f}% (max: 1%). "
                f"Data quality issue detected."
            )
        else:
            missing_premium_pct = 0.0

        print(f"✓ Sales data quality: {sales.shape}, {missing_premium_pct:.3f}% missing premiums")

    def test_wink_data_quality(self):
        """Validate WINK competitive rates data quality."""
        filepath = FIXTURE_PATH / "raw_wink_data.parquet"

        if not filepath.exists():
            pytest.skip("WINK data fixture doesn't exist")

        # Load data
        wink = pd.read_parquet(filepath)

        # Check size
        assert len(wink) > 5000, (
            f"WINK data too small: {len(wink)} rows (expected > 5,000). "
            f"Fixture may be incomplete."
        )

        # Check for numeric rate columns
        numeric_cols = wink.select_dtypes(include=['number']).columns

        assert len(numeric_cols) > 0, (
            "WINK data has no numeric columns. Schema may be corrupted."
        )

        print(f"✓ WINK data quality: {wink.shape}, {len(numeric_cols)} numeric columns")

    def test_final_weekly_dataset_quality(self):
        """Validate final weekly dataset quality."""
        filepath = FIXTURE_PATH / "final_weekly_dataset.parquet"

        if not filepath.exists():
            pytest.skip("Final weekly dataset fixture doesn't exist")

        # Load data
        final_df = pd.read_parquet(filepath)

        # Expected shape based on RILA 6Y20B baseline
        # Note: Actual fixtures have 203 weeks (updated dataset)
        expected_min_weeks = 150  # At least 150 weeks of data
        expected_min_features = 100  # Should have many features

        assert len(final_df) >= expected_min_weeks, (
            f"Final dataset has {len(final_df)} weeks (expected >= {expected_min_weeks}). "
            f"Data may be incomplete or incorrectly aggregated."
        )

        assert final_df.shape[1] >= expected_min_features, (
            f"Final dataset has {final_df.shape[1]} features (expected >= {expected_min_features}). "
            f"Feature engineering may have failed."
        )

        # Check for NaN values (should be minimal in final dataset)
        total_values = final_df.shape[0] * final_df.shape[1]
        nan_count = final_df.isna().sum().sum()
        nan_pct = nan_count / total_values * 100

        # Allow up to 5% NaN in final dataset (some features may have legitimate NaN)
        assert nan_pct < 5.0, (
            f"Final dataset has {nan_pct:.2f}% NaN values (max: 5%). "
            f"Data quality issue detected."
        )

        print(f"✓ Final weekly dataset quality: {final_df.shape}, {nan_pct:.3f}% NaN")

    def test_market_weights_quality(self):
        """Validate market weights data quality."""
        filepath = FIXTURE_PATH / "market_share_weights.parquet"

        if not filepath.exists():
            pytest.skip("Market weights fixture doesn't exist")

        # Load data
        weights = pd.read_parquet(filepath)

        # Check size
        assert len(weights) > 0, "Market weights data is empty"

        # Check for weight columns (should be between 0 and 1)
        numeric_cols = weights.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            if 'weight' in col.lower() or 'share' in col.lower():
                # Weights should be between 0 and 1
                assert weights[col].min() >= 0, (
                    f"Market weight column '{col}' has negative values"
                )
                assert weights[col].max() <= 1, (
                    f"Market weight column '{col}' has values > 1"
                )

        print(f"✓ Market weights quality: {weights.shape}")


class TestFixtureMetadataIntegrity:
    """
    Tests for fixture metadata integrity.

    Metadata should accurately reflect the fixture data that was captured.
    """

    def test_metadata_data_shapes_match_actual(self):
        """
        Verify metadata data_shape matches actual fixture files.

        This ensures metadata is in sync with fixture files.
        """
        metadata_path = FIXTURE_PATH / "refresh_metadata.json"

        if not metadata_path.exists():
            pytest.skip("Metadata file doesn't exist")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if 'data_shape' not in metadata:
            pytest.skip("Metadata missing data_shape field")

        data_shapes = metadata['data_shape']

        # Check sales shape
        if 'sales' in data_shapes:
            sales_path = FIXTURE_PATH / "raw_sales_data.parquet"
            if sales_path.exists():
                actual_sales = pd.read_parquet(sales_path)
                expected_shape = tuple(data_shapes['sales'])

                assert actual_sales.shape == expected_shape, (
                    f"Sales data shape mismatch. "
                    f"Metadata: {expected_shape}, Actual: {actual_sales.shape}. "
                    f"Re-run fixture refresh."
                )

        # Check rates shape
        if 'rates' in data_shapes:
            rates_path = FIXTURE_PATH / "raw_wink_data.parquet"
            if rates_path.exists():
                actual_rates = pd.read_parquet(rates_path)
                expected_shape = tuple(data_shapes['rates'])

                assert actual_rates.shape == expected_shape, (
                    f"WINK rates shape mismatch. "
                    f"Metadata: {expected_shape}, Actual: {actual_rates.shape}. "
                    f"Re-run fixture refresh."
                )

        print(f"✓ Metadata data_shape matches actual fixture files")


# Summary test for CI

def test_fixtures_ci_ready():
    """
    Summary test for CI pipeline.

    This test provides a single pass/fail for fixture validity in CI.
    If this test passes, fixtures are ready for development.
    """
    issues = []

    # Check freshness
    metadata_path = FIXTURE_PATH / "refresh_metadata.json"
    if not metadata_path.exists():
        issues.append("Metadata file missing")
    else:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            refresh_date = datetime.fromisoformat(metadata['refresh_date'])
            days_old = (datetime.now() - refresh_date).days

            if days_old > 90:
                issues.append(f"Fixtures too old ({days_old} days)")

        except (KeyError, ValueError, json.JSONDecodeError):
            issues.append("Metadata invalid or corrupted")

    # Check completeness
    required_files = [
        "raw_sales_data.parquet",
        "raw_wink_data.parquet",
        "final_weekly_dataset.parquet",
    ]

    for filename in required_files:
        if not (FIXTURE_PATH / filename).exists():
            issues.append(f"Missing required file: {filename}")

    # Report result
    if issues:
        pytest.fail(
            f"Fixture validity issues detected:\n" +
            "\n".join(f"  - {issue}" for issue in issues) +
            f"\n\nRun fixture refresh: python tests/fixtures/refresh_fixtures.py"
        )

    print(f"✓ Fixtures CI-ready: All validation checks passed")
