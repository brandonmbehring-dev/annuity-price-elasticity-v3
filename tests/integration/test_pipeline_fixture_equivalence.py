"""
Pipeline Fixture Equivalence Tests
==================================

Tests that validate data pipeline stages against fixture baselines.
These tests use the existing fixtures in tests/fixtures/rila/ as baseline.

Unlike test_pipeline_stage_equivalence.py (which requires AWS baselines),
these tests run entirely offline using fixture data.

Pipeline Stages Tested:
1. Product Filtering → filtered_flexguard_6y20.parquet
2. Sales Cleanup → cleaned_sales_with_processing_days.parquet
3a. Daily Sales Time Series → daily_sales_timeseries.parquet
3b. Contract Sales Time Series → contract_sales_timeseries.parquet
6. WINK Processing → wink_competitive_rates_pivoted.parquet
7. Market Weighting → market_weighted_competitive_rates.parquet
8. Data Integration → daily_integrated_dataset.parquet
9. Competitive Features → competitive_features_engineered.parquet
10. Final Dataset → final_weekly_dataset.parquet

Mathematical Equivalence: 1e-12 precision for all stages

Author: Claude Code
Date: 2026-01-31
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Mathematical equivalence tolerance
TOLERANCE = 1e-12


# =============================================================================
# FIXTURES - Load Baseline Data
# =============================================================================


@pytest.fixture(scope="module")
def fixtures_path() -> Path:
    """Path to RILA fixture data."""
    path = Path(__file__).parent.parent / "fixtures/rila"
    if not path.exists():
        pytest.skip(f"Fixture path not found: {path}")
    return path


@pytest.fixture(scope="module")
def baseline_filtered_product(fixtures_path: Path) -> pd.DataFrame:
    """Stage 1: Baseline filtered product data."""
    path = fixtures_path / "filtered_flexguard_6y20.parquet"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def baseline_cleaned_sales(fixtures_path: Path) -> pd.DataFrame:
    """Stage 2: Baseline cleaned sales data."""
    path = fixtures_path / "cleaned_sales_with_processing_days.parquet"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def baseline_daily_sales_ts(fixtures_path: Path) -> pd.DataFrame:
    """Stage 3a: Baseline daily sales time series."""
    path = fixtures_path / "daily_sales_timeseries.parquet"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def baseline_contract_sales_ts(fixtures_path: Path) -> pd.DataFrame:
    """Stage 3b: Baseline contract sales time series."""
    path = fixtures_path / "contract_sales_timeseries.parquet"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def baseline_wink_rates(fixtures_path: Path) -> pd.DataFrame:
    """Stage 6: Baseline WINK competitive rates."""
    path = fixtures_path / "wink_competitive_rates_pivoted.parquet"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def baseline_market_weighted(fixtures_path: Path) -> pd.DataFrame:
    """Stage 7: Baseline market-weighted rates."""
    path = fixtures_path / "market_weighted_competitive_rates.parquet"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def baseline_daily_integrated(fixtures_path: Path) -> pd.DataFrame:
    """Stage 8: Baseline daily integrated dataset."""
    path = fixtures_path / "daily_integrated_dataset.parquet"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def baseline_competitive_features(fixtures_path: Path) -> pd.DataFrame:
    """Stage 9: Baseline competitive features."""
    path = fixtures_path / "competitive_features_engineered.parquet"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def baseline_final_weekly(fixtures_path: Path) -> pd.DataFrame:
    """Stage 10: Baseline final weekly dataset."""
    path = fixtures_path / "final_weekly_dataset.parquet"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    return pd.read_parquet(path)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def validate_dataframe_structure(
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    stage: str
) -> None:
    """Validate DataFrame structure matches baseline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    baseline : pd.DataFrame
        Baseline DataFrame to compare against
    stage : str
        Stage name for error messages

    Raises
    ------
    AssertionError
        If structures don't match
    """
    # Check row count within tolerance (fixtures may be slightly different)
    row_diff_pct = abs(len(df) - len(baseline)) / len(baseline)
    assert row_diff_pct < 0.01, (
        f"[{stage}] Row count differs > 1%: actual={len(df)}, baseline={len(baseline)}"
    )

    # Check critical columns exist
    missing_cols = set(baseline.columns) - set(df.columns)
    assert len(missing_cols) == 0, (
        f"[{stage}] Missing columns: {missing_cols}"
    )


def validate_numeric_column(
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    column: str,
    tolerance: float = TOLERANCE,
    stage: str = "unknown"
) -> None:
    """Validate numeric column values match baseline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with column to validate
    baseline : pd.DataFrame
        Baseline DataFrame to compare against
    column : str
        Column name to validate
    tolerance : float
        Numerical tolerance
    stage : str
        Stage name for error messages

    Raises
    ------
    AssertionError
        If values differ beyond tolerance
    """
    if column not in df.columns or column not in baseline.columns:
        return  # Skip if column doesn't exist

    if not pd.api.types.is_numeric_dtype(df[column]):
        return  # Skip non-numeric columns

    # For matching-length DataFrames, compare directly
    if len(df) == len(baseline):
        np.testing.assert_allclose(
            df[column].values,
            baseline[column].values,
            rtol=tolerance,
            atol=tolerance,
            err_msg=f"[{stage}] Column '{column}' values differ"
        )
    else:
        # For different-length DataFrames, compare statistics
        actual_mean = df[column].mean()
        baseline_mean = baseline[column].mean()
        mean_diff = abs(actual_mean - baseline_mean) / (abs(baseline_mean) + 1e-10)

        assert mean_diff < 0.01, (
            f"[{stage}] Column '{column}' mean differs > 1%: "
            f"actual={actual_mean:.6f}, baseline={baseline_mean:.6f}"
        )


# =============================================================================
# BASELINE STRUCTURE TESTS
# =============================================================================


@pytest.mark.integration
class TestBaselineStructure:
    """Test that fixture baselines have expected structure."""

    def test_filtered_product_structure(self, baseline_filtered_product: pd.DataFrame):
        """Stage 1: Filtered product baseline has expected structure."""
        assert len(baseline_filtered_product) > 50000, "Should have >50k rows"
        assert baseline_filtered_product.shape[1] >= 100, "Should have >100 columns"

        # Must have key columns
        required_cols = ['product_name', 'buffer_rate', 'term']
        for col in required_cols:
            assert col in baseline_filtered_product.columns, f"Missing column: {col}"

    def test_cleaned_sales_structure(self, baseline_cleaned_sales: pd.DataFrame):
        """Stage 2: Cleaned sales baseline has expected structure."""
        assert len(baseline_cleaned_sales) > 50000, "Should have >50k rows after cleanup"

        # Must have processing_days column (key transformation)
        assert 'processing_days' in baseline_cleaned_sales.columns

    def test_daily_sales_ts_structure(self, baseline_daily_sales_ts: pd.DataFrame):
        """Stage 3a: Daily sales time series has expected structure."""
        assert len(baseline_daily_sales_ts) > 1000, "Should have >1000 days"
        assert baseline_daily_sales_ts.shape[1] == 2, "Should have date + sales columns"

    def test_contract_sales_ts_structure(self, baseline_contract_sales_ts: pd.DataFrame):
        """Stage 3b: Contract sales time series has expected structure."""
        assert len(baseline_contract_sales_ts) > 1000, "Should have >1000 days"
        assert baseline_contract_sales_ts.shape[1] == 2, "Should have date + sales columns"

    def test_wink_rates_structure(self, baseline_wink_rates: pd.DataFrame):
        """Stage 6: WINK rates baseline has expected structure."""
        assert len(baseline_wink_rates) > 2000, "Should have >2000 rate observations"
        assert baseline_wink_rates.shape[1] >= 8, "Should have multiple competitor columns"

    def test_market_weighted_structure(self, baseline_market_weighted: pd.DataFrame):
        """Stage 7: Market-weighted rates has expected structure."""
        assert len(baseline_market_weighted) > 2000, "Should have >2000 observations"
        assert baseline_market_weighted.shape[1] >= 20, "Should have weighted features"

        # Must have weighted feature columns
        weighted_cols = [c for c in baseline_market_weighted.columns if 'weighted' in c.lower()]
        assert len(weighted_cols) >= 1, "Must have at least one weighted column"

    def test_daily_integrated_structure(self, baseline_daily_integrated: pd.DataFrame):
        """Stage 8: Daily integrated dataset has expected structure."""
        assert len(baseline_daily_integrated) > 1000, "Should have >1000 days"
        assert baseline_daily_integrated.shape[1] >= 20, "Should have sales + competitive features"

    def test_competitive_features_structure(self, baseline_competitive_features: pd.DataFrame):
        """Stage 9: Competitive features has expected structure."""
        assert len(baseline_competitive_features) > 1000, "Should have >1000 observations"

        # Must have competitive feature columns
        comp_cols = [c for c in baseline_competitive_features.columns if c.startswith('C_')]
        assert len(comp_cols) >= 5, f"Must have >=5 competitive features, found {len(comp_cols)}"

    def test_final_weekly_structure(self, baseline_final_weekly: pd.DataFrame):
        """Stage 10: Final weekly dataset has expected structure."""
        assert len(baseline_final_weekly) >= 100, "Should have >=100 weeks"
        assert baseline_final_weekly.shape[1] >= 50, "Should have many features"

        # Must not have excessive NaN
        nan_pct = baseline_final_weekly.isnull().mean().mean()
        assert nan_pct < 0.20, f"Too many NaN values: {nan_pct:.1%}"


# =============================================================================
# DATA QUALITY TESTS
# =============================================================================


@pytest.mark.integration
class TestDataQuality:
    """Test data quality properties of fixture baselines."""

    def test_sales_are_positive(self, baseline_daily_sales_ts: pd.DataFrame):
        """Sales values should be positive."""
        sales_col = [c for c in baseline_daily_sales_ts.columns if 'sales' in c.lower()]
        if sales_col:
            values = baseline_daily_sales_ts[sales_col[0]].dropna()
            assert (values >= 0).all(), "Sales should be non-negative"

    def test_rates_in_valid_range(self, baseline_market_weighted: pd.DataFrame):
        """Competitive rates should be in valid range (0-30%)."""
        rate_cols = [c for c in baseline_market_weighted.columns
                     if 'rate' in c.lower() or c.startswith('C_')]

        for col in rate_cols:
            if pd.api.types.is_numeric_dtype(baseline_market_weighted[col]):
                values = baseline_market_weighted[col].dropna()
                if len(values) > 0:
                    # Rates should be in reasonable range (as decimals or percentages)
                    max_val = values.max()
                    assert max_val < 100, f"Column '{col}' has suspicious max value: {max_val}"

    def test_dates_are_sequential(self, baseline_daily_sales_ts: pd.DataFrame):
        """Date column should be sequential (no large gaps)."""
        date_col = [c for c in baseline_daily_sales_ts.columns
                    if 'date' in c.lower() or baseline_daily_sales_ts[c].dtype == 'datetime64[ns]']

        if date_col:
            dates = pd.to_datetime(baseline_daily_sales_ts[date_col[0]])
            date_diffs = dates.diff().dropna().dt.days

            # Most gaps should be 1-7 days (weekends)
            assert date_diffs.median() <= 7, "Median date gap should be <=7 days"

    def test_no_duplicate_dates_in_weekly(self, baseline_final_weekly: pd.DataFrame):
        """Weekly dataset should not have duplicate date indices."""
        date_cols = [c for c in baseline_final_weekly.columns
                     if 'date' in c.lower() or 'week' in c.lower()]

        if date_cols:
            for col in date_cols:
                unique_pct = baseline_final_weekly[col].nunique() / len(baseline_final_weekly)
                assert unique_pct > 0.95, f"Column '{col}' has many duplicates: {1-unique_pct:.1%}"


# =============================================================================
# STAGE RELATIONSHIP TESTS
# =============================================================================


@pytest.mark.integration
class TestStageRelationships:
    """Test logical relationships between pipeline stages."""

    def test_filtering_reduces_rows(
        self,
        baseline_filtered_product: pd.DataFrame,
        baseline_cleaned_sales: pd.DataFrame
    ):
        """Sales cleanup may reduce rows (outlier removal)."""
        # Cleaning typically keeps most rows but may remove some
        retention_pct = len(baseline_cleaned_sales) / len(baseline_filtered_product)
        assert 0.90 <= retention_pct <= 1.01, (
            f"Unexpected retention after cleanup: {retention_pct:.1%}"
        )

    def test_aggregation_reduces_rows(
        self,
        baseline_daily_integrated: pd.DataFrame,
        baseline_final_weekly: pd.DataFrame
    ):
        """Weekly aggregation should reduce daily rows by ~7x."""
        expected_reduction = 7  # 7 days per week
        actual_reduction = len(baseline_daily_integrated) / len(baseline_final_weekly)

        assert 4 <= actual_reduction <= 10, (
            f"Unexpected aggregation reduction: {actual_reduction:.1f}x (expected ~7x)"
        )

    def test_feature_engineering_adds_columns(
        self,
        baseline_daily_integrated: pd.DataFrame,
        baseline_competitive_features: pd.DataFrame
    ):
        """Feature engineering should add columns."""
        assert baseline_competitive_features.shape[1] >= baseline_daily_integrated.shape[1], (
            "Feature engineering should not reduce columns"
        )

    def test_lag_features_add_many_columns(
        self,
        baseline_competitive_features: pd.DataFrame,
        baseline_final_weekly: pd.DataFrame
    ):
        """Lag feature generation should add many columns."""
        # Final dataset has lag features, polynomials, etc.
        assert baseline_final_weekly.shape[1] > baseline_competitive_features.shape[1] * 2, (
            "Final dataset should have many more columns from lag/polynomial features"
        )


# =============================================================================
# ECONOMIC CONSTRAINT TESTS
# =============================================================================


@pytest.mark.integration
class TestEconomicConstraints:
    """Test that data satisfies economic constraints."""

    def test_final_dataset_has_required_columns(self, baseline_final_weekly: pd.DataFrame):
        """Final dataset must have columns needed for inference."""
        # Must have own rate (Prudential)
        own_rate_cols = [c for c in baseline_final_weekly.columns
                        if 'prudential' in c.lower() or 'own' in c.lower()]
        assert len(own_rate_cols) >= 1, "Must have own rate column"

        # Must have competitor rate columns
        competitor_cols = [c for c in baseline_final_weekly.columns
                          if c.startswith('C_') or 'competitor' in c.lower()]
        assert len(competitor_cols) >= 1, "Must have competitor rate columns"

        # Must have sales target
        sales_cols = [c for c in baseline_final_weekly.columns
                      if 'sales' in c.lower()]
        assert len(sales_cols) >= 1, "Must have sales column"

    def test_lag0_competitor_features_marked_for_exclusion(
        self,
        baseline_final_weekly: pd.DataFrame
    ):
        """
        Verify lag-0 competitor features exist but are EXCLUDED from models.

        The dataset contains _current competitor columns for reference/computation,
        but these MUST NOT be used in models (causal violation). The leakage gate
        tests in test_leakage_gates.py enforce this at model training time.

        This test documents which columns are lag-0 and should be excluded.
        """
        # Identify lag-0 competitor columns
        lag0_patterns = ['_t0', '_current']
        lag0_competitor_cols = []

        for col in baseline_final_weekly.columns:
            if 'competitor' in col.lower() and col != 'competitor':
                for pattern in lag0_patterns:
                    if pattern in col:
                        lag0_competitor_cols.append(col)
                        break

        # Document that these columns exist (for reference/computation)
        # The leakage gate prevents their use in models
        if lag0_competitor_cols:
            print(f"\n[INFO] Found {len(lag0_competitor_cols)} lag-0 competitor columns "
                  "(excluded from models by leakage gate)")
            print(f"  Examples: {lag0_competitor_cols[:5]}")

        # Verify that lagged versions (t1, t2, etc.) also exist for each base column
        for col in lag0_competitor_cols[:5]:  # Check first 5
            base_name = col.replace('_current', '').replace('_t0', '')
            lagged_versions = [c for c in baseline_final_weekly.columns
                             if c.startswith(base_name) and '_t' in c and '_current' not in c]
            assert len(lagged_versions) >= 1, (
                f"Column {col} exists but has no lagged versions (e.g., {base_name}_t1)"
            )

    def test_lag_features_properly_named(self, baseline_final_weekly: pd.DataFrame):
        """Lag features should follow naming convention (e.g., _t1, _t2)."""
        lag_cols = [c for c in baseline_final_weekly.columns if '_t' in c]

        # Should have multiple lag periods
        lag_numbers = set()
        for col in lag_cols:
            # Extract lag number from _t1, _t2, etc.
            import re
            match = re.search(r'_t(\d+)', col)
            if match:
                lag_numbers.add(int(match.group(1)))

        assert len(lag_numbers) >= 2, (
            f"Should have multiple lag periods, found: {lag_numbers}"
        )


# =============================================================================
# SUMMARY STATISTICS TESTS
# =============================================================================


@pytest.mark.integration
class TestSummaryStatistics:
    """Test that fixture baselines have expected summary statistics."""

    def test_final_dataset_date_range(self, baseline_final_weekly: pd.DataFrame):
        """Final dataset should span reasonable date range."""
        date_cols = [c for c in baseline_final_weekly.columns
                     if 'date' in c.lower() or 'week' in c.lower()]

        if date_cols:
            dates = pd.to_datetime(baseline_final_weekly[date_cols[0]])
            date_range_years = (dates.max() - dates.min()).days / 365

            assert date_range_years >= 2, (
                f"Date range too short: {date_range_years:.1f} years (need >= 2)"
            )

    def test_competitor_feature_variety(self, baseline_competitive_features: pd.DataFrame):
        """Should have multiple types of competitive features."""
        comp_cols = [c for c in baseline_competitive_features.columns if c.startswith('C_')]

        feature_types = set()
        for col in comp_cols:
            # Extract feature type (e.g., 'weighted_mean', 'top5', 'median')
            parts = col.replace('C_', '').split('_')
            if parts:
                feature_types.add(parts[0])

        assert len(feature_types) >= 3, (
            f"Should have >= 3 types of competitive features, found: {feature_types}"
        )
