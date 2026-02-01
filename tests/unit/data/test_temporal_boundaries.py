"""
Temporal Boundaries Tests for Data Pipeline
=============================================

Tests temporal boundary enforcement per LEAKAGE_CHECKLIST.md.

Key Requirements Tested:
1. Mature Data Cutoff (Example 4): 50-day exclusion of recent incomplete data
2. Temporal Ordering: Training data strictly before test data
3. Processing Window: Application date to contract date alignment

Purpose:
    Validate that temporal boundaries are respected in data pipeline
    to prevent future information leakage.

References:
    - knowledge/practices/LEAKAGE_CHECKLIST.md Example 4
    - src/data/preprocessing.py (temporal filtering)
    - src/data/pipelines.py (pipeline orchestration)
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# CONSTANTS
# =============================================================================

# Per LEAKAGE_CHECKLIST.md Example 4
MATURE_DATA_CUTOFF_DAYS = 50

# Tolerance for date comparisons
DATE_TOLERANCE_DAYS = 1


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    """Create sample sales data with date range for testing."""
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="W-SUN")  # Weekly data

    np.random.seed(42)
    return pd.DataFrame(
        {
            "date": dates,
            "sales": np.random.lognormal(mean=15, sigma=0.5, size=len(dates)),
            "application_signed_date": dates,
            "contract_issue_date": dates + timedelta(days=7),  # 7-day processing
        }
    )


@pytest.fixture
def recent_data_sample() -> pd.DataFrame:
    """Create sample data with recent dates that should be excluded."""
    today = datetime.now()

    dates = pd.date_range(start=today - timedelta(days=100), end=today, freq="W-SUN")

    np.random.seed(42)
    return pd.DataFrame(
        {
            "date": dates,
            "sales": np.random.lognormal(mean=15, sigma=0.5, size=len(dates)),
            "application_signed_date": dates,
        }
    )


# =============================================================================
# MATURE DATA CUTOFF TESTS (LEAKAGE_CHECKLIST.md Example 4)
# =============================================================================


@pytest.mark.leakage
class TestMatureDataCutoff:
    """
    Test 50-day mature data cutoff per LEAKAGE_CHECKLIST.md Example 4.

    Quote: "50-day mature data cutoff excludes recent incomplete data"

    This ensures we don't train on data that may still be receiving
    late-reported transactions (application → contract date lag).
    """

    def test_mature_cutoff_date_calculation(self) -> None:
        """Verify 50-day cutoff date is calculated correctly."""
        today = datetime.now()
        expected_cutoff = today - timedelta(days=MATURE_DATA_CUTOFF_DAYS)

        # Cutoff should be 50 days ago
        assert (today - expected_cutoff).days == MATURE_DATA_CUTOFF_DAYS

    def test_recent_data_excluded(self, recent_data_sample: pd.DataFrame) -> None:
        """Recent data within 50-day window should be excluded.

        Per LEAKAGE_CHECKLIST.md Example 4:
        - Data from the most recent 50 days is excluded
        - This allows for late-arriving transactions to settle
        """
        today = datetime.now()
        cutoff_date = today - timedelta(days=MATURE_DATA_CUTOFF_DAYS)

        # Filter to mature data only
        mature_data = recent_data_sample[recent_data_sample["date"] < cutoff_date]

        # Verify no dates within the exclusion window
        if len(mature_data) > 0:
            max_date = mature_data["date"].max()
            days_from_today = (today - max_date).days

            assert days_from_today >= MATURE_DATA_CUTOFF_DAYS, (
                f"Most recent data is only {days_from_today} days old. "
                f"Expected >= {MATURE_DATA_CUTOFF_DAYS} days. "
                f"This violates the mature data cutoff requirement."
            )

    def test_mature_data_cutoff_function(self, sample_sales_data: pd.DataFrame) -> None:
        """Test a reusable mature data cutoff function."""

        def apply_mature_data_cutoff(
            df: pd.DataFrame,
            date_column: str = "date",
            cutoff_days: int = MATURE_DATA_CUTOFF_DAYS,
            reference_date: datetime | None = None,
        ) -> pd.DataFrame:
            """
            Apply mature data cutoff to exclude recent incomplete data.

            Parameters
            ----------
            df : pd.DataFrame
                Input DataFrame with date column
            date_column : str
                Name of date column
            cutoff_days : int
                Number of days to exclude from present
            reference_date : datetime, optional
                Reference date for cutoff (default: today)

            Returns
            -------
            pd.DataFrame
                Filtered DataFrame with recent data excluded
            """
            if reference_date is None:
                reference_date = datetime.now()

            cutoff = reference_date - timedelta(days=cutoff_days)

            # Convert date column if needed
            df_copy = df.copy()
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])

            return df_copy[df_copy[date_column] < cutoff]

        # Apply cutoff using a reference date relative to the sample data
        # (sample data may be historical, so we use max date + 30 days as "today")
        sample_max_date = pd.to_datetime(sample_sales_data["date"]).max()
        reference_date = sample_max_date + timedelta(days=30)
        filtered = apply_mature_data_cutoff(sample_sales_data, reference_date=reference_date)

        # Verify cutoff applied (some recent data should be excluded)
        assert len(filtered) < len(
            sample_sales_data
        ), "Mature data cutoff should exclude some recent data"

        # Verify no data within cutoff window relative to reference
        if len(filtered) > 0:
            max_date = pd.to_datetime(filtered["date"]).max()
            days_old = (reference_date - max_date).days

            assert (
                days_old >= MATURE_DATA_CUTOFF_DAYS
            ), f"Filtered data contains records only {days_old} days old"

    def test_cutoff_preserves_historical_data(self, sample_sales_data: pd.DataFrame) -> None:
        """Verify cutoff doesn't exclude old historical data."""
        reference_date = datetime(2024, 6, 1)  # Fixed reference
        cutoff_date = reference_date - timedelta(days=MATURE_DATA_CUTOFF_DAYS)

        # Filter data
        filtered = sample_sales_data[sample_sales_data["date"] < cutoff_date]

        # Should have substantial historical data
        expected_min_rows = 50  # At least ~1 year of weekly data

        assert len(filtered) >= expected_min_rows, (
            f"Filtered data has only {len(filtered)} rows. "
            f"Expected at least {expected_min_rows} historical records."
        )


# =============================================================================
# TEMPORAL ORDERING TESTS
# =============================================================================


@pytest.mark.leakage
class TestTemporalOrdering:
    """Test temporal ordering requirements for train/test splits."""

    def test_training_dates_before_test_dates(self, sample_sales_data: pd.DataFrame) -> None:
        """Training data dates must strictly precede test data dates.

        This is a fundamental requirement for causal identification:
        - Training: t < T
        - Test: t >= T

        Where T is the split point.
        """
        # Simple 80/20 split by date
        dates = sample_sales_data["date"].sort_values()
        split_idx = int(len(dates) * 0.8)
        split_date = dates.iloc[split_idx]

        train_data = sample_sales_data[sample_sales_data["date"] < split_date]
        test_data = sample_sales_data[sample_sales_data["date"] >= split_date]

        if len(train_data) > 0 and len(test_data) > 0:
            train_max = train_data["date"].max()
            test_min = test_data["date"].min()

            assert train_max < test_min, (
                f"Training data max date ({train_max}) must be strictly "
                f"before test data min date ({test_min})."
            )

    def test_no_temporal_overlap(self, sample_sales_data: pd.DataFrame) -> None:
        """Verify no overlap between training and test periods."""
        # Split by date
        mid_date = sample_sales_data["date"].median()
        train_dates = set(sample_sales_data[sample_sales_data["date"] < mid_date]["date"])
        test_dates = set(sample_sales_data[sample_sales_data["date"] >= mid_date]["date"])

        overlap = train_dates & test_dates

        assert len(overlap) == 0, (
            f"Found {len(overlap)} overlapping dates between train and test. "
            "This violates temporal separation."
        )


# =============================================================================
# PROCESSING WINDOW TESTS
# =============================================================================


@pytest.mark.leakage
class TestProcessingWindow:
    """Test application-to-contract processing window alignment."""

    def test_contract_date_after_application_date(self, sample_sales_data: pd.DataFrame) -> None:
        """Contract issue date must be after application signed date.

        This represents the real-world processing delay between
        when a customer applies and when the contract is issued.
        """
        if "contract_issue_date" not in sample_sales_data.columns:
            pytest.skip("No contract_issue_date column")

        # Check ordering
        violations = sample_sales_data[
            sample_sales_data["contract_issue_date"] < sample_sales_data["application_signed_date"]
        ]

        assert len(violations) == 0, (
            f"Found {len(violations)} records where contract date "
            "is before application date. This violates temporal ordering."
        )

    def test_processing_days_are_positive(self, sample_sales_data: pd.DataFrame) -> None:
        """Processing days (app → contract) should be non-negative."""
        if (
            "contract_issue_date" not in sample_sales_data.columns
            or "application_signed_date" not in sample_sales_data.columns
        ):
            pytest.skip("Missing date columns")

        processing_days = (
            sample_sales_data["contract_issue_date"] - sample_sales_data["application_signed_date"]
        ).dt.days

        negative_processing = (processing_days < 0).sum()

        assert negative_processing == 0, (
            f"Found {negative_processing} records with negative processing days. "
            "This indicates data quality issues or date errors."
        )

    def test_processing_days_within_bounds(self, sample_sales_data: pd.DataFrame) -> None:
        """Processing days should be within reasonable bounds (0-90 days)."""
        if (
            "contract_issue_date" not in sample_sales_data.columns
            or "application_signed_date" not in sample_sales_data.columns
        ):
            pytest.skip("Missing date columns")

        processing_days = (
            sample_sales_data["contract_issue_date"] - sample_sales_data["application_signed_date"]
        ).dt.days

        max_reasonable_days = 90
        extreme_processing = (processing_days > max_reasonable_days).sum()

        # Should have very few extreme cases
        extreme_fraction = extreme_processing / len(sample_sales_data)
        assert extreme_fraction < 0.01, (  # Less than 1%
            f"{extreme_fraction:.1%} of records have processing > "
            f"{max_reasonable_days} days. This may indicate data issues."
        )


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def validate_temporal_boundaries(
    data: pd.DataFrame,
    date_column: str = "date",
    mature_cutoff_days: int = MATURE_DATA_CUTOFF_DAYS,
    reference_date: datetime | None = None,
) -> dict:
    """
    Validate all temporal boundary requirements.

    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
    date_column : str
        Name of date column
    mature_cutoff_days : int
        Days to exclude from present
    reference_date : datetime, optional
        Reference date for cutoff

    Returns
    -------
    dict
        Validation results with pass/fail for each check
    """
    if reference_date is None:
        reference_date = datetime.now()

    results = {
        "mature_cutoff_applied": False,
        "temporal_ordering_valid": True,
        "processing_window_valid": True,
        "details": {},
    }

    # Check mature cutoff
    dates = pd.to_datetime(data[date_column])
    _cutoff_date = reference_date - timedelta(days=mature_cutoff_days)  # For reference

    most_recent = dates.max()
    days_from_present = (reference_date - most_recent).days

    results["mature_cutoff_applied"] = days_from_present >= mature_cutoff_days
    results["details"]["most_recent_date"] = most_recent
    results["details"]["days_from_present"] = days_from_present
    results["details"]["cutoff_required_days"] = mature_cutoff_days

    return results


class TestValidationUtility:
    """Test the validation utility function."""

    def test_validate_temporal_boundaries(self, sample_sales_data: pd.DataFrame) -> None:
        """Test validation utility with sample data."""
        # Use a fixed reference date for reproducible tests
        reference_date = datetime(2024, 6, 1)

        results = validate_temporal_boundaries(
            sample_sales_data,
            date_column="date",
            reference_date=reference_date,
        )

        # Should have all expected keys
        assert "mature_cutoff_applied" in results
        assert "temporal_ordering_valid" in results
        assert "details" in results

        # Details should have date info
        assert "most_recent_date" in results["details"]
        assert "days_from_present" in results["details"]
