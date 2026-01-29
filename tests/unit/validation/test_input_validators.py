"""
Tests for src.validation.input_validators module.

Tests canonical input validation functions with fail-fast behavior.
"""

import pytest
import pandas as pd
import numpy as np

from src.validation.input_validators import (
    validate_required_string,
    validate_dataframe_column,
    validate_target_variable,
)
from src.core.exceptions import DataValidationError


class TestValidateRequiredString:
    """Tests for validate_required_string function."""

    def test_valid_string_returns_none(self):
        """Valid non-empty string should return None (no error)."""
        result = validate_required_string("valid_value", "param_name")
        assert result is None

    def test_empty_string_raises_by_default(self):
        """Empty string should raise DataValidationError by default (fail-fast)."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_required_string("", "param_name")
        assert "param_name" in str(exc_info.value)
        assert "non-empty string" in str(exc_info.value)

    def test_empty_string_returns_error_when_not_raising(self):
        """Empty string should return error message when raise_on_error=False."""
        result = validate_required_string("", "param_name", raise_on_error=False)
        assert result is not None
        assert "CRITICAL" in result
        assert "param_name" in result

    def test_none_value_raises_by_default(self):
        """None value should raise DataValidationError by default."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_required_string(None, "param_name")
        assert "non-empty string" in str(exc_info.value)

    def test_none_value_returns_error_when_not_raising(self):
        """None value should return error message when raise_on_error=False."""
        result = validate_required_string(None, "param_name", raise_on_error=False)
        assert result is not None
        assert "CRITICAL" in result

    def test_non_string_raises_by_default(self):
        """Non-string value should raise DataValidationError by default."""
        with pytest.raises(DataValidationError):
            validate_required_string(123, "param_name")

    def test_non_string_returns_error_when_not_raising(self):
        """Non-string value should return error message when raise_on_error=False."""
        result = validate_required_string(123, "param_name", raise_on_error=False)
        assert result is not None
        assert "CRITICAL" in result

    def test_context_included_in_error(self):
        """Context should be included in error message when provided."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_required_string("", "param_name", context="for feature selection")
        assert "for feature selection" in str(exc_info.value)


class TestValidateDataframeColumn:
    """Tests for validate_dataframe_column function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "sales_target_current": [100, 200, 300],
            "competitor_rate_t1": [0.05, 0.06, 0.07],
            "string_column": ["a", "b", "c"],
        })

    def test_existing_column_returns_none(self, sample_df):
        """Existing column should return None (no error)."""
        result = validate_dataframe_column(sample_df, "sales_target_current")
        assert result is None

    def test_missing_column_raises_by_default(self, sample_df):
        """Missing column should raise DataValidationError by default (fail-fast)."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataframe_column(sample_df, "nonexistent_column")
        assert "nonexistent_column" in str(exc_info.value)

    def test_missing_column_returns_error_when_not_raising(self, sample_df):
        """Missing column should return error message when raise_on_error=False."""
        result = validate_dataframe_column(
            sample_df, "nonexistent_column", raise_on_error=False
        )
        assert result is not None
        assert "nonexistent_column" in result

    def test_numeric_requirement_passes_for_numeric(self, sample_df):
        """Numeric column should pass numeric requirement."""
        result = validate_dataframe_column(
            sample_df, "sales_target_current", require_numeric=True
        )
        assert result is None

    def test_numeric_requirement_raises_for_string(self, sample_df):
        """String column should raise DataValidationError for numeric requirement."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataframe_column(
                sample_df, "string_column", require_numeric=True
            )
        assert "numeric" in str(exc_info.value).lower()

    def test_numeric_requirement_returns_error_when_not_raising(self, sample_df):
        """String column should return error when raise_on_error=False."""
        result = validate_dataframe_column(
            sample_df, "string_column", require_numeric=True, raise_on_error=False
        )
        assert result is not None
        assert "numeric" in result.lower()


class TestValidateTargetVariable:
    """Tests for validate_target_variable function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "sales_target_current": [100.0, 200.0, 300.0],
            "sales_contract": [50.0, 100.0, 150.0],
            "date": pd.date_range("2024-01-01", periods=3),
        })

    def test_valid_target_raises_nothing(self, sample_df):
        """Valid target variable should not raise exception."""
        # Signature: validate_target_variable(target_variable, df=None, mode="raise")
        errors = validate_target_variable("sales_target_current", df=sample_df)
        assert len(errors) == 0

    def test_missing_target_raises_value_error(self, sample_df):
        """Missing target variable should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_target_variable("nonexistent_target", df=sample_df, mode="raise")
        assert "nonexistent_target" in str(exc_info.value)

    def test_warn_mode_returns_warnings(self, sample_df):
        """Warn mode should return list of warnings instead of raising."""
        warnings = validate_target_variable(
            "nonexistent_target", df=sample_df, mode="warn"
        )
        assert isinstance(warnings, list)
        assert len(warnings) > 0

    def test_empty_target_raises_in_raise_mode(self):
        """Empty target variable should raise ValueError in raise mode."""
        with pytest.raises(ValueError):
            validate_target_variable("", mode="raise")

    def test_empty_target_returns_warnings_in_warn_mode(self):
        """Empty target variable should return warnings in warn mode."""
        warnings = validate_target_variable("", mode="warn")
        assert len(warnings) > 0
        assert "CRITICAL" in warnings[0]
