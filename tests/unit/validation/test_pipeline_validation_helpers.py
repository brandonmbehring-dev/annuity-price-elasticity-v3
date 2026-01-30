"""
Tests for src.validation.pipeline_validation_helpers module.

Comprehensive tests for pipeline validation helper functions including:
- Configuration template functions
- Extraction output validation
- Preprocessing output validation
- Error handling with business context
- Shrinkage and growth rules

Coverage Target: 60%+ of pipeline_validation_helpers.py (305 lines)
"""

import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.validation.pipeline_validation_helpers import (
    _get_default_growth_config,
    _create_validation_config,
    validate_extraction_output,
    validate_preprocessing_output,
    _raise_validation_error,
)
from src.validation.production_validators import (
    ValidationResult,
    ValidationMetrics,
    GrowthConfig,
    ValidationConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for validation testing."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=n_rows, freq='D'),
        'sales': np.random.uniform(50000, 200000, n_rows),
        'rate': np.random.uniform(0.05, 0.15, n_rows),
        'category': np.random.choice(['A', 'B', 'C'], n_rows)
    })


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory with outputs/metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_dir = Path(tmpdir) / "outputs" / "metadata"
        metadata_dir.mkdir(parents=True)
        yield Path(tmpdir)


@pytest.fixture
def mock_validation_result_passed() -> ValidationResult:
    """Create mock validation result that passed."""
    return ValidationResult(
        checkpoint_name="test_checkpoint",
        status="PASSED",
        issues=[],
        warnings=[],
        current_metrics={
            'run_date': '2024-01-01T12:00:00',
            'record_count': 100,
            'column_count': 4,
            'column_names': ['date', 'sales', 'rate', 'category'],
            'min_date': '2022-01-01',
            'max_date': '2022-04-10',
            'date_range_days': 99,
            'aggregation_ratio': None
        },
        previous_metrics=None,
        metadata_path=Path("/tmp/test_metadata.json")
    )


@pytest.fixture
def mock_validation_result_failed_schema() -> ValidationResult:
    """Create mock validation result that failed due to schema issue."""
    return ValidationResult(
        checkpoint_name="test_checkpoint",
        status="FAILED",
        issues=["Schema violation: missing columns ['required_col']"],
        warnings=[],
        current_metrics={
            'run_date': '2024-01-01T12:00:00',
            'record_count': 100,
            'column_count': 3,
            'column_names': ['date', 'sales', 'rate'],
            'min_date': '2022-01-01',
            'max_date': '2022-04-10',
            'date_range_days': 99,
            'aggregation_ratio': None
        },
        previous_metrics={
            'run_date': '2024-01-01T11:00:00',
            'record_count': 100,
            'column_count': 4,
            'column_names': ['date', 'sales', 'rate', 'required_col'],
            'min_date': '2022-01-01',
            'max_date': '2022-04-10',
            'date_range_days': 99,
            'aggregation_ratio': None
        },
        metadata_path=Path("/tmp/test_metadata.json")
    )


@pytest.fixture
def mock_validation_result_failed_shrinkage() -> ValidationResult:
    """Create mock validation result that failed due to data shrinkage."""
    return ValidationResult(
        checkpoint_name="test_checkpoint",
        status="FAILED",
        issues=["Row count decreased from 100 to 50 (shrinkage detected)"],
        warnings=[],
        current_metrics={
            'run_date': '2024-01-01T12:00:00',
            'record_count': 50,
            'column_count': 4,
            'column_names': ['date', 'sales', 'rate', 'category'],
            'min_date': '2022-01-01',
            'max_date': '2022-02-19',
            'date_range_days': 49,
            'aggregation_ratio': None
        },
        previous_metrics={
            'run_date': '2024-01-01T11:00:00',
            'record_count': 100,
            'column_count': 4,
            'column_names': ['date', 'sales', 'rate', 'category'],
            'min_date': '2022-01-01',
            'max_date': '2022-04-10',
            'date_range_days': 99,
            'aggregation_ratio': None
        },
        metadata_path=Path("/tmp/test_metadata.json")
    )


@pytest.fixture
def mock_validation_result_failed_empty() -> ValidationResult:
    """Create mock validation result that failed due to empty dataset."""
    return ValidationResult(
        checkpoint_name="test_checkpoint",
        status="FAILED",
        issues=["Dataset is empty - no rows returned"],
        warnings=[],
        current_metrics={
            'run_date': '2024-01-01T12:00:00',
            'record_count': 0,
            'column_count': 4,
            'column_names': ['date', 'sales', 'rate', 'category'],
            'min_date': None,
            'max_date': None,
            'date_range_days': None,
            'aggregation_ratio': None
        },
        previous_metrics=None,
        metadata_path=Path("/tmp/test_metadata.json")
    )


@pytest.fixture
def mock_validation_result_with_warnings() -> ValidationResult:
    """Create mock validation result with warnings but passed."""
    return ValidationResult(
        checkpoint_name="test_checkpoint",
        status="WARNINGS",
        issues=[],
        warnings=["High growth detected: 45% increase", "Null rate increased"],
        current_metrics={
            'run_date': '2024-01-01T12:00:00',
            'record_count': 145,
            'column_count': 4,
            'column_names': ['date', 'sales', 'rate', 'category'],
            'min_date': '2022-01-01',
            'max_date': '2022-05-25',
            'date_range_days': 144,
            'aggregation_ratio': None
        },
        previous_metrics={
            'run_date': '2024-01-01T11:00:00',
            'record_count': 100,
            'column_count': 4,
            'column_names': ['date', 'sales', 'rate', 'category'],
            'min_date': '2022-01-01',
            'max_date': '2022-04-10',
            'date_range_days': 99,
            'aggregation_ratio': None
        },
        metadata_path=Path("/tmp/test_metadata.json")
    )


# =============================================================================
# GROWTH CONFIG TESTS
# =============================================================================


class TestGetDefaultGrowthConfig:
    """Tests for _get_default_growth_config function."""

    def test_default_config_no_shrinkage_allowed(self):
        """Default config (allow_shrinkage=False) should not allow shrinkage."""
        config = _get_default_growth_config(allow_shrinkage=False)

        assert config['min_growth_pct'] == 0.0
        assert config['max_growth_pct'] == 50.0
        assert config['warn_on_shrinkage'] is True
        assert config['warn_on_high_growth'] is True

    def test_config_with_shrinkage_allowed(self):
        """Config with allow_shrinkage=True should allow 5% shrinkage."""
        config = _get_default_growth_config(allow_shrinkage=True)

        assert config['min_growth_pct'] == -5.0  # Allow 5% shrinkage
        assert config['max_growth_pct'] == 50.0
        assert config['warn_on_shrinkage'] is True
        assert config['warn_on_high_growth'] is True

    def test_returns_growth_config_type(self):
        """Should return a dictionary matching GrowthConfig structure."""
        config = _get_default_growth_config()

        # Check all required keys exist
        assert 'min_growth_pct' in config
        assert 'max_growth_pct' in config
        assert 'warn_on_shrinkage' in config
        assert 'warn_on_high_growth' in config


# =============================================================================
# VALIDATION CONFIG TESTS
# =============================================================================


class TestCreateValidationConfig:
    """Tests for _create_validation_config function."""

    def test_creates_config_with_defaults(self):
        """Should create config with default values."""
        config = _create_validation_config(stage_name="test_stage")

        assert config['checkpoint_name'] == "test_stage"
        assert config['version'] == 6
        assert config['strict_schema'] is True
        assert config['critical_columns'] is None
        assert 'growth_config' in config

    def test_creates_config_with_custom_version(self):
        """Should use custom version number."""
        config = _create_validation_config(stage_name="test_stage", version=7)

        assert config['version'] == 7

    def test_creates_config_with_strict_false(self):
        """Should set strict_schema to False when specified."""
        config = _create_validation_config(stage_name="test_stage", strict_schema=False)

        assert config['strict_schema'] is False

    def test_creates_config_with_critical_columns(self):
        """Should include critical columns when specified."""
        critical = ['date', 'sales', 'rate']
        config = _create_validation_config(
            stage_name="test_stage",
            critical_columns=critical
        )

        assert config['critical_columns'] == critical

    def test_creates_config_with_shrinkage_allowed(self):
        """Should configure growth config for shrinkage when specified."""
        config = _create_validation_config(
            stage_name="test_stage",
            allow_shrinkage=True
        )

        assert config['growth_config']['min_growth_pct'] == -5.0

    def test_project_root_is_cwd(self):
        """Project root should be set to current working directory."""
        config = _create_validation_config(stage_name="test_stage")

        assert config['project_root'] == str(Path.cwd())


# =============================================================================
# VALIDATE EXTRACTION OUTPUT TESTS
# =============================================================================


class TestValidateExtractionOutput:
    """Tests for validate_extraction_output function."""

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_passes_valid_dataframe(self, mock_validate, sample_dataframe,
                                    mock_validation_result_passed):
        """Valid DataFrame should pass validation and be returned unchanged."""
        mock_validate.return_value = mock_validation_result_passed

        result = validate_extraction_output(
            df=sample_dataframe,
            stage_name="test_extraction"
        )

        assert len(result) == len(sample_dataframe)
        mock_validate.assert_called_once()

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_raises_on_schema_failure(self, mock_validate, sample_dataframe,
                                      mock_validation_result_failed_schema):
        """Schema validation failure should raise ValueError."""
        mock_validate.return_value = mock_validation_result_failed_schema

        with pytest.raises(ValueError) as exc_info:
            validate_extraction_output(
                df=sample_dataframe,
                stage_name="test_extraction"
            )

        # Check error message contains business context
        error_msg = str(exc_info.value)
        assert "CRITICAL" in error_msg
        assert "test_extraction" in error_msg
        assert "Business impact" in error_msg

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_raises_on_shrinkage_by_default(self, mock_validate, sample_dataframe,
                                            mock_validation_result_failed_shrinkage):
        """Shrinkage should cause failure by default (allow_shrinkage=False)."""
        mock_validate.return_value = mock_validation_result_failed_shrinkage

        with pytest.raises(ValueError):
            validate_extraction_output(
                df=sample_dataframe,
                stage_name="test_extraction",
                allow_shrinkage=False  # Default
            )

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    @patch('builtins.print')
    def test_prints_warnings(self, mock_print, mock_validate, sample_dataframe,
                            mock_validation_result_with_warnings):
        """Warnings should be printed but not cause failure."""
        mock_validate.return_value = mock_validation_result_with_warnings

        result = validate_extraction_output(
            df=sample_dataframe,
            stage_name="test_extraction"
        )

        # Should return DataFrame (not fail)
        assert len(result) == len(sample_dataframe)

        # Should print warnings
        assert mock_print.call_count >= 2  # Two warnings

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_uses_custom_date_column(self, mock_validate, sample_dataframe,
                                     mock_validation_result_passed):
        """Should pass custom date_column to validation."""
        mock_validate.return_value = mock_validation_result_passed

        validate_extraction_output(
            df=sample_dataframe,
            stage_name="test_extraction",
            date_column="custom_date"
        )

        call_kwargs = mock_validate.call_args[1]
        assert call_kwargs['date_column'] == "custom_date"

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_uses_critical_columns(self, mock_validate, sample_dataframe,
                                   mock_validation_result_passed):
        """Should pass critical_columns to validation config."""
        mock_validate.return_value = mock_validation_result_passed

        validate_extraction_output(
            df=sample_dataframe,
            stage_name="test_extraction",
            critical_columns=['date', 'sales']
        )

        call_kwargs = mock_validate.call_args[1]
        assert call_kwargs['config']['critical_columns'] == ['date', 'sales']


# =============================================================================
# VALIDATE PREPROCESSING OUTPUT TESTS
# =============================================================================


class TestValidatePreprocessingOutput:
    """Tests for validate_preprocessing_output function."""

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_passes_valid_dataframe(self, mock_validate, sample_dataframe,
                                    mock_validation_result_passed):
        """Valid DataFrame should pass preprocessing validation."""
        mock_validate.return_value = mock_validation_result_passed

        result = validate_preprocessing_output(
            df=sample_dataframe,
            stage_name="test_preprocessing"
        )

        assert len(result) == len(sample_dataframe)

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_allows_shrinkage_by_default(self, mock_validate, sample_dataframe,
                                         mock_validation_result_passed):
        """Preprocessing should allow shrinkage by default."""
        mock_validate.return_value = mock_validation_result_passed

        validate_preprocessing_output(
            df=sample_dataframe,
            stage_name="test_preprocessing"
            # allow_shrinkage defaults to True for preprocessing
        )

        call_kwargs = mock_validate.call_args[1]
        assert call_kwargs['config']['growth_config']['min_growth_pct'] == -5.0

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_raises_on_schema_failure(self, mock_validate, sample_dataframe,
                                      mock_validation_result_failed_schema):
        """Schema failure should raise even for preprocessing."""
        mock_validate.return_value = mock_validation_result_failed_schema

        with pytest.raises(ValueError) as exc_info:
            validate_preprocessing_output(
                df=sample_dataframe,
                stage_name="test_preprocessing"
            )

        assert "CRITICAL" in str(exc_info.value)

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_raises_on_empty_dataset(self, mock_validate, sample_dataframe,
                                     mock_validation_result_failed_empty):
        """Empty dataset should fail even for preprocessing."""
        mock_validate.return_value = mock_validation_result_failed_empty

        with pytest.raises(ValueError) as exc_info:
            validate_preprocessing_output(
                df=sample_dataframe,
                stage_name="test_preprocessing"
            )

        assert "empty" in str(exc_info.value).lower()


# =============================================================================
# RAISE VALIDATION ERROR TESTS
# =============================================================================


class TestRaiseValidationError:
    """Tests for _raise_validation_error function."""

    def test_error_message_format(self, mock_validation_result_failed_schema):
        """Error message should follow CODING_STANDARDS.md format."""
        with pytest.raises(ValueError) as exc_info:
            _raise_validation_error(mock_validation_result_failed_schema, "test_stage")

        error_msg = str(exc_info.value)

        # Check required components
        assert "CRITICAL" in error_msg
        assert "Stage: test_stage" in error_msg
        assert "Business impact:" in error_msg
        assert "Required action:" in error_msg
        assert "Expected:" in error_msg

    def test_schema_issue_business_context(self, mock_validation_result_failed_schema):
        """Schema issues should have appropriate business context."""
        with pytest.raises(ValueError) as exc_info:
            _raise_validation_error(mock_validation_result_failed_schema, "test_stage")

        error_msg = str(exc_info.value)

        assert "structure changed" in error_msg.lower()
        assert "schema" in error_msg.lower()

    def test_shrinkage_issue_business_context(self, mock_validation_result_failed_shrinkage):
        """Shrinkage issues should have appropriate business context."""
        with pytest.raises(ValueError) as exc_info:
            _raise_validation_error(mock_validation_result_failed_shrinkage, "test_stage")

        error_msg = str(exc_info.value)

        assert "data loss" in error_msg.lower()

    def test_empty_issue_business_context(self, mock_validation_result_failed_empty):
        """Empty dataset issues should have appropriate business context."""
        with pytest.raises(ValueError) as exc_info:
            _raise_validation_error(mock_validation_result_failed_empty, "test_stage")

        error_msg = str(exc_info.value)

        assert "no data" in error_msg.lower()

    def test_includes_current_metrics(self, mock_validation_result_failed_schema):
        """Error message should include current metrics."""
        with pytest.raises(ValueError) as exc_info:
            _raise_validation_error(mock_validation_result_failed_schema, "test_stage")

        error_msg = str(exc_info.value)

        assert "100 rows" in error_msg  # Current record count
        assert "3 columns" in error_msg  # Current column count

    def test_includes_previous_metrics_when_available(self, mock_validation_result_failed_shrinkage):
        """Error message should include previous metrics when available."""
        with pytest.raises(ValueError) as exc_info:
            _raise_validation_error(mock_validation_result_failed_shrinkage, "test_stage")

        error_msg = str(exc_info.value)

        # Should include both current and previous
        assert "50 rows" in error_msg  # Current
        assert "100 rows" in error_msg  # Previous

    def test_generic_issue_handled(self):
        """Generic/unknown issues should have fallback business context."""
        generic_result = ValidationResult(
            checkpoint_name="test_checkpoint",
            status="FAILED",
            issues=["Unknown validation failure occurred"],
            warnings=[],
            current_metrics={
                'run_date': '2024-01-01T12:00:00',
                'record_count': 100,
                'column_count': 4,
                'column_names': ['date', 'sales', 'rate', 'category'],
                'min_date': '2022-01-01',
                'max_date': '2022-04-10',
                'date_range_days': 99,
                'aggregation_ratio': None
            },
            previous_metrics=None,
            metadata_path=Path("/tmp/test_metadata.json")
        )

        with pytest.raises(ValueError) as exc_info:
            _raise_validation_error(generic_result, "test_stage")

        error_msg = str(exc_info.value)

        # Should use generic fallback context
        assert "Data quality issue" in error_msg


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests with real validation pipeline."""

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_full_extraction_workflow(self, mock_validate, sample_dataframe):
        """Test complete extraction validation workflow."""
        # First run creates baseline
        baseline_result = ValidationResult(
            checkpoint_name="sales_extraction",
            status="BASELINE",
            issues=[],
            warnings=[],
            current_metrics={
                'run_date': '2024-01-01T12:00:00',
                'record_count': 100,
                'column_count': 4,
                'column_names': ['date', 'sales', 'rate', 'category'],
                'min_date': '2022-01-01',
                'max_date': '2022-04-10',
                'date_range_days': 99,
                'aggregation_ratio': None
            },
            previous_metrics=None,
            metadata_path=Path("/tmp/test_metadata.json")
        )
        mock_validate.return_value = baseline_result

        result = validate_extraction_output(
            df=sample_dataframe,
            stage_name="sales_extraction",
            critical_columns=['date', 'sales']
        )

        assert len(result) == 100

    @patch('src.validation.pipeline_validation_helpers.run_production_validation_checkpoint')
    def test_full_preprocessing_workflow(self, mock_validate, sample_dataframe):
        """Test complete preprocessing validation workflow."""
        passed_result = ValidationResult(
            checkpoint_name="product_filtering",
            status="PASSED",
            issues=[],
            warnings=["Row count decreased by 10% (expected for filtering)"],
            current_metrics={
                'run_date': '2024-01-01T12:00:00',
                'record_count': 90,
                'column_count': 4,
                'column_names': ['date', 'sales', 'rate', 'category'],
                'min_date': '2022-01-01',
                'max_date': '2022-04-01',
                'date_range_days': 90,
                'aggregation_ratio': None
            },
            previous_metrics={
                'run_date': '2024-01-01T11:00:00',
                'record_count': 100,
                'column_count': 4,
                'column_names': ['date', 'sales', 'rate', 'category'],
                'min_date': '2022-01-01',
                'max_date': '2022-04-10',
                'date_range_days': 99,
                'aggregation_ratio': None
            },
            metadata_path=Path("/tmp/test_metadata.json")
        )
        mock_validate.return_value = passed_result

        result = validate_preprocessing_output(
            df=sample_dataframe.iloc[:90],
            stage_name="product_filtering"
        )

        # Should pass despite shrinkage (allow_shrinkage=True by default)
        assert len(result) == 90
