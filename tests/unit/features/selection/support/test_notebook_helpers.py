"""
Tests for Notebook Helpers Module.

Tests cover:
- Feature flag system (FEATURE_FLAGS, create_feature_flags, handle_feature_flags)
- Display functions (display_results_summary, format_feature_selection_results)
- Progress tracking (create_progress_tracker)
- Diagnostic generation (generate_diagnostic_information)
- Configuration helpers (create_configuration_helpers)
- Notebook utilities (provide_notebook_utilities)

Design Principles:
- Mock IPython.display for notebook-specific functions
- Test both DI pattern and global fallback
- Cover error handling and edge cases

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, MagicMock
import io
import sys

from src.features.selection.support.notebook_helpers import (
    # Feature flags
    FEATURE_FLAGS,
    create_feature_flags,
    get_feature_flags,
    handle_feature_flags,
    # Display functions
    display_results_summary,
    format_feature_selection_results,
    _format_executive_summary,
    _format_coefficients_summary,
    _format_metadata_summary,
    # Progress tracking
    create_progress_tracker,
    # Diagnostics
    generate_diagnostic_information,
    _collect_pipeline_diagnostics,
    _analyze_data_diagnostics,
    _generate_diagnostic_recommendations,
    # Configuration
    create_configuration_helpers,
    # Notebook utilities
    provide_notebook_utilities,
    _build_utilities,
    _build_display_helpers,
    _build_validation_utilities,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_model_series():
    """Create sample model Series for testing."""
    return pd.Series({
        'features': ('competitor_mid_t1', 'prudential_rate_t0'),
        'aic': 1234.567,
        'r_squared': 0.8543,
        'n_features': 2,
        'coefficients': {
            'Intercept': 10.5,
            'competitor_mid_t1': -0.25,
            'prudential_rate_t0': 0.15
        }
    })


@pytest.fixture
def sample_model_dataframe():
    """Create sample model DataFrame for testing."""
    return pd.DataFrame({
        'features': [
            ('f1', 'f2'),
            ('f3', 'f4'),
            ('f5',)
        ],
        'aic': [100.5, 105.2, 98.7],
        'r_squared': [0.85, 0.82, 0.87],
        'n_features': [2, 2, 1]
    })


@pytest.fixture
def sample_analysis_metadata():
    """Create sample analysis metadata."""
    return {
        'selection_method': 'AIC-based forward selection',
        'total_models': 150,
        'constraints_applied': True,
        'bootstrap_completed': True
    }


@pytest.fixture
def sample_pipeline_state():
    """Create sample pipeline state for diagnostics."""
    return {
        'pipeline_progress': ['data_loading', 'preprocessing'],
        'last_completed_stage': 'preprocessing',
        'feature_config': {'max_features': 4},
        'data_frame': pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    }


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for utility testing."""
    return pd.DataFrame({
        'competitor_mid_t1': [1.0, 2.0, 3.0],
        'prudential_rate_t0': [0.05, 0.06, 0.07],
        'sales_target_t1': [100, 110, 120],
        'other_col': ['a', 'b', 'c']
    })


# =============================================================================
# Tests for Feature Flags
# =============================================================================


class TestFeatureFlags:
    """Tests for FEATURE_FLAGS global and related functions."""

    def test_feature_flags_is_dict(self):
        """Test that FEATURE_FLAGS is a dictionary."""
        assert isinstance(FEATURE_FLAGS, dict)

    def test_feature_flags_has_required_keys(self):
        """Test that FEATURE_FLAGS has expected keys."""
        expected_keys = [
            'USE_ATOMIC_FUNCTIONS',
            'ENABLE_VALIDATION',
            'SHOW_DETAILED_OUTPUT',
            'ENABLE_BOOTSTRAP_DEFAULT',
            'STRICT_CONSTRAINTS_DEFAULT',
            'AUTO_DISPLAY_RESULTS'
        ]
        for key in expected_keys:
            assert key in FEATURE_FLAGS

    def test_feature_flags_values_are_bool(self):
        """Test that all feature flag values are booleans."""
        for value in FEATURE_FLAGS.values():
            assert isinstance(value, bool)


class TestCreateFeatureFlags:
    """Tests for create_feature_flags factory function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        flags = create_feature_flags()
        assert isinstance(flags, dict)

    def test_returns_new_instance(self):
        """Test that each call returns new instance (not global)."""
        flags1 = create_feature_flags()
        flags2 = create_feature_flags()

        # Modify one, other should be unaffected
        flags1['USE_ATOMIC_FUNCTIONS'] = False
        assert flags2['USE_ATOMIC_FUNCTIONS'] is True

    def test_has_same_keys_as_global(self):
        """Test that factory creates flags with same keys as global."""
        flags = create_feature_flags()
        assert set(flags.keys()) == set(FEATURE_FLAGS.keys())

    def test_default_values(self):
        """Test that default values are set correctly."""
        flags = create_feature_flags()
        assert flags['USE_ATOMIC_FUNCTIONS'] is True
        assert flags['ENABLE_VALIDATION'] is True
        assert flags['ENABLE_BOOTSTRAP_DEFAULT'] is False


class TestGetFeatureFlags:
    """Tests for get_feature_flags backward compatibility function."""

    def test_returns_global_flags(self):
        """Test that function returns global FEATURE_FLAGS."""
        result = get_feature_flags()
        assert result is FEATURE_FLAGS

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = get_feature_flags()
        assert isinstance(result, dict)


class TestHandleFeatureFlags:
    """Tests for handle_feature_flags management function."""

    def test_returns_current_flags_with_no_updates(self):
        """Test that function returns flags when no updates provided."""
        flags = create_feature_flags()
        result = handle_feature_flags(feature_flags=flags)

        assert result == flags

    def test_updates_provided_flags(self, capsys):
        """Test that function updates provided flags dictionary."""
        flags = create_feature_flags()
        original_value = flags['USE_ATOMIC_FUNCTIONS']

        result = handle_feature_flags(
            flag_updates={'USE_ATOMIC_FUNCTIONS': not original_value},
            feature_flags=flags
        )

        assert result['USE_ATOMIC_FUNCTIONS'] is not original_value

        # Check print output
        captured = capsys.readouterr()
        assert 'Feature flag updated' in captured.out

    def test_warns_on_unknown_flag(self):
        """Test that function warns on unknown flag name."""
        flags = create_feature_flags()

        with pytest.warns(UserWarning, match="Unknown feature flag"):
            handle_feature_flags(
                flag_updates={'NONEXISTENT_FLAG': True},
                feature_flags=flags
            )

    def test_returns_copy_not_original(self):
        """Test that function returns a copy."""
        flags = create_feature_flags()
        result = handle_feature_flags(feature_flags=flags)

        # Modify result, original should be unchanged
        result['USE_ATOMIC_FUNCTIONS'] = not result['USE_ATOMIC_FUNCTIONS']
        assert flags['USE_ATOMIC_FUNCTIONS'] is not result['USE_ATOMIC_FUNCTIONS']


# =============================================================================
# Tests for Display Functions
# =============================================================================


class TestDisplayResultsSummary:
    """Tests for display_results_summary function."""

    def test_handles_none_results(self, capsys):
        """Test that function handles None results."""
        display_results_summary(None)

        captured = capsys.readouterr()
        assert "No results available" in captured.out

    def test_displays_series_results(self, sample_model_series, capsys):
        """Test display of Series (single model) results."""
        display_results_summary(sample_model_series)

        captured = capsys.readouterr()
        assert "FEATURE SELECTION RESULTS SUMMARY" in captured.out
        assert "Selected Model" in captured.out
        assert "AIC Score" in captured.out

    def test_displays_dataframe_results(self, sample_model_dataframe, capsys):
        """Test display of DataFrame (multiple models) results."""
        display_results_summary(sample_model_dataframe)

        captured = capsys.readouterr()
        assert "Model Rankings" in captured.out

    def test_handles_show_coefficients_false(self, sample_model_series, capsys):
        """Test that show_coefficients=False hides coefficients."""
        display_results_summary(sample_model_series, show_coefficients=False)

        captured = capsys.readouterr()
        # Should not show individual coefficient lines
        assert "competitor_mid_t1:" not in captured.out

    def test_handles_unknown_type(self, capsys):
        """Test handling of unknown result type."""
        display_results_summary("some string result")

        captured = capsys.readouterr()
        assert "Results type:" in captured.out

    def test_handles_exception_gracefully(self, capsys):
        """Test graceful handling of display errors."""
        # Create Series that will cause formatting issues
        bad_series = pd.Series({'aic': 'not_a_number'})

        display_results_summary(bad_series)

        captured = capsys.readouterr()
        assert "Results display failed" in captured.out or "FEATURE SELECTION" in captured.out


class TestFormatExecutiveSummary:
    """Tests for _format_executive_summary helper."""

    def test_returns_string(self):
        """Test that function returns string."""
        result = _format_executive_summary(
            features=('f1', 'f2'),
            aic_score=100.5,
            r_squared=0.85,
            n_features=2
        )
        assert isinstance(result, str)

    def test_contains_features(self):
        """Test that output contains features."""
        result = _format_executive_summary(
            features=('feature_a', 'feature_b'),
            aic_score=100.0,
            r_squared=0.85,
            n_features=2
        )
        assert 'feature_a' in result

    def test_contains_aic_formatted(self):
        """Test that AIC is formatted with 3 decimals."""
        result = _format_executive_summary(
            features=('f1',),
            aic_score=123.456789,
            r_squared=0.85,
            n_features=1
        )
        assert '123.457' in result

    def test_contains_r_squared_formatted(self):
        """Test that R-squared is formatted with 4 decimals."""
        result = _format_executive_summary(
            features=('f1',),
            aic_score=100.0,
            r_squared=0.123456789,
            n_features=1
        )
        assert '0.1235' in result


class TestFormatCoefficientsSummary:
    """Tests for _format_coefficients_summary helper."""

    def test_returns_string(self):
        """Test that function returns string."""
        result = _format_coefficients_summary({'Intercept': 10.0})
        assert isinstance(result, str)

    def test_intercept_labeled_baseline(self):
        """Test that Intercept is labeled as baseline."""
        result = _format_coefficients_summary({'Intercept': 10.5})
        assert 'baseline level' in result

    def test_competitor_negative_pass(self):
        """Test that negative competitor coefficient shows PASS."""
        result = _format_coefficients_summary({'competitor_mid_t1': -0.25})
        assert '[PASS]' in result

    def test_competitor_positive_warn(self):
        """Test that positive competitor coefficient shows WARN."""
        result = _format_coefficients_summary({'competitor_mid_t1': 0.25})
        assert '[WARN]' in result

    def test_prudential_positive_pass(self):
        """Test that positive prudential coefficient shows PASS."""
        result = _format_coefficients_summary({'prudential_rate_t0': 0.15})
        assert '[PASS]' in result

    def test_prudential_negative_warn(self):
        """Test that negative prudential coefficient shows WARN."""
        result = _format_coefficients_summary({'prudential_rate_t0': -0.15})
        assert '[WARN]' in result


class TestFormatMetadataSummary:
    """Tests for _format_metadata_summary helper."""

    def test_returns_string(self, sample_analysis_metadata):
        """Test that function returns string."""
        result = _format_metadata_summary(sample_analysis_metadata)
        assert isinstance(result, str)

    def test_contains_selection_method(self, sample_analysis_metadata):
        """Test that output contains selection method."""
        result = _format_metadata_summary(sample_analysis_metadata)
        assert 'AIC-based forward selection' in result

    def test_constraints_applied_shows_applied(self, sample_analysis_metadata):
        """Test that applied constraints show 'Applied'."""
        result = _format_metadata_summary(sample_analysis_metadata)
        assert 'Applied' in result

    def test_constraints_not_applied(self):
        """Test that unapplied constraints show 'Not Applied'."""
        metadata = {'constraints_applied': False}
        result = _format_metadata_summary(metadata)
        assert 'Not Applied' in result


class TestFormatFeatureSelectionResults:
    """Tests for format_feature_selection_results main function."""

    def test_returns_dict(self, sample_model_series, sample_analysis_metadata):
        """Test that function returns dictionary."""
        result = format_feature_selection_results(
            sample_model_series, sample_analysis_metadata
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_model_series, sample_analysis_metadata):
        """Test that result has required keys."""
        result = format_feature_selection_results(
            sample_model_series, sample_analysis_metadata
        )

        assert 'executive_summary' in result
        assert 'coefficients_summary' in result
        assert 'metadata_summary' in result
        assert 'full_report' in result

    def test_raises_on_none_model(self, sample_analysis_metadata):
        """Test that None model raises ValueError."""
        with pytest.raises(ValueError, match="CRITICAL"):
            format_feature_selection_results(None, sample_analysis_metadata)

    def test_raises_on_empty_model(self, sample_analysis_metadata):
        """Test that empty model raises ValueError."""
        empty_series = pd.Series(dtype=object)

        with pytest.raises(ValueError, match="CRITICAL"):
            format_feature_selection_results(empty_series, sample_analysis_metadata)

    def test_full_report_combines_sections(self, sample_model_series, sample_analysis_metadata):
        """Test that full_report contains all sections."""
        result = format_feature_selection_results(
            sample_model_series, sample_analysis_metadata
        )

        full_report = result['full_report']
        assert 'FINAL FEATURE SELECTION RESULTS' in full_report
        assert 'Model Coefficients' in full_report
        assert 'Analysis Metadata' in full_report


# =============================================================================
# Tests for Progress Tracking
# =============================================================================


class TestCreateProgressTracker:
    """Tests for create_progress_tracker function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        tracker = create_progress_tracker(3)
        assert isinstance(tracker, dict)

    def test_has_required_keys(self):
        """Test that tracker has required keys."""
        tracker = create_progress_tracker(3)

        assert 'total_stages' in tracker
        assert 'stage_names' in tracker
        assert 'current_stage' in tracker
        assert 'completed_stages' in tracker
        assert 'stage_status' in tracker
        assert 'start_time' in tracker
        assert 'stage_times' in tracker

    def test_total_stages_set(self):
        """Test that total_stages is set correctly."""
        tracker = create_progress_tracker(5)
        assert tracker['total_stages'] == 5

    def test_default_stage_names(self):
        """Test that default stage names are generated."""
        tracker = create_progress_tracker(3)

        assert tracker['stage_names'] == ['Stage 1', 'Stage 2', 'Stage 3']

    def test_custom_stage_names(self):
        """Test that custom stage names are used."""
        names = ['Loading', 'Processing', 'Output']
        tracker = create_progress_tracker(3, stage_names=names)

        assert tracker['stage_names'] == names

    def test_initial_stage_status_pending(self):
        """Test that all stages start as pending."""
        tracker = create_progress_tracker(3)

        for status in tracker['stage_status'].values():
            assert status == 'pending'

    def test_raises_on_zero_stages(self):
        """Test that zero stages raises ValueError."""
        with pytest.raises(ValueError, match="CRITICAL"):
            create_progress_tracker(0)

    def test_raises_on_negative_stages(self):
        """Test that negative stages raises ValueError."""
        with pytest.raises(ValueError, match="CRITICAL"):
            create_progress_tracker(-5)

    def test_warns_on_mismatched_names(self):
        """Test that mismatched name count generates warning."""
        with pytest.warns(UserWarning, match="Stage name count"):
            create_progress_tracker(3, stage_names=['One', 'Two'])


# =============================================================================
# Tests for Diagnostic Generation
# =============================================================================


class TestCollectPipelineDiagnostics:
    """Tests for _collect_pipeline_diagnostics helper."""

    def test_returns_dict(self, sample_pipeline_state):
        """Test that function returns dictionary."""
        result = _collect_pipeline_diagnostics(sample_pipeline_state)
        assert isinstance(result, dict)

    def test_extracts_stages_completed(self, sample_pipeline_state):
        """Test that stages_completed is extracted."""
        result = _collect_pipeline_diagnostics(sample_pipeline_state)
        assert result['stages_completed'] == ['data_loading', 'preprocessing']

    def test_extracts_last_completed(self, sample_pipeline_state):
        """Test that last_completed is extracted."""
        result = _collect_pipeline_diagnostics(sample_pipeline_state)
        assert result['last_completed'] == 'preprocessing'

    def test_handles_empty_state(self):
        """Test handling of empty pipeline state."""
        result = _collect_pipeline_diagnostics({})
        assert result['pipeline_status'] == 'empty'


class TestAnalyzeDataDiagnostics:
    """Tests for _analyze_data_diagnostics helper."""

    def test_returns_dict(self, sample_pipeline_state):
        """Test that function returns dictionary."""
        result = _analyze_data_diagnostics(sample_pipeline_state)
        assert isinstance(result, dict)

    def test_identifies_dataframes(self, sample_pipeline_state):
        """Test that DataFrames are identified."""
        result = _analyze_data_diagnostics(sample_pipeline_state)
        assert 'data_frame' in result['datasets_available']

    def test_captures_shapes(self, sample_pipeline_state):
        """Test that DataFrame shapes are captured."""
        result = _analyze_data_diagnostics(sample_pipeline_state)
        assert result['dataset_shapes']['data_frame'] == (3, 2)

    def test_flags_empty_dataframe(self):
        """Test that empty DataFrames are flagged."""
        state = {'empty_df': pd.DataFrame()}
        result = _analyze_data_diagnostics(state)

        assert any('empty dataset' in flag for flag in result['missing_data_flags'])

    def test_flags_null_values(self):
        """Test that null values are flagged."""
        state = {'df_with_nulls': pd.DataFrame({'a': [1, None, 3]})}
        result = _analyze_data_diagnostics(state)

        assert any('contains null values' in flag for flag in result['missing_data_flags'])


class TestGenerateDiagnosticRecommendations:
    """Tests for _generate_diagnostic_recommendations helper."""

    def test_returns_list(self):
        """Test that function returns list."""
        pipeline_diag = {'stages_completed': ['stage1']}
        data_diag = {'missing_data_flags': [], 'datasets_available': ['df1']}

        result = _generate_diagnostic_recommendations(pipeline_diag, data_diag)
        assert isinstance(result, list)

    def test_recommends_when_no_stages(self):
        """Test recommendation when no stages completed."""
        pipeline_diag = {'stages_completed': []}
        data_diag = {'missing_data_flags': [], 'datasets_available': []}

        result = _generate_diagnostic_recommendations(pipeline_diag, data_diag)
        assert any('Pipeline not started' in rec for rec in result)

    def test_recommends_when_data_issues(self):
        """Test recommendation when data quality issues exist."""
        pipeline_diag = {'stages_completed': ['stage1']}
        data_diag = {'missing_data_flags': ['issue1'], 'datasets_available': ['df1']}

        result = _generate_diagnostic_recommendations(pipeline_diag, data_diag)
        assert any('Data quality issues' in rec for rec in result)

    def test_recommends_when_no_datasets(self):
        """Test recommendation when no datasets available."""
        pipeline_diag = {'stages_completed': ['stage1']}
        data_diag = {'missing_data_flags': [], 'datasets_available': []}

        result = _generate_diagnostic_recommendations(pipeline_diag, data_diag)
        assert any('No datasets found' in rec for rec in result)


class TestGenerateDiagnosticInformation:
    """Tests for generate_diagnostic_information main function."""

    def test_returns_dict(self, sample_pipeline_state):
        """Test that function returns dictionary."""
        result = generate_diagnostic_information(sample_pipeline_state)
        assert isinstance(result, dict)

    def test_has_required_sections(self, sample_pipeline_state):
        """Test that result has required sections."""
        result = generate_diagnostic_information(sample_pipeline_state)

        assert 'pipeline_diagnostics' in result
        assert 'data_diagnostics' in result
        assert 'configuration_diagnostics' in result
        assert 'recommendations' in result

    def test_includes_error_context(self, sample_pipeline_state):
        """Test that error context is included."""
        error_msg = "Test error context"
        result = generate_diagnostic_information(
            sample_pipeline_state, error_context=error_msg
        )

        assert result['error_context'] == error_msg

    def test_default_error_context(self, sample_pipeline_state):
        """Test that default error context is used when not provided."""
        result = generate_diagnostic_information(sample_pipeline_state)

        assert result['error_context'] == 'No error context provided'

    def test_handles_exception_gracefully(self):
        """Test graceful handling of diagnostic generation errors."""
        # Create an object that will cause issues during iteration
        class BadObject:
            def keys(self):
                raise RuntimeError("Simulated error")

        result = generate_diagnostic_information(BadObject())

        assert 'diagnostic_error' in result
        assert 'recommendations' in result


# =============================================================================
# Tests for Configuration Helpers
# =============================================================================


class TestCreateConfigurationHelpers:
    """Tests for create_configuration_helpers function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        result = create_configuration_helpers()
        assert isinstance(result, dict)

    def test_has_templates(self):
        """Test that templates section exists."""
        result = create_configuration_helpers()
        assert 'templates' in result

    def test_has_validation(self):
        """Test that validation section exists."""
        result = create_configuration_helpers()
        assert 'validation' in result

    def test_has_helper_functions(self):
        """Test that helper_functions section exists."""
        result = create_configuration_helpers()
        assert 'helper_functions' in result

    def test_templates_has_feature_config(self):
        """Test that feature_config_template exists."""
        result = create_configuration_helpers()
        assert 'feature_config_template' in result['templates']

    def test_templates_has_constraint_config(self):
        """Test that constraint_config_template exists."""
        result = create_configuration_helpers()
        assert 'constraint_config_template' in result['templates']

    def test_templates_has_bootstrap_config(self):
        """Test that bootstrap_config_template exists."""
        result = create_configuration_helpers()
        assert 'bootstrap_config_template' in result['templates']

    def test_validate_config_function_works(self):
        """Test that validate_config helper function works."""
        result = create_configuration_helpers()
        validate_config = result['helper_functions']['validate_config']

        # Valid config should pass
        valid_config = {
            'max_candidate_features': 4,
            'target_variable': 'sales_target_current',
            'candidate_features': []
        }
        assert validate_config(valid_config, 'feature_config') is True

        # Invalid config should fail
        invalid_config = {'max_candidate_features': 4}
        assert validate_config(invalid_config, 'feature_config') is False


# =============================================================================
# Tests for Notebook Utilities
# =============================================================================


class TestBuildUtilities:
    """Tests for _build_utilities helper function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        result = _build_utilities()
        assert isinstance(result, dict)

    def test_has_display_dataframe_info(self):
        """Test that display_dataframe_info utility exists."""
        result = _build_utilities()
        assert 'display_dataframe_info' in result

    def test_display_dataframe_info_works(self, sample_dataframe, capsys):
        """Test that display_dataframe_info prints correct info."""
        utilities = _build_utilities()
        utilities['display_dataframe_info'](sample_dataframe)

        captured = capsys.readouterr()
        assert 'Shape:' in captured.out
        assert 'Columns:' in captured.out

    def test_check_missing_data_works(self, sample_dataframe):
        """Test that check_missing_data identifies nulls."""
        utilities = _build_utilities()

        # No missing data
        result = utilities['check_missing_data'](sample_dataframe)
        assert len(result) == 0

        # With missing data
        df_with_nulls = sample_dataframe.copy()
        df_with_nulls.loc[0, 'competitor_mid_t1'] = None
        result = utilities['check_missing_data'](df_with_nulls)
        assert 'competitor_mid_t1' in result.index

    def test_format_large_numbers(self):
        """Test number formatting utility."""
        utilities = _build_utilities()
        formatter = utilities['format_large_numbers']

        assert formatter(1500) == '1,500'
        assert formatter(0.5) == '0.50'

    def test_create_feature_summary(self):
        """Test feature summary utility."""
        utilities = _build_utilities()
        summarizer = utilities['create_feature_summary']

        features = [
            'competitor_mid_t1',
            'competitor_top5_t1',
            'prudential_rate_t0',
            'sales_target_t1',
            'other_feature'
        ]

        summary = summarizer(features)
        assert summary['total_features'] == 5
        assert summary['competitor_features'] == 2
        assert summary['prudential_features'] == 1
        assert summary['autoregressive_features'] == 1


class TestBuildDisplayHelpers:
    """Tests for _build_display_helpers function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        result = _build_display_helpers()
        assert isinstance(result, dict)

    def test_show_progress_works(self, capsys):
        """Test that show_progress prints correctly."""
        helpers = _build_display_helpers()
        helpers['show_progress'](5, 10)

        captured = capsys.readouterr()
        assert 'Progress:' in captured.out
        assert '50.0%' in captured.out

    def test_display_config_summary_works(self, capsys):
        """Test that display_config_summary prints JSON."""
        helpers = _build_display_helpers()
        helpers['display_config_summary']({'key': 'value'})

        captured = capsys.readouterr()
        assert 'key' in captured.out
        assert 'value' in captured.out

    def test_create_results_table_with_dataframe(self, sample_model_dataframe):
        """Test results table creation with DataFrame."""
        helpers = _build_display_helpers()
        result = helpers['create_results_table'](sample_model_dataframe)

        assert isinstance(result, str)

    def test_create_results_table_with_non_dataframe(self):
        """Test results table creation with non-DataFrame."""
        helpers = _build_display_helpers()
        result = helpers['create_results_table']("string result")

        assert result == "string result"


class TestBuildValidationUtilities:
    """Tests for _build_validation_utilities function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        result = _build_validation_utilities()
        assert isinstance(result, dict)

    def test_check_data_quality(self, sample_dataframe):
        """Test data quality checking utility."""
        utilities = _build_validation_utilities()
        quality = utilities['check_data_quality'](sample_dataframe)

        assert quality['empty'] is False
        assert quality['null_percentage'] == 0.0
        assert quality['duplicate_rows'] == 0

    def test_check_data_quality_empty_df(self):
        """Test data quality check on empty DataFrame."""
        utilities = _build_validation_utilities()
        quality = utilities['check_data_quality'](pd.DataFrame())

        assert quality['empty'] is True

    def test_validate_target_variable(self, sample_dataframe):
        """Test target variable validation."""
        utilities = _build_validation_utilities()

        assert utilities['validate_target_variable'](sample_dataframe, 'competitor_mid_t1') is True
        assert utilities['validate_target_variable'](sample_dataframe, 'nonexistent') is False

    def test_check_feature_availability(self, sample_dataframe):
        """Test feature availability checking."""
        utilities = _build_validation_utilities()

        features = ['competitor_mid_t1', 'nonexistent', 'prudential_rate_t0']
        available = utilities['check_feature_availability'](sample_dataframe, features)

        assert 'competitor_mid_t1' in available
        assert 'prudential_rate_t0' in available
        assert 'nonexistent' not in available


class TestProvideNotebookUtilities:
    """Tests for provide_notebook_utilities main function."""

    def test_returns_dict(self):
        """Test that function returns dictionary."""
        result = provide_notebook_utilities()
        assert isinstance(result, dict)

    def test_has_required_sections(self):
        """Test that result has required sections."""
        result = provide_notebook_utilities()

        assert 'utilities' in result
        assert 'display_helpers' in result
        assert 'validation_utilities' in result
        assert 'feature_flags' in result

    def test_uses_global_flags_when_not_provided(self):
        """Test that global flags are used when not provided."""
        result = provide_notebook_utilities()

        # Should contain same keys as global
        assert set(result['feature_flags'].keys()) == set(FEATURE_FLAGS.keys())

    def test_uses_provided_flags(self):
        """Test that provided flags are used."""
        custom_flags = {'CUSTOM_FLAG': True}
        result = provide_notebook_utilities(feature_flags=custom_flags)

        assert result['feature_flags'] == custom_flags

    def test_utilities_section_has_functions(self):
        """Test that utilities section has callable functions."""
        result = provide_notebook_utilities()

        assert callable(result['utilities']['display_dataframe_info'])
        assert callable(result['utilities']['check_missing_data'])


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for notebook helpers."""

    def test_empty_coefficients_summary(self):
        """Test coefficient summary with empty dict."""
        result = _format_coefficients_summary({})
        assert 'Model Coefficients' in result

    def test_display_results_with_large_dataframe(self, capsys):
        """Test display with large DataFrame (truncation)."""
        large_df = pd.DataFrame({
            'features': [f'model_{i}' for i in range(20)],
            'aic': range(20),
            'r_squared': [0.8] * 20
        })

        display_results_summary(large_df)

        captured = capsys.readouterr()
        # Should show "Top 10"
        assert "Top 10" in captured.out

    def test_metadata_with_missing_keys(self):
        """Test metadata summary with missing keys."""
        sparse_metadata = {}
        result = _format_metadata_summary(sparse_metadata)

        assert 'Not specified' in result or 'Unknown' in result

    def test_progress_tracker_with_single_stage(self):
        """Test progress tracker with single stage."""
        tracker = create_progress_tracker(1)

        assert tracker['total_stages'] == 1
        assert len(tracker['stage_names']) == 1

    def test_diagnostics_with_non_dataframe_values(self):
        """Test data diagnostics with non-DataFrame values."""
        state = {
            'string_val': 'test',
            'int_val': 42,
            'list_val': [1, 2, 3]
        }

        result = _analyze_data_diagnostics(state)
        assert len(result['datasets_available']) == 0

    def test_utilities_with_empty_dataframe(self):
        """Test utilities with empty DataFrame."""
        utilities = _build_utilities()
        empty_df = pd.DataFrame()

        # Should handle gracefully
        result = utilities['check_missing_data'](empty_df)
        assert result == "Empty DataFrame"

        result = utilities['summarize_numeric_columns'](empty_df)
        assert result == "No numeric columns"
