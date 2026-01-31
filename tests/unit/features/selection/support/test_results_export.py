"""
Tests for results_export module.

Target: 12% → 60%+ coverage
Tests organized by function categories:
- Export data building
- File writing
- Main export function
- DVC checkpoint management
- MLflow finalization
- Analysis summary generation
- Numpy type conversion
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.features.selection.support.results_export import (
    _build_export_data,
    _write_export_file,
    export_feature_selection_results,
    manage_dvc_checkpoints,
    finalize_mlflow_experiment,
    generate_analysis_summary,
    convert_numpy_types,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_model_series():
    """Sample final selected model as Series."""
    return pd.Series({
        'features': 'own_rate_t1 + competitor_weighted_t2',
        'aic': 1250.5,
        'r_squared': 0.78,
        'n_features': 2,
        'coefficients': {'own_rate_t1': 0.15, 'competitor_weighted_t2': -0.08}
    })


@pytest.fixture
def sample_metadata():
    """Sample results metadata."""
    return {
        'analysis_date': '2026-01-30',
        'product': '6Y20B',
        'n_observations': 500
    }


@pytest.fixture
def sample_pipeline_results(sample_model_series):
    """Sample pipeline results."""
    return {
        'aic_results': pd.DataFrame({
            'features': ['a', 'b', 'c'],
            'aic': [100, 105, 110],
            'converged': [True, True, False]
        }),
        'valid_models': pd.DataFrame({
            'features': ['a', 'b'],
            'aic': [100, 105]
        }),
        'bootstrap_results': []
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    return str(tmp_path / "outputs")


# =============================================================================
# Export Data Building Tests
# =============================================================================


class TestBuildExportData:
    """Tests for _build_export_data."""

    def test_returns_dict(self, sample_model_series, sample_metadata):
        """Returns dictionary with expected structure."""
        result = _build_export_data(
            sample_model_series, 'bootstrap_stability', sample_metadata
        )

        assert isinstance(result, dict)

    def test_contains_metadata_section(self, sample_model_series, sample_metadata):
        """Contains feature_selection_metadata section."""
        result = _build_export_data(
            sample_model_series, 'bootstrap_stability', sample_metadata
        )

        assert 'feature_selection_metadata' in result
        assert result['feature_selection_metadata']['selection_method'] == 'bootstrap_stability'

    def test_contains_selected_model_section(self, sample_model_series, sample_metadata):
        """Contains selected_model section with correct values."""
        result = _build_export_data(
            sample_model_series, 'bootstrap_stability', sample_metadata
        )

        assert 'selected_model' in result
        model = result['selected_model']
        assert model['features'] == 'own_rate_t1 + competitor_weighted_t2'
        assert model['aic_score'] == 1250.5
        assert model['r_squared'] == 0.78

    def test_contains_analysis_metadata(self, sample_model_series, sample_metadata):
        """Contains analysis_metadata from input."""
        result = _build_export_data(
            sample_model_series, 'bootstrap_stability', sample_metadata
        )

        assert 'analysis_metadata' in result
        assert result['analysis_metadata']['product'] == '6Y20B'

    def test_includes_timestamp(self, sample_model_series, sample_metadata):
        """Includes analysis timestamp."""
        result = _build_export_data(
            sample_model_series, 'bootstrap_stability', sample_metadata
        )

        assert 'analysis_timestamp' in result['feature_selection_metadata']

    def test_handles_missing_values(self, sample_metadata):
        """Handles model with missing values gracefully."""
        sparse_model = pd.Series({'features': 'a + b'})

        result = _build_export_data(sparse_model, 'aic', sample_metadata)

        assert result['selected_model']['aic_score'] == 0.0
        assert result['selected_model']['r_squared'] == 0.0


# =============================================================================
# File Writing Tests
# =============================================================================


class TestWriteExportFile:
    """Tests for _write_export_file."""

    def test_creates_output_directory(self, temp_output_dir):
        """Creates output directory if not exists."""
        export_data = {'test': 'data'}

        _write_export_file(export_data, temp_output_dir)

        assert os.path.exists(temp_output_dir)

    def test_writes_json_file(self, temp_output_dir):
        """Writes JSON file with correct name."""
        export_data = {'test': 'data'}

        result = _write_export_file(export_data, temp_output_dir)

        expected_path = os.path.join(temp_output_dir, "final_selected_features.json")
        assert result == expected_path
        assert os.path.exists(expected_path)

    def test_json_content_correct(self, temp_output_dir):
        """JSON file contains correct data."""
        export_data = {'key': 'value', 'number': 42}

        path = _write_export_file(export_data, temp_output_dir)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded == export_data

    def test_prints_confirmation(self, temp_output_dir, capsys):
        """Prints confirmation message."""
        export_data = {'test': 'data'}

        _write_export_file(export_data, temp_output_dir)

        captured = capsys.readouterr()
        assert 'Feature selection results exported' in captured.out


# =============================================================================
# Main Export Function Tests
# =============================================================================


class TestExportFeatureSelectionResults:
    """Tests for export_feature_selection_results."""

    def test_raises_with_none_model(self, sample_metadata):
        """Raises ValueError when model is None."""
        with pytest.raises(ValueError, match="CRITICAL: No model provided"):
            export_feature_selection_results(None, 'aic', sample_metadata)

    def test_raises_with_empty_model(self, sample_metadata):
        """Raises ValueError when model is empty."""
        empty_model = pd.Series(dtype=object)

        with pytest.raises(ValueError, match="CRITICAL: No model provided"):
            export_feature_selection_results(empty_model, 'aic', sample_metadata)

    def test_returns_success_dict(self, sample_model_series, sample_metadata, tmp_path):
        """Returns dict with export status on success."""
        with patch('src.features.selection.support.results_export._write_export_file') as mock_write:
            mock_write.return_value = str(tmp_path / "test.json")

            result = export_feature_selection_results(
                sample_model_series, 'bootstrap', sample_metadata
            )

            assert result['export_status'] == 'success'
            assert result['selection_method'] == 'bootstrap'

    def test_raises_on_export_failure(self, sample_model_series, sample_metadata):
        """Raises ValueError on export failure."""
        with patch('src.features.selection.support.results_export._write_export_file') as mock_write:
            mock_write.side_effect = PermissionError("Cannot write")

            with pytest.raises(ValueError, match="export failed"):
                export_feature_selection_results(
                    sample_model_series, 'aic', sample_metadata
                )


# =============================================================================
# DVC Checkpoint Management Tests
# =============================================================================


class TestManageDvcCheckpoints:
    """Tests for manage_dvc_checkpoints."""

    def test_returns_no_files_status_when_empty(self):
        """Returns no_files status for empty list."""
        result = manage_dvc_checkpoints([])

        assert result['status'] == 'no_files'
        assert result['tracked_files'] == []

    def test_warns_on_empty_list(self):
        """Warns when no files provided."""
        with pytest.warns(UserWarning, match="No checkpoint files"):
            manage_dvc_checkpoints([])

    def test_warns_on_missing_file(self, tmp_path):
        """Warns when file doesn't exist."""
        missing_file = str(tmp_path / "nonexistent.parquet")

        with pytest.warns(UserWarning, match="not found"):
            result = manage_dvc_checkpoints([missing_file])

        assert missing_file in result['failed_files']

    def test_tracks_existing_file(self, tmp_path):
        """Tracks existing file with DVC."""
        # Create a test file
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test")

        with patch('os.system', return_value=0) as mock_system:
            result = manage_dvc_checkpoints([str(test_file)])

            mock_system.assert_called_once()
            assert str(test_file) in result['tracked_files']
            assert result['status'] == 'success'

    def test_handles_dvc_failure(self, tmp_path):
        """Handles DVC command failure."""
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test")

        with patch('os.system', return_value=1):  # Non-zero = failure
            result = manage_dvc_checkpoints([str(test_file)])

            assert str(test_file) in result['failed_files']

    def test_partial_success_status(self, tmp_path):
        """Returns partial_success when some files fail."""
        file1 = tmp_path / "good.parquet"
        file1.write_text("test")
        missing = str(tmp_path / "missing.parquet")

        with patch('os.system', return_value=0):
            result = manage_dvc_checkpoints([str(file1), missing])

            assert result['status'] == 'partial_success'

    def test_handles_exception(self, tmp_path):
        """Handles exception during DVC operations."""
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test")

        with patch('os.system', side_effect=Exception("DVC error")):
            result = manage_dvc_checkpoints([str(test_file)])

            assert 'error' in result['status']
            assert result['dvc_available'] == False  # noqa: E712


# =============================================================================
# MLflow Finalization Tests
# =============================================================================


class TestFinalizeMlflowExperiment:
    """Tests for finalize_mlflow_experiment."""

    def test_returns_success_on_completion(self):
        """Returns 'success' when MLflow finalization completes."""
        mock_mlflow = MagicMock()
        mock_mlflow.safe_mlflow_log_param = MagicMock()
        mock_mlflow.safe_mlflow_log_metric = MagicMock()
        mock_mlflow.end_mlflow_experiment = MagicMock()

        with patch.dict('sys.modules', {'src.config.mlflow_config': mock_mlflow}):
            result = finalize_mlflow_experiment('bootstrap', 'run_123')

            assert result == 'success'

    def test_returns_not_available_on_import_error(self):
        """Returns not_available when MLflow cannot be imported."""
        # Remove mlflow_config from modules to trigger ImportError
        import sys
        original = sys.modules.get('src.config.mlflow_config')

        with patch.dict('sys.modules', {'src.config.mlflow_config': None}):
            # The import will fail, returning 'mlflow_not_available'
            result = finalize_mlflow_experiment('aic', 'run_456')

        # Should handle gracefully
        assert result == 'mlflow_not_available'

    def test_logs_selection_method(self):
        """Logs enhanced selection method to MLflow."""
        mock_mlflow = MagicMock()

        with patch.dict('sys.modules', {'src.config.mlflow_config': mock_mlflow}):
            finalize_mlflow_experiment('stability_analysis', 'run_789')

            # Should log the selection method
            mock_mlflow.safe_mlflow_log_param.assert_any_call(
                'enhanced_selection_method', 'stability_analysis'
            )


# =============================================================================
# Analysis Summary Tests
# =============================================================================


class TestGenerateAnalysisSummary:
    """Tests for generate_analysis_summary."""

    def test_raises_with_empty_pipeline_results(self, sample_model_series):
        """Raises ValueError with empty pipeline results."""
        with pytest.raises(ValueError, match="Insufficient data"):
            generate_analysis_summary({}, sample_model_series)

    def test_raises_with_none_model(self, sample_pipeline_results):
        """Raises ValueError with None model."""
        with pytest.raises(ValueError, match="Insufficient data"):
            generate_analysis_summary(sample_pipeline_results, None)

    def test_returns_summary_dict(self, sample_pipeline_results, sample_model_series):
        """Returns dictionary with expected sections."""
        result = generate_analysis_summary(sample_pipeline_results, sample_model_series)

        assert 'executive_summary' in result
        assert 'analysis_statistics' in result
        assert 'business_insights' in result
        assert 'technical_metadata' in result

    def test_executive_summary_content(self, sample_pipeline_results, sample_model_series):
        """Executive summary contains model details."""
        result = generate_analysis_summary(sample_pipeline_results, sample_model_series)

        exec_sum = result['executive_summary']
        assert exec_sum['selected_features'] == 'own_rate_t1 + competitor_weighted_t2'
        assert exec_sum['aic_score'] == 1250.5

    def test_analysis_statistics(self, sample_pipeline_results, sample_model_series):
        """Analysis statistics contains counts."""
        result = generate_analysis_summary(sample_pipeline_results, sample_model_series)

        stats = result['analysis_statistics']
        assert stats['total_combinations_evaluated'] == 3
        assert stats['converged_models'] == 2
        assert stats['economically_valid_models'] == 2

    def test_handles_empty_aic_results(self, sample_model_series):
        """Handles empty AIC results gracefully."""
        pipeline = {
            'aic_results': pd.DataFrame(),
            'valid_models': pd.DataFrame(),
            'bootstrap_results': []
        }

        result = generate_analysis_summary(pipeline, sample_model_series)

        assert result['analysis_statistics']['converged_models'] == 0


# =============================================================================
# Numpy Type Conversion Tests
# =============================================================================


class TestConvertNumpyTypes:
    """Tests for convert_numpy_types."""

    def test_converts_int64(self):
        """Converts numpy int64 to Python int."""
        result = convert_numpy_types(np.int64(42))

        assert result == 42
        assert isinstance(result, int)

    def test_converts_float64(self):
        """Converts numpy float64 to Python float."""
        result = convert_numpy_types(np.float64(3.14))

        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_converts_array_to_list(self):
        """Converts numpy array to Python list."""
        arr = np.array([1, 2, 3])

        result = convert_numpy_types(arr)

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_converts_nested_dict(self):
        """Recursively converts values in nested dict."""
        data = {
            'a': np.int64(1),
            'b': {'c': np.float64(2.5)}
        }

        result = convert_numpy_types(data)

        assert result['a'] == 1
        assert isinstance(result['a'], int)
        assert result['b']['c'] == 2.5
        assert isinstance(result['b']['c'], float)

    def test_converts_list_elements(self):
        """Converts numpy types in lists."""
        data = [np.int64(1), np.float64(2.5), 'string']

        result = convert_numpy_types(data)

        assert result == [1, 2.5, 'string']
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_converts_nan_to_none(self):
        """Converts NaN to None for JSON compatibility."""
        result = convert_numpy_types(np.nan)

        assert result is None

    def test_preserves_regular_types(self):
        """Preserves regular Python types unchanged."""
        data = {'string': 'hello', 'int': 42, 'float': 3.14, 'list': [1, 2]}

        result = convert_numpy_types(data)

        assert result == data

    def test_handles_int32(self):
        """Handles numpy int32."""
        result = convert_numpy_types(np.int32(100))

        assert result == 100
        assert isinstance(result, int)

    def test_handles_float32(self):
        """Handles numpy float32."""
        result = convert_numpy_types(np.float32(1.5))

        assert result == pytest.approx(1.5)
        assert isinstance(result, float)
