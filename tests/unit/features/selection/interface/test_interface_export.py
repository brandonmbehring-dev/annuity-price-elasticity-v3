"""
Tests for interface_export module.

Target: 12% → 60%+ coverage
Tests organized by function categories:
- JSON serialization helpers
- Metadata creation functions
- Bootstrap results export
- File I/O functions
- MLflow integration
- Main export function
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, mock_open
from dataclasses import dataclass
from typing import List, Dict, Any

from src.features.selection.interface.interface_export import (
    # JSON serialization
    _convert_numpy_types,
    # Metadata creation
    _create_feature_selection_metadata,
    _create_selected_model_metadata,
    _create_selection_process_metadata,
    _create_configuration_metadata,
    _export_model_metadata,
    # Bootstrap export
    _export_bootstrap_results,
    # File I/O
    _log_dvc_success,
    _export_to_files,
    # MLflow
    _finalize_mlflow_experiment,
    # Main function
    export_final_model_selection,
    # Dual validation
    save_dual_validation_results,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_selected_model():
    """Sample selected model as pandas Series."""
    return pd.Series({
        'features': 'own_rate_t1 + comp_t2',
        'aic': 1234.56,
        'r_squared': 0.85,
        'n_features': 2,
        'coefficients': {'own_rate_t1': 0.15, 'comp_t2': -0.08}
    })


@pytest.fixture
def sample_results_df():
    """Sample results DataFrame."""
    return pd.DataFrame({
        'model_id': [1, 2, 3],
        'aic': [100, 105, 110],
        'converged': [True, True, False]
    })


@pytest.fixture
def sample_valid_results():
    """Sample valid results DataFrame."""
    return pd.DataFrame({
        'model_id': [1, 2],
        'aic': [100, 105]
    })


@dataclass
class MockBootstrapResult:
    """Mock BootstrapResult for testing."""
    bootstrap_aics: List[float]
    stability_assessment: str
    aic_stability_coefficient: float
    r2_stability_coefficient: float
    successful_fits: int
    total_attempts: int


@pytest.fixture
def sample_bootstrap_results():
    """Sample bootstrap results for testing."""
    np.random.seed(42)
    return [
        MockBootstrapResult(
            bootstrap_aics=np.random.normal(100, 5, 50).tolist(),
            stability_assessment='Stable',
            aic_stability_coefficient=0.05,
            r2_stability_coefficient=0.08,
            successful_fits=50,
            total_attempts=50
        )
    ]


@pytest.fixture
def sample_configs():
    """Sample configuration dictionaries."""
    return {
        'feature_config': {'max_features': 10},
        'constraint_config': {'own_rate_positive': True},
        'bootstrap_config': {'n_iterations': 100}
    }


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestConvertNumpyTypes:
    """Tests for _convert_numpy_types."""

    def test_converts_numpy_int64(self):
        """Converts numpy int64 to Python int."""
        result = _convert_numpy_types(np.int64(42))

        assert isinstance(result, int)
        assert result == 42

    def test_converts_numpy_int32(self):
        """Converts numpy int32 to Python int."""
        result = _convert_numpy_types(np.int32(42))

        assert isinstance(result, int)
        assert result == 42

    def test_converts_numpy_float64(self):
        """Converts numpy float64 to Python float."""
        result = _convert_numpy_types(np.float64(3.14))

        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_converts_numpy_float32(self):
        """Converts numpy float32 to Python float."""
        result = _convert_numpy_types(np.float32(3.14))

        assert isinstance(result, float)

    def test_converts_numpy_array(self):
        """Converts numpy array to Python list."""
        arr = np.array([1, 2, 3])
        result = _convert_numpy_types(arr)

        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_converts_dict_recursively(self):
        """Converts dict values recursively."""
        data = {'a': np.int64(1), 'b': {'c': np.float64(2.5)}}
        result = _convert_numpy_types(data)

        assert isinstance(result['a'], int)
        assert isinstance(result['b']['c'], float)

    def test_converts_list_recursively(self):
        """Converts list items recursively."""
        data = [np.int64(1), np.float64(2.0), 'string']
        result = _convert_numpy_types(data)

        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert result[2] == 'string'

    def test_passes_through_standard_types(self):
        """Passes through standard Python types unchanged."""
        data = {'int': 42, 'float': 3.14, 'str': 'hello', 'list': [1, 2]}
        result = _convert_numpy_types(data)

        assert result == data

    def test_handles_numpy_scalar_with_item(self):
        """Handles numpy scalars with .item() method."""
        val = np.array(42)  # 0-d array, has .item()
        result = _convert_numpy_types(val)

        assert result == 42


# =============================================================================
# Metadata Creation Tests
# =============================================================================


class TestCreateFeatureSelectionMetadata:
    """Tests for _create_feature_selection_metadata."""

    def test_returns_dict_with_expected_keys(self):
        """Returns dictionary with expected keys."""
        result = _create_feature_selection_metadata('original_aic_only')

        expected_keys = ['analysis_timestamp', 'refactoring_version',
                        'mathematical_equivalence', 'configuration_system', 'selection_method']
        for key in expected_keys:
            assert key in result

    def test_includes_selection_method(self):
        """Includes the selection method."""
        result = _create_feature_selection_metadata('stability_weighted')

        assert result['selection_method'] == 'stability_weighted'

    def test_timestamp_is_iso_format(self):
        """Timestamp is in ISO format."""
        result = _create_feature_selection_metadata('test')

        # ISO format has 'T' separator
        assert 'T' in result['analysis_timestamp']


class TestCreateSelectedModelMetadata:
    """Tests for _create_selected_model_metadata."""

    def test_returns_dict_with_expected_keys(self, sample_selected_model):
        """Returns dictionary with expected keys."""
        result = _create_selected_model_metadata(sample_selected_model)

        expected_keys = ['features', 'aic_score', 'r_squared', 'n_features', 'coefficients']
        for key in expected_keys:
            assert key in result

    def test_converts_to_native_types(self, sample_selected_model):
        """Converts values to native Python types."""
        result = _create_selected_model_metadata(sample_selected_model)

        assert isinstance(result['aic_score'], float)
        assert isinstance(result['r_squared'], float)
        assert isinstance(result['n_features'], int)


class TestCreateSelectionProcessMetadata:
    """Tests for _create_selection_process_metadata."""

    def test_returns_dict_with_expected_keys(self, sample_results_df, sample_valid_results):
        """Returns dictionary with expected keys."""
        result = _create_selection_process_metadata(
            sample_results_df, sample_valid_results, []
        )

        expected_keys = ['total_combinations_evaluated', 'converged_models',
                        'economically_valid_models', 'constraint_violations']
        for key in expected_keys:
            assert key in result

    def test_counts_converged_models(self, sample_results_df, sample_valid_results):
        """Counts converged models correctly."""
        result = _create_selection_process_metadata(
            sample_results_df, sample_valid_results, []
        )

        assert result['converged_models'] == 2  # Two True values

    def test_handles_missing_converged_column(self, sample_valid_results):
        """Handles DataFrame without converged column."""
        df_no_converged = pd.DataFrame({'aic': [100, 105]})

        result = _create_selection_process_metadata(
            df_no_converged, sample_valid_results, []
        )

        assert result['converged_models'] == 0

    def test_counts_constraint_violations(self, sample_results_df, sample_valid_results):
        """Counts constraint violations correctly."""
        violations = ['v1', 'v2', 'v3']

        result = _create_selection_process_metadata(
            sample_results_df, sample_valid_results, violations
        )

        assert result['constraint_violations'] == 3


class TestCreateConfigurationMetadata:
    """Tests for _create_configuration_metadata."""

    def test_returns_dict_with_expected_keys(self, sample_configs):
        """Returns dictionary with expected keys."""
        result = _create_configuration_metadata(
            sample_configs['feature_config'],
            sample_configs['constraint_config'],
            sample_configs['bootstrap_config']
        )

        assert 'feature_config' in result
        assert 'constraint_config' in result
        assert 'bootstrap_config' in result

    def test_converts_numpy_types_in_configs(self):
        """Converts numpy types within configs."""
        config = {'value': np.int64(42)}

        result = _create_configuration_metadata(config, {}, {})

        assert isinstance(result['feature_config']['value'], int)


class TestExportModelMetadata:
    """Tests for _export_model_metadata."""

    def test_returns_basic_export(self, sample_selected_model):
        """Returns basic export with metadata and selected model."""
        result = _export_model_metadata(sample_selected_model, 'original_aic_only')

        assert 'feature_selection_metadata' in result
        assert 'selected_model' in result

    def test_includes_selection_process_when_provided(
        self, sample_selected_model, sample_results_df, sample_valid_results
    ):
        """Includes selection process when results DataFrames provided."""
        result = _export_model_metadata(
            sample_selected_model, 'test',
            results_df=sample_results_df,
            valid_results_sorted=sample_valid_results
        )

        assert 'selection_process' in result

    def test_includes_configuration_when_all_provided(
        self, sample_selected_model, sample_configs
    ):
        """Includes configuration when all configs provided."""
        result = _export_model_metadata(
            sample_selected_model, 'test',
            feature_config=sample_configs['feature_config'],
            constraint_config=sample_configs['constraint_config'],
            bootstrap_config=sample_configs['bootstrap_config']
        )

        assert 'configuration' in result


# =============================================================================
# Bootstrap Export Tests
# =============================================================================


class TestExportBootstrapResults:
    """Tests for _export_bootstrap_results."""

    def test_returns_unchanged_if_no_bootstrap(self, sample_selected_model):
        """Returns unchanged dict if no bootstrap results."""
        base_export = {'test': 'data'}

        result = _export_bootstrap_results(
            base_export, [], sample_selected_model, 'test'
        )

        assert result == base_export

    def test_adds_stability_analysis(self, sample_selected_model, sample_bootstrap_results):
        """Adds stability_analysis section with bootstrap results."""
        base_export = {}

        result = _export_bootstrap_results(
            base_export, sample_bootstrap_results, sample_selected_model, 'stability_weighted'
        )

        assert 'stability_analysis' in result

    def test_stability_analysis_keys(self, sample_selected_model, sample_bootstrap_results):
        """Stability analysis contains expected keys."""
        base_export = {}

        result = _export_bootstrap_results(
            base_export, sample_bootstrap_results, sample_selected_model, 'test'
        )

        expected_keys = ['selection_method', 'best_model_stability',
                        'aic_stability_coefficient', 'bootstrap_median_aic']
        for key in expected_keys:
            assert key in result['stability_analysis']


# =============================================================================
# File I/O Tests
# =============================================================================


class TestLogDvcSuccess:
    """Tests for _log_dvc_success."""

    def test_prints_success_message(self, capsys):
        """Prints success message with path."""
        _log_dvc_success('/path/to/file.json')

        captured = capsys.readouterr()
        assert 'SUCCESS' in captured.out
        assert '/path/to/file.json' in captured.out


class TestExportToFiles:
    """Tests for _export_to_files."""

    def test_creates_output_directory(self):
        """Creates output directory if needed."""
        with patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('os.system', return_value=0):
            _export_to_files({'test': 'data'})

            mock_makedirs.assert_called_with('outputs/feature_selection', exist_ok=True)

    def test_writes_json_file(self):
        """Writes JSON file."""
        with patch('os.makedirs'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_dump, \
             patch('os.system', return_value=0):
            _export_to_files({'test': 'data'})

            mock_dump.assert_called_once()

    def test_returns_success_status_on_dvc_success(self):
        """Returns success status when DVC succeeds."""
        with patch('os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('os.system', return_value=0):
            result = _export_to_files({'test': 'data'})

            assert result['dvc_status'] == 'success'

    def test_returns_failed_status_on_dvc_failure(self):
        """Returns failed status when DVC fails."""
        with patch('os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'), \
             patch('os.system', return_value=1):  # Non-zero = failure
            result = _export_to_files({'test': 'data'})

            assert result['dvc_status'] == 'failed'

    def test_handles_exception(self):
        """Handles exceptions during export."""
        with patch('os.makedirs', side_effect=Exception("Test error")):
            result = _export_to_files({'test': 'data'})

            assert 'error' in result['dvc_status']


# =============================================================================
# MLflow Tests
# =============================================================================


class TestFinalizeMlflowExperiment:
    """Tests for _finalize_mlflow_experiment."""

    def test_returns_success_on_completion(self):
        """Returns success when MLflow completes."""
        # Patch at the module where it's imported (inside the function)
        with patch('src.config.mlflow_config.safe_mlflow_log_param'), \
             patch('src.config.mlflow_config.end_mlflow_experiment'):
            result = _finalize_mlflow_experiment('test_method', 'run_123')

            assert result == 'success'

    def test_returns_error_on_failure(self):
        """Returns error message on failure."""
        # Force import to fail
        with patch.dict('sys.modules', {'src.config.mlflow_config': None}):
            result = _finalize_mlflow_experiment('test_method', 'run_123')

            assert 'error' in result

    def test_logs_enhanced_params(self):
        """Logs enhanced MLflow parameters."""
        with patch('src.config.mlflow_config.safe_mlflow_log_param') as mock_log, \
             patch('src.config.mlflow_config.end_mlflow_experiment'):
            _finalize_mlflow_experiment('stability_weighted', 'run_123')

            # Should log multiple params
            assert mock_log.call_count >= 4


# =============================================================================
# Main Export Function Tests
# =============================================================================


class TestExportFinalModelSelection:
    """Tests for export_final_model_selection."""

    def test_returns_status_dict(self, sample_selected_model):
        """Returns status dictionary."""
        with patch('src.features.selection.interface.interface_export._export_to_files') as mock_export, \
             patch('src.features.selection.interface.interface_export._finalize_mlflow_experiment') as mock_mlflow:
            mock_export.return_value = {'export_paths': [], 'dvc_status': 'success'}
            mock_mlflow.return_value = 'success'

            result = export_final_model_selection(sample_selected_model)

            assert 'export_paths' in result
            assert 'mlflow_status' in result
            assert 'dvc_status' in result
            assert 'final_results' in result

    def test_includes_bootstrap_when_provided(
        self, sample_selected_model, sample_bootstrap_results
    ):
        """Includes bootstrap results when provided."""
        with patch('src.features.selection.interface.interface_export._export_to_files') as mock_export, \
             patch('src.features.selection.interface.interface_export._finalize_mlflow_experiment') as mock_mlflow:
            mock_export.return_value = {'export_paths': [], 'dvc_status': 'success'}
            mock_mlflow.return_value = 'success'

            result = export_final_model_selection(
                sample_selected_model,
                bootstrap_results=sample_bootstrap_results
            )

            assert 'stability_analysis' in result['final_results']

    def test_handles_orchestration_error(self, sample_selected_model, capsys):
        """Handles orchestration errors gracefully."""
        with patch('src.features.selection.interface.interface_export._export_model_metadata',
                   side_effect=Exception("Test error")):
            result = export_final_model_selection(sample_selected_model)

            assert 'orchestration_error' in result['dvc_status']


# =============================================================================
# Dual Validation Export Tests
# =============================================================================


class TestSaveDualValidationResults:
    """Tests for save_dual_validation_results."""

    def test_saves_to_file(self):
        """Saves summary to file."""
        summary = {'test': 'data'}

        with patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_dump:
            save_dual_validation_results(summary, '/path/to/output.json')

            mock_file.assert_called_once_with('/path/to/output.json', 'w')
            mock_dump.assert_called_once()

    def test_converts_numpy_types(self):
        """Converts numpy types before saving."""
        summary = {'value': np.int64(42)}

        with patch('builtins.open', mock_open()), \
             patch('json.dump') as mock_dump:
            save_dual_validation_results(summary, '/path/to/output.json')

            # Check that converted value was passed to dump
            call_args = mock_dump.call_args[0][0]
            assert isinstance(call_args['value'], int)

    def test_prints_success(self, capsys):
        """Prints success message."""
        with patch('builtins.open', mock_open()), \
             patch('json.dump'):
            save_dual_validation_results({}, '/path/to/output.json')

            captured = capsys.readouterr()
            assert 'SUCCESS' in captured.out

    def test_handles_error(self, capsys):
        """Handles file write errors."""
        with patch('builtins.open', side_effect=Exception("Write error")):
            save_dual_validation_results({}, '/path/to/output.json')

            captured = capsys.readouterr()
            assert 'WARNING' in captured.out
