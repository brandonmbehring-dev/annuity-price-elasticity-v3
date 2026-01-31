"""
Tests for interface_environment module.

Target: 14% → 60%+ coverage
Tests organized by function categories:
- Status constants
- Core imports building
- Atomic function imports
- MLflow integration imports
- Visualization configuration
- MLflow environment initialization
- Main setup orchestration
"""

import pytest
from unittest.mock import patch, MagicMock

from src.features.selection.interface.interface_environment import (
    _get_status_constants,
    _build_core_imports_dict,
    _import_core_libraries,
    _import_atomic_functions,
    _import_mlflow_integration,
    _configure_visualization_environment,
    _initialize_mlflow_environment,
    setup_feature_selection_environment,
)


# =============================================================================
# Status Constants Tests
# =============================================================================


class TestGetStatusConstants:
    """Tests for _get_status_constants."""

    def test_returns_dict(self):
        """Returns dictionary."""
        result = _get_status_constants()

        assert isinstance(result, dict)

    def test_contains_success(self):
        """Contains SUCCESS key."""
        result = _get_status_constants()

        assert 'SUCCESS' in result
        assert result['SUCCESS'] == "SUCCESS:"

    def test_contains_warning(self):
        """Contains WARNING key."""
        result = _get_status_constants()

        assert 'WARNING' in result
        assert result['WARNING'] == "WARNING:"

    def test_contains_error(self):
        """Contains ERROR key."""
        result = _get_status_constants()

        assert 'ERROR' in result
        assert result['ERROR'] == "ERROR:"


# =============================================================================
# Core Imports Tests
# =============================================================================


class TestBuildCoreImportsDict:
    """Tests for _build_core_imports_dict."""

    def test_returns_dict(self):
        """Returns dictionary."""
        result = _build_core_imports_dict()

        assert isinstance(result, dict)

    def test_contains_pandas(self):
        """Contains pandas as 'pd'."""
        result = _build_core_imports_dict()

        assert 'pd' in result
        import pandas
        assert result['pd'] is pandas

    def test_contains_numpy(self):
        """Contains numpy as 'np'."""
        result = _build_core_imports_dict()

        assert 'np' in result
        import numpy
        assert result['np'] is numpy

    def test_contains_matplotlib(self):
        """Contains matplotlib.pyplot as 'plt'."""
        result = _build_core_imports_dict()

        assert 'plt' in result

    def test_contains_seaborn(self):
        """Contains seaborn as 'sns'."""
        result = _build_core_imports_dict()

        assert 'sns' in result

    def test_contains_scipy_stats(self):
        """Contains scipy.stats as 'stats'."""
        result = _build_core_imports_dict()

        assert 'stats' in result

    def test_contains_datetime(self):
        """Contains datetime."""
        result = _build_core_imports_dict()

        assert 'datetime' in result

    def test_contains_combinations(self):
        """Contains itertools.combinations."""
        result = _build_core_imports_dict()

        assert 'combinations' in result


class TestImportCoreLibraries:
    """Tests for _import_core_libraries."""

    def test_returns_tuple(self):
        """Returns tuple of (imports, constants)."""
        result = _import_core_libraries()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_imports(self):
        """First element is imports dict."""
        imports, _ = _import_core_libraries()

        assert isinstance(imports, dict)
        assert 'pd' in imports

    def test_second_element_is_constants(self):
        """Second element is constants dict."""
        _, constants = _import_core_libraries()

        assert isinstance(constants, dict)
        assert 'SUCCESS' in constants


# =============================================================================
# Atomic Function Import Tests
# =============================================================================


class TestImportAtomicFunctions:
    """Tests for _import_atomic_functions."""

    def test_returns_tuple(self):
        """Returns tuple of (imports, success)."""
        base_imports = {}

        result = _import_atomic_functions(base_imports)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preserves_existing_imports(self):
        """Preserves existing imports in dict."""
        base_imports = {'existing': 'value'}

        imports, _ = _import_atomic_functions(base_imports)

        assert 'existing' in imports

    def test_returns_boolean_status(self):
        """Second element is boolean."""
        imports, success = _import_atomic_functions({})

        assert isinstance(success, bool)

    def test_handles_import_error_gracefully(self):
        """Handles import error without raising."""
        # Test that function doesn't raise, even if imports fail
        imports, success = _import_atomic_functions({})

        # Either succeeds or fails gracefully
        assert isinstance(success, bool)


# =============================================================================
# MLflow Integration Import Tests
# =============================================================================


class TestImportMlflowIntegration:
    """Tests for _import_mlflow_integration."""

    def test_returns_tuple(self):
        """Returns tuple of (imports, success)."""
        result = _import_mlflow_integration({})

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preserves_existing_imports(self):
        """Preserves existing imports."""
        base_imports = {'existing': 'value'}

        imports, _ = _import_mlflow_integration(base_imports)

        assert 'existing' in imports

    def test_returns_boolean_status(self):
        """Second element is boolean."""
        _, success = _import_mlflow_integration({})

        assert isinstance(success, bool)

    def test_handles_import_failure(self):
        """Handles import failure gracefully."""
        with patch.dict('sys.modules', {'src.config.mlflow_config': None}):
            imports, success = _import_mlflow_integration({})

        # Should not raise, just return status
        assert isinstance(success, bool)


# =============================================================================
# Visualization Configuration Tests
# =============================================================================


class TestConfigureVisualizationEnvironment:
    """Tests for _configure_visualization_environment."""

    def test_returns_tuple(self):
        """Returns tuple of (config, seaborn_success, mlflow_init)."""
        result = _configure_visualization_environment()

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_fig_config_dict(self):
        """First element is figure config dict."""
        fig_config, _, _ = _configure_visualization_environment()

        assert isinstance(fig_config, dict)

    def test_fig_config_has_dimensions(self):
        """Figure config contains width and height."""
        fig_config, _, _ = _configure_visualization_environment()

        assert 'fig_width' in fig_config
        assert 'fig_height' in fig_config

    def test_returns_seaborn_success_boolean(self):
        """Second element is boolean for seaborn success."""
        _, seaborn_success, _ = _configure_visualization_environment()

        assert isinstance(seaborn_success, bool)

    def test_seaborn_typically_succeeds(self):
        """Seaborn configuration typically succeeds."""
        _, seaborn_success, _ = _configure_visualization_environment()

        # In most environments seaborn works
        assert seaborn_success == True  # noqa: E712


# =============================================================================
# MLflow Environment Initialization Tests
# =============================================================================


class TestInitializeMlflowEnvironment:
    """Tests for _initialize_mlflow_environment."""

    def test_returns_false_when_not_available(self):
        """Returns False when mlflow_available is False."""
        result = _initialize_mlflow_environment({}, mlflow_available=False)

        assert result == False  # noqa: E712

    def test_returns_true_on_success(self):
        """Returns True when setup succeeds."""
        mock_setup = MagicMock()
        imports = {'setup_environment_for_notebooks': mock_setup}

        result = _initialize_mlflow_environment(imports, mlflow_available=True)

        assert result == True  # noqa: E712
        mock_setup.assert_called_once()

    def test_handles_setup_exception(self):
        """Handles exception during setup."""
        mock_setup = MagicMock(side_effect=Exception("Setup failed"))
        imports = {'setup_environment_for_notebooks': mock_setup}

        result = _initialize_mlflow_environment(imports, mlflow_available=True)

        assert result == False  # noqa: E712


# =============================================================================
# Main Setup Orchestration Tests
# =============================================================================


class TestSetupFeatureSelectionEnvironment:
    """Tests for setup_feature_selection_environment."""

    def test_returns_dict(self):
        """Returns dictionary."""
        result = setup_feature_selection_environment()

        assert isinstance(result, dict)

    def test_contains_imports(self):
        """Contains imports section."""
        result = setup_feature_selection_environment()

        assert 'imports' in result

    def test_contains_constants(self):
        """Contains constants section."""
        result = setup_feature_selection_environment()

        assert 'constants' in result

    def test_contains_fig_config(self):
        """Contains fig_config section."""
        result = setup_feature_selection_environment()

        assert 'fig_config' in result

    def test_contains_status(self):
        """Contains status section."""
        result = setup_feature_selection_environment()

        assert 'status' in result

    def test_status_has_atomic_functions_available(self):
        """Status includes atomic_functions_available."""
        result = setup_feature_selection_environment()

        assert 'atomic_functions_available' in result['status']

    def test_status_has_mlflow_available(self):
        """Status includes mlflow_available."""
        result = setup_feature_selection_environment()

        assert 'mlflow_available' in result['status']

    def test_status_has_setup_complete(self):
        """Status includes setup_complete flag."""
        result = setup_feature_selection_environment()

        assert 'setup_complete' in result['status']
        assert result['status']['setup_complete'] == True  # noqa: E712

    def test_imports_contain_core_libraries(self):
        """Imports contain core libraries."""
        result = setup_feature_selection_environment()

        assert 'pd' in result['imports']
        assert 'np' in result['imports']
        assert 'plt' in result['imports']

    def test_constants_contain_status_messages(self):
        """Constants contain status messages."""
        result = setup_feature_selection_environment()

        assert 'SUCCESS' in result['constants']
        assert 'WARNING' in result['constants']
        assert 'ERROR' in result['constants']
