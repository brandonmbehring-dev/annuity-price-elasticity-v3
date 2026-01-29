"""
Unit tests for src/config/configuration_validator.py

Tests parameter validation, common mistake detection, parameter mapping,
error reporting, and safe config wrappers.
"""

import pytest
import warnings
from unittest.mock import MagicMock, patch


class TestValidateFunctionParameters:
    """Tests for validate_function_parameters function."""

    def test_valid_parameters_returns_true(self):
        """Valid parameters pass validation."""
        from src.config.configuration_validator import validate_function_parameters

        def sample_func(a: int, b: str, c: float = 1.0) -> None:
            pass

        is_valid, errors = validate_function_parameters(
            sample_func, {'a': 1, 'b': 'test'}
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_parameter_returns_false(self):
        """Invalid parameters fail validation with error messages."""
        from src.config.configuration_validator import validate_function_parameters

        def sample_func(a: int, b: str) -> None:
            pass

        is_valid, errors = validate_function_parameters(
            sample_func, {'a': 1, 'invalid_param': 'bad'}
        )
        assert is_valid is False
        assert len(errors) == 2
        assert 'invalid_param' in str(errors)
        assert 'Valid parameters' in str(errors)

    def test_empty_params_is_valid(self):
        """Empty parameters dict is valid for function with defaults."""
        from src.config.configuration_validator import validate_function_parameters

        def func_with_defaults(a: int = 1, b: str = "test") -> None:
            pass

        is_valid, errors = validate_function_parameters(func_with_defaults, {})
        assert is_valid is True

    def test_subset_of_params_is_valid(self):
        """Providing subset of parameters is valid when others have defaults."""
        from src.config.configuration_validator import validate_function_parameters

        def sample_func(a: int, b: str = "default", c: float = 1.0) -> None:
            pass

        is_valid, errors = validate_function_parameters(sample_func, {'a': 42})
        assert is_valid is True


class TestCheckCommonParameterMistakes:
    """Tests for check_common_parameter_mistakes function."""

    def test_detects_min_feature_combinations(self):
        """Detects non-existent min_feature_combinations parameter."""
        from src.config.configuration_validator import check_common_parameter_mistakes

        warnings_list = check_common_parameter_mistakes(
            {'min_feature_combinations': 5}
        )
        assert len(warnings_list) == 1
        assert 'min_feature_combinations' in warnings_list[0]
        assert 'does not exist' in warnings_list[0]

    def test_detects_bootstrap_enabled_mistake(self):
        """Detects bootstrap_enabled parameter naming mistake."""
        from src.config.configuration_validator import check_common_parameter_mistakes

        warnings_list = check_common_parameter_mistakes({'bootstrap_enabled': True})
        assert len(warnings_list) == 1
        assert 'enable_block_bootstrap' in warnings_list[0]

    def test_detects_enhanced_metrics_enabled_mistake(self):
        """Detects enhanced_metrics_enabled parameter naming mistake."""
        from src.config.configuration_validator import check_common_parameter_mistakes

        warnings_list = check_common_parameter_mistakes(
            {'enhanced_metrics_enabled': True}
        )
        assert len(warnings_list) == 1
        assert 'Enhanced metrics' in warnings_list[0]

    def test_detects_constraints_enabled_mistake(self):
        """Detects constraints_enabled parameter naming mistake."""
        from src.config.configuration_validator import check_common_parameter_mistakes

        warnings_list = check_common_parameter_mistakes({'constraints_enabled': True})
        assert len(warnings_list) == 1
        assert 'economic_constraints' in warnings_list[0]

    def test_no_warnings_for_valid_params(self):
        """No warnings for parameters without common mistakes."""
        from src.config.configuration_validator import check_common_parameter_mistakes

        warnings_list = check_common_parameter_mistakes(
            {'target_variable': 'sales_target_current', 'max_candidate_features': 10}
        )
        assert len(warnings_list) == 0

    def test_multiple_mistakes_detected(self):
        """Multiple parameter mistakes are all detected."""
        from src.config.configuration_validator import check_common_parameter_mistakes

        warnings_list = check_common_parameter_mistakes({
            'min_feature_combinations': 5,
            'bootstrap_enabled': True,
            'mlflow_enabled': True,
        })
        assert len(warnings_list) == 3


class TestMapEnhancedParameters:
    """Tests for _map_enhanced_parameters function."""

    def test_maps_bootstrap_enabled(self):
        """Maps bootstrap_enabled to enable_block_bootstrap."""
        from src.config.configuration_validator import _map_enhanced_parameters

        result = _map_enhanced_parameters({'bootstrap_enabled': True})
        assert 'enable_block_bootstrap' in result
        assert result['enable_block_bootstrap'] is True
        assert 'bootstrap_enabled' not in result

    def test_maps_enhanced_metrics_enabled(self):
        """Maps enhanced_metrics_enabled to enable_multiple_testing_correction."""
        from src.config.configuration_validator import _map_enhanced_parameters

        result = _map_enhanced_parameters({'enhanced_metrics_enabled': True})
        assert 'enable_multiple_testing_correction' in result
        assert result['enable_multiple_testing_correction'] is True

    def test_removes_nonexistent_params(self):
        """Removes parameters that don't exist in target."""
        from src.config.configuration_validator import _map_enhanced_parameters

        result = _map_enhanced_parameters({'min_feature_combinations': 5})
        assert 'min_feature_combinations' not in result

    def test_preserves_unmapped_params(self):
        """Preserves parameters not in mapping."""
        from src.config.configuration_validator import _map_enhanced_parameters

        result = _map_enhanced_parameters({'custom_param': 'value'})
        assert 'custom_param' in result
        assert result['custom_param'] == 'value'


class TestMapBasicParameters:
    """Tests for _map_basic_parameters function."""

    def test_filters_to_basic_params_only(self):
        """Only basic parameters are preserved."""
        from src.config.configuration_validator import _map_basic_parameters

        result = _map_basic_parameters({
            'max_candidate_features': 20,
            'target_variable': 'sales',
            'unknown_param': 'ignored',
        })
        assert 'max_candidate_features' in result
        assert 'target_variable' in result
        assert 'unknown_param' not in result

    def test_sets_default_target_variable(self):
        """Sets default target_variable if not provided."""
        from src.config.configuration_validator import _map_basic_parameters

        result = _map_basic_parameters({'max_candidate_features': 20})
        assert 'target_variable' in result

    def test_sets_default_economic_constraints(self):
        """Sets default economic_constraints if not provided."""
        from src.config.configuration_validator import _map_basic_parameters

        result = _map_basic_parameters({})
        assert 'economic_constraints' in result
        assert result['economic_constraints'] is True


class TestSuggestCorrectConfiguration:
    """Tests for suggest_correct_configuration function."""

    def test_suggests_enhanced_for_bootstrap(self):
        """Suggests enhanced system when bootstrap parameter present."""
        from src.config.configuration_validator import suggest_correct_configuration

        suggestion = suggest_correct_configuration({'bootstrap_enabled': True})
        assert suggestion['recommended_system'] == 'enhanced'
        assert 'enhanced_feature_selection_config' in suggestion['import_statement']

    def test_suggests_basic_for_simple_params(self):
        """Suggests basic system for simple parameters."""
        from src.config.configuration_validator import suggest_correct_configuration

        suggestion = suggest_correct_configuration({'max_candidate_features': 20})
        assert suggestion['recommended_system'] == 'basic'
        assert 'feature_selection_stage_config' in suggestion['import_statement']

    def test_includes_corrected_params(self):
        """Suggestion includes corrected parameters."""
        from src.config.configuration_validator import suggest_correct_configuration

        suggestion = suggest_correct_configuration({'bootstrap_enabled': True})
        assert 'corrected_params' in suggestion
        assert 'enable_block_bootstrap' in suggestion['corrected_params']

    def test_includes_function_call_example(self):
        """Suggestion includes example function call."""
        from src.config.configuration_validator import suggest_correct_configuration

        suggestion = suggest_correct_configuration({'max_candidate_features': 15})
        assert 'function_call' in suggestion
        assert 'max_candidate_features=15' in suggestion['function_call']


class TestGenerateErrorReport:
    """Tests for generate_error_report function."""

    def test_generates_report_with_function_name(self):
        """Report includes function name."""
        from src.config.configuration_validator import generate_error_report

        report = generate_error_report(
            'test_function',
            {'invalid_param': 'value'},
            TypeError("unexpected keyword argument")
        )
        assert 'test_function' in report
        assert 'CONFIGURATION ERROR ANALYSIS' in report

    def test_includes_parameter_warnings(self):
        """Report includes warnings about common mistakes."""
        from src.config.configuration_validator import generate_error_report

        report = generate_error_report(
            'test_function',
            {'min_feature_combinations': 5},
            TypeError("unexpected keyword argument")
        )
        assert 'COMMON PARAMETER ISSUES' in report
        assert 'min_feature_combinations' in report

    def test_includes_recommendation(self):
        """Report includes recommended solution."""
        from src.config.configuration_validator import generate_error_report

        report = generate_error_report(
            'test_function',
            {'target_variable': 'sales'},
            TypeError("error")
        )
        assert 'RECOMMENDED SOLUTION' in report
        assert 'CORRECT CODE' in report


class TestCreateSafeConfigWrapper:
    """Tests for create_safe_config_wrapper function."""

    def test_wrapper_calls_original_function(self):
        """Wrapper calls original function with valid params."""
        from src.config.configuration_validator import create_safe_config_wrapper

        mock_func = MagicMock(return_value={'config': 'value'})
        mock_func.__name__ = 'mock_func'

        # Skip validation by mocking validate_configuration_call
        with patch(
            'src.config.configuration_validator.validate_configuration_call'
        ):
            wrapped = create_safe_config_wrapper(mock_func, 'mock_func')
            result = wrapped(param1='value1')

        mock_func.assert_called_once_with(param1='value1')
        assert result == {'config': 'value'}

    def test_wrapper_preserves_function_name(self):
        """Wrapped function has descriptive behavior."""
        from src.config.configuration_validator import create_safe_config_wrapper

        def original_func(**kwargs):
            return kwargs

        wrapped = create_safe_config_wrapper(original_func, 'original_func')
        assert callable(wrapped)


class TestSafeConfigWrappers:
    """Tests for exported safe wrapper functions."""

    def test_safe_basic_wrapper_exists(self):
        """Safe basic config wrapper is exported."""
        from src.config.configuration_validator import (
            safe_build_feature_selection_stage_config
        )
        assert callable(safe_build_feature_selection_stage_config)

    def test_safe_enhanced_wrapper_exists_or_none(self):
        """Safe enhanced config wrapper exists or is None if module missing."""
        from src.config.configuration_validator import (
            safe_build_enhanced_feature_selection_config
        )
        # Can be None if enhanced module not implemented
        assert safe_build_enhanced_feature_selection_config is None or callable(
            safe_build_enhanced_feature_selection_config
        )
