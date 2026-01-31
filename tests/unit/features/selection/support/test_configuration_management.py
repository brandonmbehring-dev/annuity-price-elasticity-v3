"""
Tests for configuration_management module.

Target: 17% → 60%+ coverage
Tests organized by function categories:
- Feature validation
- Configuration building
- Configuration integrity validation
- Parameter inheritance
- Experiment configuration
- Configuration updates
"""

import pytest
from unittest.mock import patch, MagicMock

from src.features.selection.support.configuration_management import (
    _validate_feature_count,
    _validate_target_variable,
    _get_default_candidate_features,
    build_comprehensive_feature_config,
    _get_required_config_keys,
    _validate_required_keys,
    _validate_config_data_types,
    validate_configuration_integrity,
    _validate_inheritance_inputs,
    _apply_overrides,
    manage_parameter_inheritance,
    _build_constraint_config,
    _build_bootstrap_config,
    _build_mlflow_experiment_config,
    create_experiment_configurations,
    _validate_update_inputs,
    _apply_updates_with_logging,
    handle_configuration_updates,
)


# =============================================================================
# Feature Validation Tests
# =============================================================================


class TestValidateFeatureCount:
    """Tests for _validate_feature_count."""

    def test_accepts_valid_count(self):
        """Accepts valid feature count (1-10)."""
        _validate_feature_count(4)  # Should not raise

    def test_accepts_minimum(self):
        """Accepts minimum value of 1."""
        _validate_feature_count(1)  # Should not raise

    def test_accepts_maximum(self):
        """Accepts maximum value of 10."""
        _validate_feature_count(10)  # Should not raise

    def test_raises_on_zero(self):
        """Raises ValueError for zero."""
        with pytest.raises(ValueError, match="must be 1-10"):
            _validate_feature_count(0)

    def test_raises_on_negative(self):
        """Raises ValueError for negative."""
        with pytest.raises(ValueError, match="must be 1-10"):
            _validate_feature_count(-1)

    def test_raises_on_too_high(self):
        """Raises ValueError for values > 10."""
        with pytest.raises(ValueError, match="must be 1-10"):
            _validate_feature_count(11)


class TestValidateTargetVariable:
    """Tests for _validate_target_variable."""

    def test_accepts_valid_target(self):
        """Accepts valid target variable string."""
        _validate_target_variable("sales_target_current")  # Should not raise

    def test_raises_on_empty_string(self):
        """Raises ValueError for empty string."""
        with pytest.raises(ValueError):
            _validate_target_variable("")


class TestGetDefaultCandidateFeatures:
    """Tests for _get_default_candidate_features."""

    def test_returns_list(self):
        """Returns list of features."""
        result = _get_default_candidate_features()

        assert isinstance(result, list)

    def test_list_not_empty(self):
        """List contains at least one feature."""
        result = _get_default_candidate_features()

        assert len(result) > 0


# =============================================================================
# Configuration Building Tests
# =============================================================================


class TestBuildComprehensiveFeatureConfig:
    """Tests for build_comprehensive_feature_config."""

    def test_returns_dict(self):
        """Returns dictionary."""
        result = build_comprehensive_feature_config()

        assert isinstance(result, dict)

    def test_default_values(self):
        """Has correct default values."""
        result = build_comprehensive_feature_config()

        assert result['max_candidate_features'] == 4
        assert result['target_variable'] == "sales_target_current"
        assert result['analysis_start_date'] == "2022-08-01"

    def test_custom_values(self):
        """Accepts custom values."""
        result = build_comprehensive_feature_config(
            max_candidate_features=6,
            target_variable="custom_target",
            analysis_start_date="2023-01-01"
        )

        assert result['max_candidate_features'] == 6
        assert result['target_variable'] == "custom_target"
        assert result['analysis_start_date'] == "2023-01-01"

    def test_custom_candidate_features(self):
        """Accepts custom candidate features."""
        custom_features = ['feature_a', 'feature_b', 'feature_c', 'feature_d']

        result = build_comprehensive_feature_config(
            candidate_features=custom_features
        )

        assert result['candidate_features'] == custom_features

    def test_custom_base_features(self):
        """Accepts custom base features."""
        base_features = ['base_1', 'base_2']

        result = build_comprehensive_feature_config(
            base_features=base_features
        )

        assert result['base_features'] == base_features

    def test_contains_transformation_settings(self):
        """Contains transformation settings."""
        result = build_comprehensive_feature_config()

        assert 'target_transformation' in result
        assert 'holiday_exclusion' in result
        assert 'autoregressive_transform' in result

    def test_raises_on_invalid_feature_count(self):
        """Raises ValueError for invalid feature count."""
        with pytest.raises(ValueError):
            build_comprehensive_feature_config(max_candidate_features=0)


# =============================================================================
# Configuration Integrity Validation Tests
# =============================================================================


class TestGetRequiredConfigKeys:
    """Tests for _get_required_config_keys."""

    def test_returns_dict(self):
        """Returns dictionary."""
        result = _get_required_config_keys()

        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Contains required keys."""
        result = _get_required_config_keys()

        assert 'max_candidate_features' in result
        assert 'target_variable' in result
        assert 'candidate_features' in result
        assert 'base_features' in result


class TestValidateRequiredKeys:
    """Tests for _validate_required_keys."""

    def test_no_errors_when_all_present(self):
        """No errors when all keys present."""
        config = {
            'max_candidate_features': 4,
            'target_variable': 'target',
            'candidate_features': ['a', 'b'],
            'base_features': []
        }
        errors = []

        _validate_required_keys(config, errors)

        assert len(errors) == 0

    def test_adds_error_for_missing_key(self):
        """Adds error for missing key."""
        config = {'target_variable': 'target'}  # Missing other keys
        errors = []

        _validate_required_keys(config, errors)

        assert len(errors) > 0


class TestValidateConfigDataTypes:
    """Tests for _validate_config_data_types."""

    def test_no_errors_for_valid_types(self):
        """No errors for valid data types."""
        config = {
            'max_candidate_features': 4,
            'candidate_features': ['a', 'b', 'c', 'd'],
            'target_variable': 'target'
        }
        errors = []

        _validate_config_data_types(config, errors)

        assert len(errors) == 0

    def test_error_for_negative_feature_count(self):
        """Error for negative feature count."""
        config = {'max_candidate_features': -1}
        errors = []

        _validate_config_data_types(config, errors)

        assert any('positive integer' in e for e in errors)

    def test_error_for_non_list_features(self):
        """Error when candidate_features is not a list."""
        config = {'candidate_features': 'not_a_list', 'max_candidate_features': 1}
        errors = []

        _validate_config_data_types(config, errors)

        assert any('must be list' in e for e in errors)

    def test_error_for_empty_target(self):
        """Error for empty target variable."""
        config = {'target_variable': ''}
        errors = []

        _validate_config_data_types(config, errors)

        assert any('non-empty string' in e for e in errors)


class TestValidateConfigurationIntegrity:
    """Tests for validate_configuration_integrity."""

    def test_returns_tuple(self):
        """Returns tuple of (bool, list)."""
        config = build_comprehensive_feature_config()

        result = validate_configuration_integrity(config)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_valid_config_passes(self):
        """Valid configuration passes."""
        config = build_comprehensive_feature_config()

        is_valid, errors = validate_configuration_integrity(config)

        assert is_valid == True  # noqa: E712
        assert len(errors) == 0

    def test_invalid_config_fails(self):
        """Invalid configuration fails."""
        config = {}  # Missing all required keys

        is_valid, errors = validate_configuration_integrity(config)

        assert is_valid == False  # noqa: E712
        assert len(errors) > 0

    def test_raises_for_non_dict(self):
        """Raises ValueError for non-dict input."""
        with pytest.raises(ValueError, match="must be dictionary"):
            validate_configuration_integrity("not a dict")


# =============================================================================
# Parameter Inheritance Tests
# =============================================================================


class TestValidateInheritanceInputs:
    """Tests for _validate_inheritance_inputs."""

    def test_accepts_valid_inputs(self):
        """Accepts valid dict inputs."""
        _validate_inheritance_inputs({}, {})  # Should not raise

    def test_raises_for_non_dict_base(self):
        """Raises for non-dict base_config."""
        with pytest.raises(ValueError, match="Base configuration must be dictionary"):
            _validate_inheritance_inputs("not dict", {})

    def test_raises_for_non_dict_overrides(self):
        """Raises for non-dict overrides."""
        with pytest.raises(ValueError, match="User overrides must be dictionary"):
            _validate_inheritance_inputs({}, "not dict")


class TestApplyOverrides:
    """Tests for _apply_overrides."""

    def test_applies_override(self):
        """Applies override to config."""
        config = {'key': 'original'}
        overrides = {'key': 'new'}

        _apply_overrides(config, overrides)

        assert config['key'] == 'new'

    def test_adds_new_key_with_warning(self):
        """Adds new key and warns."""
        config = {}
        overrides = {'new_key': 'value'}

        with pytest.warns(UserWarning, match="Adding new parameter"):
            _apply_overrides(config, overrides)

        assert config['new_key'] == 'value'

    def test_warns_on_type_mismatch(self):
        """Warns on type mismatch."""
        config = {'key': 10}  # int
        overrides = {'key': 'string'}  # str

        with pytest.warns(UserWarning, match="Type mismatch"):
            _apply_overrides(config, overrides)


class TestManageParameterInheritance:
    """Tests for manage_parameter_inheritance."""

    def test_returns_merged_config(self):
        """Returns merged configuration."""
        base = build_comprehensive_feature_config()
        overrides = {'max_candidate_features': 5}

        result = manage_parameter_inheritance(base, overrides)

        assert result['max_candidate_features'] == 5

    def test_raises_for_invalid_result(self):
        """Raises when result is invalid."""
        base = build_comprehensive_feature_config()
        overrides = {'max_candidate_features': -1}  # Invalid

        with pytest.raises(ValueError, match="invalid configuration"):
            manage_parameter_inheritance(base, overrides)


# =============================================================================
# Experiment Configuration Tests
# =============================================================================


class TestBuildConstraintConfig:
    """Tests for _build_constraint_config."""

    def test_returns_config(self):
        """Returns constraint configuration."""
        result = _build_constraint_config()

        assert result is not None
        # May be dict or dataclass depending on implementation
        if hasattr(result, 'enabled'):
            assert result.enabled == True  # noqa: E712
        else:
            assert result['enabled'] == True  # noqa: E712


class TestBuildBootstrapConfig:
    """Tests for _build_bootstrap_config."""

    def test_returns_config(self):
        """Returns bootstrap configuration."""
        result = _build_bootstrap_config()

        assert result is not None
        # May be dict or dataclass depending on implementation
        if hasattr(result, 'enabled'):
            assert result.enabled == True  # noqa: E712
            assert result.n_samples == 100
        else:
            assert result['enabled'] == True  # noqa: E712
            assert result['n_samples'] == 100


class TestBuildMlflowExperimentConfig:
    """Tests for _build_mlflow_experiment_config."""

    def test_returns_dict(self):
        """Returns dictionary."""
        result = _build_mlflow_experiment_config()

        assert isinstance(result, dict)

    def test_contains_enabled(self):
        """Contains enabled flag."""
        result = _build_mlflow_experiment_config()

        assert result['enabled'] == True  # noqa: E712

    def test_contains_experiment_name(self):
        """Contains experiment name."""
        result = _build_mlflow_experiment_config()

        assert 'experiment_name' in result


class TestCreateExperimentConfigurations:
    """Tests for create_experiment_configurations."""

    def test_returns_dict(self):
        """Returns dictionary."""
        feature_config = build_comprehensive_feature_config()

        result = create_experiment_configurations(feature_config)

        assert isinstance(result, dict)

    def test_contains_all_configs(self):
        """Contains all configuration sections."""
        feature_config = build_comprehensive_feature_config()

        result = create_experiment_configurations(feature_config)

        assert 'constraint_config' in result
        assert 'bootstrap_config' in result
        assert 'experiment_config' in result

    def test_raises_for_empty_config(self):
        """Raises ValueError for empty config."""
        with pytest.raises(ValueError, match="Feature configuration required"):
            create_experiment_configurations({})


# =============================================================================
# Configuration Update Tests
# =============================================================================


class TestValidateUpdateInputs:
    """Tests for _validate_update_inputs."""

    def test_accepts_valid_inputs(self):
        """Accepts valid dict inputs."""
        _validate_update_inputs({}, {})  # Should not raise

    def test_raises_for_non_dict_current(self):
        """Raises for non-dict current_config."""
        with pytest.raises(ValueError, match="Current configuration must be dictionary"):
            _validate_update_inputs("not dict", {})

    def test_raises_for_non_dict_updates(self):
        """Raises for non-dict updates."""
        with pytest.raises(ValueError, match="Updates must be dictionary"):
            _validate_update_inputs({}, "not dict")


class TestApplyUpdatesWithLogging:
    """Tests for _apply_updates_with_logging."""

    def test_applies_update(self):
        """Applies update to config."""
        config = {'key': 'old'}
        updates = {'key': 'new'}

        _apply_updates_with_logging(config, updates)

        assert config['key'] == 'new'

    def test_prints_log(self, capsys):
        """Prints configuration update log."""
        config = {'key': 'old'}
        updates = {'key': 'new'}

        _apply_updates_with_logging(config, updates)

        captured = capsys.readouterr()
        assert 'Configuration update' in captured.out
        assert 'old' in captured.out
        assert 'new' in captured.out


class TestHandleConfigurationUpdates:
    """Tests for handle_configuration_updates."""

    def test_returns_updated_config(self):
        """Returns updated configuration."""
        current = build_comprehensive_feature_config()
        updates = {'max_candidate_features': 5}

        result = handle_configuration_updates(current, updates)

        assert result['max_candidate_features'] == 5

    def test_preserves_unchanged_values(self):
        """Preserves unchanged values."""
        current = build_comprehensive_feature_config()
        original_target = current['target_variable']
        updates = {'max_candidate_features': 5}

        result = handle_configuration_updates(current, updates)

        assert result['target_variable'] == original_target

    def test_raises_for_invalid_result(self):
        """Raises when result is invalid."""
        current = build_comprehensive_feature_config()
        updates = {'max_candidate_features': 0}  # Invalid

        with pytest.raises(ValueError, match="invalid state"):
            handle_configuration_updates(current, updates)
