"""
Unit tests for src/config/builders/builder_base.py.

Tests validate generic builder patterns for configuration construction.
"""

import pytest
from typing import TypedDict, Dict, Any

from src.config.builders.builder_base import (
    simple_config_builder,
    aggregate_configs,
    with_defaults,
    build_config_section,
    validate_config_keys,
    chain_builders,
    ConfigBuilderMixin,
)


# =============================================================================
# Test Fixtures - Configuration Classes
# =============================================================================


class SampleConfig(TypedDict, total=False):
    """Sample TypedDict for testing builders."""
    n_samples: int
    random_state: int
    enabled: bool


class NestedConfig(TypedDict, total=False):
    """Nested TypedDict for testing deep merge."""
    model: Dict[str, Any]
    validation: Dict[str, Any]


# =============================================================================
# simple_config_builder() Tests
# =============================================================================


class TestSimpleConfigBuilder:
    """Tests for simple_config_builder function."""

    def test_simple_config_builder_creates_typeddict(self):
        """simple_config_builder should create TypedDict with kwargs."""
        config = simple_config_builder(SampleConfig, n_samples=1000, random_state=42)

        assert isinstance(config, dict)
        assert config['n_samples'] == 1000
        assert config['random_state'] == 42

    def test_simple_config_builder_empty_kwargs(self):
        """simple_config_builder with no kwargs should create empty dict."""
        config = simple_config_builder(SampleConfig)

        assert isinstance(config, dict)
        assert len(config) == 0

    def test_simple_config_builder_with_bool_value(self):
        """simple_config_builder should handle boolean values."""
        config = simple_config_builder(SampleConfig, enabled=False)

        assert config['enabled'] is False


# =============================================================================
# aggregate_configs() Tests
# =============================================================================


class TestAggregateConfigs:
    """Tests for aggregate_configs function."""

    def test_aggregate_configs_combines_builders(self):
        """aggregate_configs should combine multiple builder outputs."""
        def model_builder():
            return {'alpha': 1.0, 'fit_intercept': True}

        def validation_builder():
            return {'tolerance': 1e-6, 'strict': True}

        result = aggregate_configs({
            'model': model_builder,
            'validation': validation_builder,
        })

        assert 'model' in result
        assert 'validation' in result
        assert result['model']['alpha'] == 1.0
        assert result['validation']['tolerance'] == 1e-6

    def test_aggregate_configs_empty_builders(self):
        """aggregate_configs with empty dict should return empty dict."""
        result = aggregate_configs({})
        assert result == {}

    def test_aggregate_configs_single_builder(self):
        """aggregate_configs with single builder should work."""
        def single_builder():
            return {'key': 'value'}

        result = aggregate_configs({'section': single_builder})
        assert result == {'section': {'key': 'value'}}


# =============================================================================
# with_defaults() Tests
# =============================================================================


class TestWithDefaults:
    """Tests for with_defaults function."""

    def test_with_defaults_shallow_merge_user_overrides(self):
        """User config should override defaults in shallow merge."""
        defaults = {'n_samples': 1000, 'random_state': 42, 'enabled': True}
        user_config = {'n_samples': 500}

        result = with_defaults(user_config, defaults)

        assert result['n_samples'] == 500  # User override
        assert result['random_state'] == 42  # Default preserved
        assert result['enabled'] is True  # Default preserved

    def test_with_defaults_shallow_merge_does_not_mutate_inputs(self):
        """Shallow merge should not mutate original dicts."""
        defaults = {'a': 1, 'b': 2}
        user_config = {'a': 999}

        _ = with_defaults(user_config, defaults)

        assert defaults['a'] == 1  # Original unchanged
        assert user_config['a'] == 999  # Original unchanged

    def test_with_defaults_deep_merge_recursive(self):
        """Deep merge should recursively merge nested dicts."""
        defaults = {
            'model': {'alpha': 1.0, 'beta': 0.5},
            'validation': {'strict': True}
        }
        user_config = {
            'model': {'alpha': 2.0}  # Override alpha, keep beta
        }

        result = with_defaults(user_config, defaults, deep_merge=True)

        assert result['model']['alpha'] == 2.0  # Overridden
        assert result['model']['beta'] == 0.5  # Preserved from defaults
        assert result['validation']['strict'] is True  # Preserved

    def test_with_defaults_deep_merge_user_adds_new_keys(self):
        """Deep merge should allow user to add new nested keys."""
        defaults = {'model': {'alpha': 1.0}}
        user_config = {'model': {'gamma': 0.1}}

        result = with_defaults(user_config, defaults, deep_merge=True)

        assert result['model']['alpha'] == 1.0  # Default preserved
        assert result['model']['gamma'] == 0.1  # New key added

    def test_with_defaults_deep_merge_replaces_non_dict_values(self):
        """Deep merge should replace non-dict values entirely."""
        defaults = {'model': {'alpha': 1.0}, 'count': 10}
        user_config = {'count': 20}

        result = with_defaults(user_config, defaults, deep_merge=True)

        assert result['count'] == 20  # Non-dict replaced entirely

    def test_with_defaults_empty_user_config_returns_defaults(self):
        """Empty user config should return defaults."""
        defaults = {'a': 1, 'b': 2}

        result = with_defaults({}, defaults)

        assert result == defaults


# =============================================================================
# build_config_section() Tests
# =============================================================================


class TestBuildConfigSection:
    """Tests for build_config_section function."""

    def test_build_config_section_wraps_with_name(self):
        """build_config_section should wrap config with section name."""
        result = build_config_section('bootstrap', SampleConfig, n_samples=1000)

        assert 'bootstrap' in result
        assert result['bootstrap']['n_samples'] == 1000

    def test_build_config_section_empty_kwargs(self):
        """build_config_section with no kwargs should create empty section."""
        result = build_config_section('empty_section', SampleConfig)

        assert 'empty_section' in result
        assert result['empty_section'] == {}


# =============================================================================
# validate_config_keys() Tests
# =============================================================================


class TestValidateConfigKeys:
    """Tests for validate_config_keys function."""

    def test_validate_config_keys_all_required_present(self):
        """validate_config_keys should return empty list when all required present."""
        config = {'n_samples': 1000, 'random_state': 42}
        required = ['n_samples', 'random_state']

        errors = validate_config_keys(config, required)

        assert errors == []

    def test_validate_config_keys_missing_required_key(self):
        """validate_config_keys should report missing required key."""
        config = {'n_samples': 1000}
        required = ['n_samples', 'random_state']

        errors = validate_config_keys(config, required)

        assert len(errors) == 1
        assert 'random_state' in errors[0]
        assert 'Missing' in errors[0]

    def test_validate_config_keys_multiple_missing(self):
        """validate_config_keys should report all missing keys."""
        config = {}
        required = ['a', 'b', 'c']

        errors = validate_config_keys(config, required)

        assert len(errors) == 3

    def test_validate_config_keys_unknown_key_with_optional(self):
        """validate_config_keys should report unknown keys when optional_keys specified."""
        config = {'n_samples': 1000, 'unknown_key': 'value'}
        required = ['n_samples']
        optional = ['random_state']

        errors = validate_config_keys(config, required, optional_keys=optional)

        assert len(errors) == 1
        assert 'unknown_key' in errors[0]
        assert 'Unknown' in errors[0]

    def test_validate_config_keys_optional_keys_allowed(self):
        """validate_config_keys should allow optional keys."""
        config = {'n_samples': 1000, 'random_state': 42}
        required = ['n_samples']
        optional = ['random_state', 'enabled']

        errors = validate_config_keys(config, required, optional_keys=optional)

        assert errors == []

    def test_validate_config_keys_no_optional_ignores_unknown(self):
        """validate_config_keys without optional_keys should ignore unknown keys."""
        config = {'n_samples': 1000, 'extra_key': 'ignored'}
        required = ['n_samples']

        errors = validate_config_keys(config, required)

        assert errors == []  # Unknown keys ignored when optional_keys=None


# =============================================================================
# chain_builders() Tests
# =============================================================================


class TestChainBuilders:
    """Tests for chain_builders function."""

    def test_chain_builders_sequences_builders(self):
        """chain_builders should apply builders in sequence."""
        def add_model(config):
            config['model'] = 'ridge'
            return config

        def add_alpha(config):
            config['alpha'] = 1.0
            return config

        result = chain_builders(add_model, add_alpha)

        assert result['model'] == 'ridge'
        assert result['alpha'] == 1.0

    def test_chain_builders_with_initial_config(self):
        """chain_builders should start with initial config."""
        def add_more(config):
            config['added'] = True
            return config

        initial = {'existing': 'value'}
        result = chain_builders(add_more, initial=initial)

        assert result['existing'] == 'value'
        assert result['added'] is True

    def test_chain_builders_no_builders(self):
        """chain_builders with no builders should return initial or empty."""
        result = chain_builders()
        assert result == {}

        result_with_initial = chain_builders(initial={'key': 'value'})
        assert result_with_initial == {'key': 'value'}

    def test_chain_builders_mutates_through_chain(self):
        """chain_builders allows each builder to see previous changes."""
        def first(config):
            config['first'] = 1
            return config

        def second(config):
            # Should see 'first' from previous builder
            config['second'] = config.get('first', 0) + 1
            return config

        result = chain_builders(first, second)

        assert result['first'] == 1
        assert result['second'] == 2  # 1 + 1


# =============================================================================
# ConfigBuilderMixin Tests
# =============================================================================


class TestConfigBuilderMixin:
    """Tests for ConfigBuilderMixin class methods."""

    def test_merge_configs_combines_dicts(self):
        """merge_configs should combine multiple dicts."""
        mixin = ConfigBuilderMixin()

        result = mixin.merge_configs({'a': 1}, {'b': 2}, {'c': 3})

        assert result == {'a': 1, 'b': 2, 'c': 3}

    def test_merge_configs_later_overrides_earlier(self):
        """merge_configs later dicts should override earlier ones."""
        result = ConfigBuilderMixin.merge_configs(
            {'a': 1, 'b': 2},
            {'b': 999, 'c': 3}
        )

        assert result['a'] == 1  # Preserved
        assert result['b'] == 999  # Overridden
        assert result['c'] == 3  # Added

    def test_merge_configs_empty_inputs(self):
        """merge_configs with empty dicts should work."""
        result = ConfigBuilderMixin.merge_configs({}, {'a': 1}, {})
        assert result == {'a': 1}

    def test_filter_none_values_removes_nones(self):
        """filter_none_values should remove None values."""
        config = {'a': 1, 'b': None, 'c': 'value', 'd': None}

        result = ConfigBuilderMixin.filter_none_values(config)

        assert result == {'a': 1, 'c': 'value'}
        assert 'b' not in result
        assert 'd' not in result

    def test_filter_none_values_preserves_falsy_non_none(self):
        """filter_none_values should preserve falsy values that aren't None."""
        config = {'a': 0, 'b': False, 'c': '', 'd': None}

        result = ConfigBuilderMixin.filter_none_values(config)

        assert result['a'] == 0
        assert result['b'] is False
        assert result['c'] == ''
        assert 'd' not in result

    def test_add_metadata_adds_section(self):
        """add_metadata should add _metadata section to config."""
        config = {'model': 'ridge'}

        result = ConfigBuilderMixin.add_metadata(config, version='v3')

        assert '_metadata' in result
        assert result['_metadata']['version'] == 'v3'
        assert result['model'] == 'ridge'  # Original preserved

    def test_add_metadata_with_extra_kwargs(self):
        """add_metadata should include extra kwargs in metadata."""
        config = {}

        result = ConfigBuilderMixin.add_metadata(
            config,
            version='v2',
            author='test',
            timestamp='2026-01-30'
        )

        assert result['_metadata']['version'] == 'v2'
        assert result['_metadata']['author'] == 'test'
        assert result['_metadata']['timestamp'] == '2026-01-30'

    def test_add_metadata_mutates_original(self):
        """add_metadata mutates and returns the original config dict."""
        config = {'existing': 'value'}

        result = ConfigBuilderMixin.add_metadata(config, version='v1')

        assert result is config  # Same object
        assert '_metadata' in config  # Original mutated
