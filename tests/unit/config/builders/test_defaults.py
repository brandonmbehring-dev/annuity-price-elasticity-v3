"""
Unit tests for src/config/builders/defaults.py.

Tests validate default configuration accessors and data integrity.
"""

import pytest
from typing import Dict, Any

from src.config.builders.defaults import (
    # Constants
    DEFAULTS,
    BOOTSTRAP_DEFAULTS,
    CROSS_VALIDATION_DEFAULTS,
    RIDGE_DEFAULTS,
    BUSINESS_FILTER_DEFAULTS,
    ECONOMIC_CONSTRAINT_DEFAULTS,
    INFERENCE_DEFAULTS,
    RATE_SCENARIO_DEFAULTS,
    CONFIDENCE_INTERVAL_DEFAULTS,
    TABLEAU_FORMATTING_DEFAULTS,
    VISUALIZATION_DEFAULTS,
    VALIDATION_FRAMEWORK_DEFAULTS,
    PRODUCT_DEFAULTS,
    # Feature tuples
    COMPETITIVE_FEATURES,
    INFERENCE_FEATURES,
    MODEL_FEATURES,
    BENCHMARK_FEATURES,
    BASE_FEATURES,
    # Functions
    get_default,
    get_product_defaults,
    get_feature_list,
)


# =============================================================================
# DEFAULTS Dictionary Structure Tests
# =============================================================================


class TestDefaultsDictionaryStructure:
    """Validate DEFAULTS dictionary contains expected categories and structure."""

    def test_defaults_contains_all_expected_categories(self):
        """DEFAULTS should contain all expected category keys."""
        expected_categories = [
            'bootstrap',
            'cross_validation',
            'ridge',
            'business_filter',
            'economic_constraint',
            'inference',
            'rate_scenario',
            'confidence_interval',
            'tableau_formatting',
            'visualization',
            'validation_framework',
            'competitive_features',
            'inference_features',
            'model_features',
            'benchmark_features',
            'base_features',
            'products',
        ]
        for category in expected_categories:
            assert category in DEFAULTS, f"Missing category: {category}"

    def test_defaults_categories_are_correct_types(self):
        """Each DEFAULTS category should be correct type."""
        # Dict categories
        dict_categories = [
            'bootstrap', 'cross_validation', 'ridge', 'business_filter',
            'economic_constraint', 'inference', 'rate_scenario',
            'confidence_interval', 'tableau_formatting', 'visualization',
            'validation_framework', 'products',
        ]
        for category in dict_categories:
            assert isinstance(DEFAULTS[category], dict), f"{category} should be dict"

        # Tuple categories (feature lists)
        tuple_categories = [
            'competitive_features', 'inference_features', 'model_features',
            'benchmark_features', 'base_features',
        ]
        for category in tuple_categories:
            assert isinstance(DEFAULTS[category], tuple), f"{category} should be tuple"


class TestBootstrapDefaults:
    """Tests for BOOTSTRAP_DEFAULTS values."""

    def test_bootstrap_n_samples_is_positive_int(self):
        """Bootstrap n_samples should be positive integer."""
        assert BOOTSTRAP_DEFAULTS['n_samples'] > 0
        assert isinstance(BOOTSTRAP_DEFAULTS['n_samples'], int)

    def test_bootstrap_random_state_is_defined(self):
        """Bootstrap should have random_state for reproducibility."""
        assert 'random_state' in BOOTSTRAP_DEFAULTS
        assert BOOTSTRAP_DEFAULTS['random_state'] == 42

    def test_bootstrap_confidence_level_in_valid_range(self):
        """Bootstrap confidence_level should be between 0 and 1."""
        cl = BOOTSTRAP_DEFAULTS['confidence_level']
        assert 0 < cl < 1


class TestInferenceDefaults:
    """Tests for INFERENCE_DEFAULTS values."""

    def test_inference_target_variable_follows_naming_convention(self):
        """Target variable should use unified _t0 naming."""
        target = INFERENCE_DEFAULTS['target_variable']
        assert target.endswith('_t0'), "Target should use _t0 suffix (unified naming)"


# =============================================================================
# get_default() Tests
# =============================================================================


class TestGetDefault:
    """Tests for get_default() function."""

    def test_get_default_unknown_category_raises_keyerror(self):
        """get_default should raise KeyError for unknown category."""
        with pytest.raises(KeyError, match="Unknown default category"):
            get_default('nonexistent_category')

    def test_get_default_returns_entire_category_dict(self):
        """get_default without key should return entire category dict."""
        result = get_default('bootstrap')
        assert isinstance(result, dict)
        assert result == {**BOOTSTRAP_DEFAULTS}

    def test_get_default_with_specific_key_returns_value(self):
        """get_default with key should return that key's value."""
        result = get_default('bootstrap', 'n_samples')
        assert result == BOOTSTRAP_DEFAULTS['n_samples']

    def test_get_default_with_overrides_merges_values(self):
        """get_default with overrides should merge into dict."""
        result = get_default('bootstrap', n_samples=500, random_state=123)
        assert result['n_samples'] == 500
        assert result['random_state'] == 123
        # Original values preserved for non-overridden keys
        assert result['confidence_level'] == BOOTSTRAP_DEFAULTS['confidence_level']

    def test_get_default_override_does_not_mutate_original(self):
        """get_default overrides should not mutate DEFAULTS."""
        original = BOOTSTRAP_DEFAULTS['n_samples']
        _ = get_default('bootstrap', n_samples=999)
        assert BOOTSTRAP_DEFAULTS['n_samples'] == original

    def test_get_default_sequence_returns_tuple(self):
        """get_default for feature category should return tuple."""
        result = get_default('competitive_features')
        assert isinstance(result, tuple)
        assert result == COMPETITIVE_FEATURES

    def test_get_default_sequence_with_key_raises_typeerror(self):
        """get_default on tuple with key should raise TypeError."""
        with pytest.raises(TypeError, match="Cannot access key"):
            get_default('competitive_features', 'some_key')

    def test_get_default_sequence_with_overrides_raises_typeerror(self):
        """get_default on tuple with overrides should raise TypeError."""
        with pytest.raises(TypeError, match="Cannot apply overrides"):
            get_default('competitive_features', foo='bar')

    def test_get_default_key_with_override_returns_override(self):
        """When key is in overrides, get_default should return override value."""
        # This tests the specific branch: overrides.get(key, defaults.get(key))
        result = get_default('bootstrap', 'n_samples', n_samples=999)
        assert result == 999


# =============================================================================
# get_product_defaults() Tests
# =============================================================================


class TestGetProductDefaults:
    """Tests for get_product_defaults() function."""

    def test_get_product_defaults_valid_product_returns_dict(self):
        """get_product_defaults for valid product should return dict."""
        result = get_product_defaults('FlexGuard_6Y20B')
        assert isinstance(result, dict)
        assert result['product_name'] == 'FlexGuard_6Y20B'
        assert result['buffer'] == 20
        assert result['term_years'] == 6
        assert result['product_type'] == 'rila'

    def test_get_product_defaults_returns_copy_not_reference(self):
        """get_product_defaults should return copy, not original."""
        result = get_product_defaults('FlexGuard_6Y20B')
        result['buffer'] = 999  # Mutate returned dict
        # Original should be unchanged
        assert PRODUCT_DEFAULTS['FlexGuard_6Y20B']['buffer'] == 20

    def test_get_product_defaults_invalid_product_raises_keyerror(self):
        """get_product_defaults for unknown product should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown product"):
            get_product_defaults('NonexistentProduct')

    def test_get_product_defaults_all_products_valid(self):
        """All products in PRODUCT_DEFAULTS should be retrievable."""
        for product_code in PRODUCT_DEFAULTS.keys():
            result = get_product_defaults(product_code)
            assert 'product_name' in result
            assert 'buffer' in result
            assert 'term_years' in result
            assert 'product_type' in result


# =============================================================================
# get_feature_list() Tests
# =============================================================================


class TestGetFeatureList:
    """Tests for get_feature_list() function."""

    @pytest.mark.parametrize("category,expected_tuple", [
        ('competitive', COMPETITIVE_FEATURES),
        ('inference', INFERENCE_FEATURES),
        ('model', MODEL_FEATURES),
        ('benchmark', BENCHMARK_FEATURES),
        ('base', BASE_FEATURES),
    ])
    def test_get_feature_list_returns_list_for_category(self, category, expected_tuple):
        """get_feature_list should return list for each valid category."""
        result = get_feature_list(category)
        assert isinstance(result, list)
        assert result == list(expected_tuple)

    def test_get_feature_list_invalid_category_raises_keyerror(self):
        """get_feature_list for unknown category should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown feature category"):
            get_feature_list('invalid_category')

    def test_get_feature_list_returns_new_list_each_call(self):
        """get_feature_list should return new list, not cached reference."""
        list1 = get_feature_list('competitive')
        list2 = get_feature_list('competitive')
        # Should be equal but not same object
        assert list1 == list2
        assert list1 is not list2


# =============================================================================
# Feature Constants Validation
# =============================================================================


class TestFeatureConstants:
    """Validate feature constant tuples contain valid feature names."""

    def test_competitive_features_use_correct_naming(self):
        """COMPETITIVE_FEATURES should use _t{N} naming convention."""
        for feature in COMPETITIVE_FEATURES:
            assert '_t' in feature, f"Feature {feature} should use _t{N} naming"

    def test_inference_features_include_prudential_rate(self):
        """INFERENCE_FEATURES should include prudential_rate."""
        assert any('prudential_rate' in f for f in INFERENCE_FEATURES)

    def test_model_features_nonempty(self):
        """MODEL_FEATURES should not be empty."""
        assert len(MODEL_FEATURES) > 0

    def test_base_features_contains_own_rate(self):
        """BASE_FEATURES should contain the own rate feature."""
        assert 'prudential_rate_t0' in BASE_FEATURES


# =============================================================================
# Economic Constraint Defaults Tests
# =============================================================================


class TestEconomicConstraintDefaults:
    """Tests for ECONOMIC_CONSTRAINT_DEFAULTS values."""

    def test_competitor_sign_is_negative(self):
        """Competitor sign should be negative (substitution effect)."""
        assert ECONOMIC_CONSTRAINT_DEFAULTS['competitor_sign'] == 'negative'

    def test_own_rate_sign_is_positive(self):
        """Own rate sign should be positive (higher rates attract)."""
        assert ECONOMIC_CONSTRAINT_DEFAULTS['own_rate_sign'] == 'positive'

    def test_constraints_enabled_by_default(self):
        """Economic constraints should be enabled by default."""
        assert ECONOMIC_CONSTRAINT_DEFAULTS['enabled'] is True
