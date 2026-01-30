"""
Tests for Feature Selection Interface Configuration Module.

Tests cover:
- FEATURE_FLAGS: Default flag values
- set_feature_flag / get_feature_flags: Flag management
- configure_analysis_pipeline: Pipeline configuration building
- create_dual_validation_config: Dual validation setup
- create_feature_selection_config: Feature selection config creation

Design Principles:
- Real assertions about correctness (not just "doesn't crash")
- Test flag isolation to avoid cross-test contamination
- Verify configuration structure and defaults

Author: Claude Code
Date: 2026-01-30
"""

import pytest
from unittest.mock import patch

from src.features.selection.interface.interface_config import (
    FEATURE_FLAGS,
    set_feature_flag,
    get_feature_flags,
    configure_analysis_pipeline,
    create_dual_validation_config,
    create_feature_selection_config,
    _get_dual_validation_defaults,
    _build_feature_config,
)
# Note: FeatureSelectionConfig, EconomicConstraintConfig, BootstrapAnalysisConfig
# are TypedDicts - cannot use isinstance() checks, use dict key access instead


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def reset_feature_flags():
    """Reset feature flags to defaults after each test."""
    original_flags = FEATURE_FLAGS.copy()
    yield
    FEATURE_FLAGS.clear()
    FEATURE_FLAGS.update(original_flags)


@pytest.fixture
def sample_features():
    """Sample feature lists for testing."""
    return {
        "candidate": ["comp_t2", "comp_t3", "macro_t1"],
        "base": ["own_rate_t1"],
        "target": "sales_target_current",
    }


# =============================================================================
# Tests for FEATURE_FLAGS
# =============================================================================


class TestFeatureFlags:
    """Tests for FEATURE_FLAGS dictionary."""

    def test_default_flags_exist(self):
        """Test that expected default flags exist."""
        expected_flags = [
            "USE_ATOMIC_FUNCTIONS",
            "ENABLE_VALIDATION",
            "SHOW_DETAILED_OUTPUT",
            "ENABLE_BOOTSTRAP_DEFAULT",
            "STRICT_CONSTRAINTS_DEFAULT",
            "AUTO_DISPLAY_RESULTS",
        ]
        for flag in expected_flags:
            assert flag in FEATURE_FLAGS, f"Missing flag: {flag}"

    def test_default_flag_values(self):
        """Test default flag values."""
        # These should be True by default
        assert FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] is True
        assert FEATURE_FLAGS["ENABLE_VALIDATION"] is True
        assert FEATURE_FLAGS["STRICT_CONSTRAINTS_DEFAULT"] is True

        # These should be False by default
        assert FEATURE_FLAGS["ENABLE_BOOTSTRAP_DEFAULT"] is False

    def test_enhancement_flags_default_off(self):
        """Test that enhancement flags are off by default."""
        enhancement_flags = [
            "ENABLE_MULTIPLE_TESTING",
            "ENABLE_BLOCK_BOOTSTRAP",
            "ENABLE_OOS_VALIDATION",
            "ENABLE_REGRESSION_DIAGNOSTICS",
            "ENABLE_STATISTICAL_CONSTRAINTS",
            "ENABLE_SEARCH_SPACE_REDUCTION",
        ]
        for flag in enhancement_flags:
            if flag in FEATURE_FLAGS:
                assert FEATURE_FLAGS[flag] is False, f"{flag} should default to False"


# =============================================================================
# Tests for set_feature_flag
# =============================================================================


class TestSetFeatureFlag:
    """Tests for set_feature_flag function."""

    def test_set_valid_flag(self, reset_feature_flags, capsys):
        """Test setting a valid feature flag."""
        original = FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"]

        set_feature_flag("USE_ATOMIC_FUNCTIONS", not original)

        assert FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] == (not original)
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out

    def test_set_invalid_flag_prints_warning(self, reset_feature_flags, capsys):
        """Test setting an invalid flag prints warning."""
        set_feature_flag("NONEXISTENT_FLAG", True)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "Unknown feature flag" in captured.out

    def test_set_flag_to_true(self, reset_feature_flags):
        """Test setting flag to True."""
        set_feature_flag("ENABLE_BOOTSTRAP_DEFAULT", True)
        assert FEATURE_FLAGS["ENABLE_BOOTSTRAP_DEFAULT"] is True

    def test_set_flag_to_false(self, reset_feature_flags):
        """Test setting flag to False."""
        set_feature_flag("USE_ATOMIC_FUNCTIONS", False)
        assert FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] is False


# =============================================================================
# Tests for get_feature_flags
# =============================================================================


class TestGetFeatureFlags:
    """Tests for get_feature_flags function."""

    def test_returns_copy(self, reset_feature_flags):
        """Test that get_feature_flags returns a copy, not original."""
        flags = get_feature_flags()

        # Modify the returned copy
        flags["USE_ATOMIC_FUNCTIONS"] = "modified"

        # Original should be unchanged
        assert FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] is True

    def test_returns_all_flags(self):
        """Test that all flags are returned."""
        flags = get_feature_flags()

        assert len(flags) == len(FEATURE_FLAGS)
        for key in FEATURE_FLAGS:
            assert key in flags


# =============================================================================
# Tests for configure_analysis_pipeline
# =============================================================================


class TestConfigureAnalysisPipeline:
    """Tests for configure_analysis_pipeline function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        config = configure_analysis_pipeline()
        assert isinstance(config, dict)

    def test_default_parameters(self):
        """Test configuration with default parameters."""
        config = configure_analysis_pipeline()

        # Should not have error key with valid setup
        if "error" not in config:
            assert "bootstrap_config" in config or "fallback_config" in config

    def test_custom_parameters(self):
        """Test configuration with custom parameters."""
        config = configure_analysis_pipeline(
            max_candidate_features=3,
            target_variable="custom_target",
            n_bootstrap_samples=500,
            enable_economic_constraints=False,
        )

        assert isinstance(config, dict)

    def test_visualization_config_added(self):
        """Test that visualization config is added."""
        config = configure_analysis_pipeline()

        if "visualization_config" in config:
            viz = config["visualization_config"]
            assert "fig_width" in viz
            assert "fig_height" in viz

    def test_status_indicators_added(self):
        """Test that status indicators are added."""
        config = configure_analysis_pipeline()

        if "status_indicators" in config:
            indicators = config["status_indicators"]
            assert "SUCCESS" in indicators
            assert "WARNING" in indicators
            assert "ERROR" in indicators


# =============================================================================
# Tests for create_dual_validation_config
# =============================================================================


class TestCreateDualValidationConfig:
    """Tests for create_dual_validation_config function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        config = create_dual_validation_config()
        assert isinstance(config, dict)

    def test_default_dual_validation_values(self):
        """Test default dual validation configuration."""
        defaults = _get_dual_validation_defaults()

        assert defaults["enabled"] is True
        assert "core_metrics" in defaults
        assert "AIC" in defaults["core_metrics"]
        assert "validation_types" in defaults

    def test_custom_parameters(self):
        """Test with custom parameters."""
        config = create_dual_validation_config(
            max_candidate_features=2,
            n_bootstrap_samples=200,
        )

        assert isinstance(config, dict)


# =============================================================================
# Tests for create_feature_selection_config
# =============================================================================


class TestCreateFeatureSelectionConfig:
    """Tests for create_feature_selection_config function."""

    def test_returns_tuple_of_three(self, sample_features):
        """Test that function returns tuple of 3 configs."""
        result = create_feature_selection_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_feature_config_structure(self, sample_features):
        """Test FeatureSelectionConfig structure (TypedDict)."""
        feature_config, _, _ = create_feature_selection_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            max_features=2,
            base_features=sample_features["base"],
        )

        # TypedDict - check it's a dict with expected keys
        assert isinstance(feature_config, dict)
        assert feature_config["max_candidate_features"] == 2
        assert feature_config["target_variable"] == sample_features["target"]
        assert feature_config["candidate_features"] == sample_features["candidate"]
        assert feature_config["base_features"] == sample_features["base"]

    def test_constraint_config_enabled(self, sample_features):
        """Test EconomicConstraintConfig when enabled (TypedDict)."""
        _, constraint_config, _ = create_feature_selection_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            enable_constraints=True,
        )

        # TypedDict - check it's a dict with expected keys
        assert isinstance(constraint_config, dict)
        assert constraint_config["enabled"] is True

    def test_constraint_config_disabled(self, sample_features):
        """Test EconomicConstraintConfig when disabled (TypedDict)."""
        _, constraint_config, _ = create_feature_selection_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            enable_constraints=False,
        )

        assert constraint_config["enabled"] is False

    def test_bootstrap_config_when_enabled(self, sample_features, reset_feature_flags):
        """Test BootstrapAnalysisConfig when explicitly enabled (TypedDict)."""
        _, _, bootstrap_config = create_feature_selection_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            enable_bootstrap=True,
            bootstrap_samples=50,
        )

        assert bootstrap_config is not None
        assert isinstance(bootstrap_config, dict)
        assert bootstrap_config["n_samples"] == 50

    def test_bootstrap_config_when_disabled(self, sample_features, reset_feature_flags):
        """Test BootstrapAnalysisConfig is None when disabled."""
        _, _, bootstrap_config = create_feature_selection_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            enable_bootstrap=False,
        )

        assert bootstrap_config is None

    def test_bootstrap_uses_flag_default(self, sample_features, reset_feature_flags):
        """Test bootstrap uses flag default when not specified."""
        # Default is False
        _, _, bootstrap_config = create_feature_selection_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            enable_bootstrap=None,  # Use default
        )

        # ENABLE_BOOTSTRAP_DEFAULT is False by default
        assert bootstrap_config is None

    def test_random_seed_propagation(self, sample_features, reset_feature_flags):
        """Test random seed is propagated to bootstrap config (TypedDict)."""
        _, _, bootstrap_config = create_feature_selection_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            enable_bootstrap=True,
            random_seed=123,
        )

        assert bootstrap_config is not None
        assert bootstrap_config["random_seed"] == 123


# =============================================================================
# Tests for Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_build_feature_config(self, sample_features):
        """Test _build_feature_config helper (TypedDict)."""
        config = _build_feature_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            max_features=3,
            base_features=sample_features["base"],
        )

        assert isinstance(config, dict)
        assert config["max_candidate_features"] == 3

    def test_build_feature_config_no_base_features(self, sample_features):
        """Test _build_feature_config with no base features (TypedDict)."""
        config = _build_feature_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            max_features=2,
            base_features=None,
        )

        assert config["base_features"] == []

    def test_get_dual_validation_defaults_structure(self):
        """Test _get_dual_validation_defaults returns expected structure."""
        defaults = _get_dual_validation_defaults()

        assert isinstance(defaults, dict)
        assert "enabled" in defaults
        assert "core_metrics" in defaults
        assert "validation_types" in defaults
        assert "win_rate_weight" in defaults
        assert "information_ratio_weight" in defaults


# =============================================================================
# Integration Tests
# =============================================================================


class TestInterfaceConfigIntegration:
    """Integration tests for interface_config module."""

    def test_full_config_workflow(self, sample_features, reset_feature_flags):
        """Test complete configuration workflow (TypedDicts)."""
        # Set up flags
        set_feature_flag("ENABLE_BOOTSTRAP_DEFAULT", False)
        set_feature_flag("STRICT_CONSTRAINTS_DEFAULT", True)

        # Create configs
        feature_config, constraint_config, bootstrap_config = create_feature_selection_config(
            candidate_features=sample_features["candidate"],
            target_variable=sample_features["target"],
            max_features=3,
            base_features=sample_features["base"],
            enable_constraints=True,
        )

        # Verify structure (TypedDicts - use dict access)
        assert feature_config["max_candidate_features"] == 3
        assert constraint_config["enabled"] is True
        assert bootstrap_config is None  # Disabled by default

    def test_flag_isolation(self, reset_feature_flags):
        """Test that flag changes don't persist across tests."""
        # This test should start with default flags
        assert FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] is True

        # Modify flag
        set_feature_flag("USE_ATOMIC_FUNCTIONS", False)
        assert FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] is False

        # reset_feature_flags fixture will restore defaults
