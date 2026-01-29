"""
Unit tests for src/config/constants.py.

Tests validate that constants module provides correct default values
and type annotations for all configuration parameters.
"""

import pytest

from src.config.constants import (
    # Statistical defaults
    DEFAULT_RANDOM_SEED,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_CONFIDENCE_LEVEL,
    REPORTING_CONFIDENCE_LEVELS,
    # Model hyperparameters
    DEFAULT_RIDGE_ALPHA,
    DEFAULT_N_ESTIMATORS,
    # Data quality thresholds
    MIN_OBSERVATIONS,
    MAX_VIF_THRESHOLD,
    R_SQUARED_WARNING_THRESHOLD,
    R_SQUARED_HALT_THRESHOLD,
    # Feature selection
    DEFAULT_MAX_CANDIDATE_MODELS,
    DEFAULT_MODELS_TO_ANALYZE,
    # Numerical precision
    NUMERICAL_TOLERANCE,
    DEFAULT_DECIMAL_PLACES,
)


class TestStatisticalDefaults:
    """Tests for statistical default constants."""

    def test_default_random_seed_is_42(self):
        """DEFAULT_RANDOM_SEED should be 42 for reproducibility."""
        assert DEFAULT_RANDOM_SEED == 42

    def test_default_n_bootstrap_is_100(self):
        """DEFAULT_N_BOOTSTRAP should be 100."""
        assert DEFAULT_N_BOOTSTRAP == 100

    def test_default_confidence_level_is_095(self):
        """DEFAULT_CONFIDENCE_LEVEL should be 0.95 (95%)."""
        assert DEFAULT_CONFIDENCE_LEVEL == 0.95

    def test_reporting_confidence_levels_is_tuple(self):
        """REPORTING_CONFIDENCE_LEVELS should be an immutable tuple."""
        assert isinstance(REPORTING_CONFIDENCE_LEVELS, tuple)
        assert REPORTING_CONFIDENCE_LEVELS == (0.80, 0.90, 0.95)

    def test_confidence_levels_are_valid_probabilities(self):
        """All confidence levels should be in (0, 1)."""
        for level in REPORTING_CONFIDENCE_LEVELS:
            assert 0.0 < level < 1.0, f"Invalid probability: {level}"


class TestModelHyperparameters:
    """Tests for model hyperparameter constants."""

    def test_default_ridge_alpha_is_positive(self):
        """DEFAULT_RIDGE_ALPHA should be positive."""
        assert DEFAULT_RIDGE_ALPHA > 0
        assert DEFAULT_RIDGE_ALPHA == 1.0

    def test_default_n_estimators_is_positive_integer(self):
        """DEFAULT_N_ESTIMATORS should be positive integer."""
        assert isinstance(DEFAULT_N_ESTIMATORS, int)
        assert DEFAULT_N_ESTIMATORS > 0
        assert DEFAULT_N_ESTIMATORS == 100


class TestDataQualityThresholds:
    """Tests for data quality threshold constants."""

    def test_min_observations_is_30(self):
        """MIN_OBSERVATIONS should be 30 for statistical validity."""
        assert MIN_OBSERVATIONS == 30

    def test_max_vif_threshold_is_10(self):
        """MAX_VIF_THRESHOLD should be 10.0 (standard threshold)."""
        assert MAX_VIF_THRESHOLD == 10.0

    def test_r_squared_warning_threshold_is_095(self):
        """R_SQUARED_WARNING_THRESHOLD should be 0.95."""
        assert R_SQUARED_WARNING_THRESHOLD == 0.95

    def test_r_squared_halt_threshold_is_099(self):
        """R_SQUARED_HALT_THRESHOLD should be 0.99."""
        assert R_SQUARED_HALT_THRESHOLD == 0.99

    def test_warning_threshold_less_than_halt_threshold(self):
        """Warning threshold should be less than halt threshold."""
        assert R_SQUARED_WARNING_THRESHOLD < R_SQUARED_HALT_THRESHOLD


class TestFeatureSelectionConstants:
    """Tests for feature selection constants."""

    def test_default_max_candidate_models_is_100(self):
        """DEFAULT_MAX_CANDIDATE_MODELS should be 100."""
        assert DEFAULT_MAX_CANDIDATE_MODELS == 100

    def test_default_models_to_analyze_is_10(self):
        """DEFAULT_MODELS_TO_ANALYZE should be 10."""
        assert DEFAULT_MODELS_TO_ANALYZE == 10


class TestNumericalPrecisionConstants:
    """Tests for numerical precision constants."""

    def test_numerical_tolerance_is_1e12(self):
        """NUMERICAL_TOLERANCE should be 1e-12 per CLAUDE.md."""
        assert NUMERICAL_TOLERANCE == 1e-12

    def test_default_decimal_places_is_4(self):
        """DEFAULT_DECIMAL_PLACES should be 4."""
        assert DEFAULT_DECIMAL_PLACES == 4


class TestTypeAnnotations:
    """Tests for type annotation correctness."""

    def test_integer_constants_are_int(self):
        """Integer constants should have int type."""
        int_constants = [
            DEFAULT_RANDOM_SEED,
            DEFAULT_N_BOOTSTRAP,
            DEFAULT_N_ESTIMATORS,
            MIN_OBSERVATIONS,
            DEFAULT_MAX_CANDIDATE_MODELS,
            DEFAULT_MODELS_TO_ANALYZE,
            DEFAULT_DECIMAL_PLACES,
        ]
        for const in int_constants:
            assert isinstance(const, int), f"{const} should be int"

    def test_float_constants_are_float(self):
        """Float constants should have float type."""
        float_constants = [
            DEFAULT_CONFIDENCE_LEVEL,
            DEFAULT_RIDGE_ALPHA,
            MAX_VIF_THRESHOLD,
            R_SQUARED_WARNING_THRESHOLD,
            R_SQUARED_HALT_THRESHOLD,
            NUMERICAL_TOLERANCE,
        ]
        for const in float_constants:
            assert isinstance(const, float), f"{const} should be float"


class TestConstantsImmutability:
    """Tests for constant immutability (where applicable)."""

    def test_reporting_confidence_levels_is_immutable(self):
        """REPORTING_CONFIDENCE_LEVELS should be immutable tuple."""
        assert isinstance(REPORTING_CONFIDENCE_LEVELS, tuple)
        with pytest.raises(TypeError):
            # Tuples don't support item assignment
            REPORTING_CONFIDENCE_LEVELS[0] = 0.99  # type: ignore
