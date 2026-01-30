"""
Tests for src.validation.config_schemas module.

Comprehensive tests for Pydantic configuration schemas including:
- Model instantiation with valid parameters
- Boundary violations (field constraints)
- Validator logic (custom validation rules)
- Backward compatibility functions
- Extra fields prevention

Coverage Target: 60%+ of config_schemas.py (201 lines)
"""

import warnings
from typing import Dict, Any

import pytest
from pydantic import ValidationError

from src.validation.config_schemas import (
    ForecastingConfigValidated,
    BootstrapModelConfigValidated,
    CrossValidationConfigValidated,
    build_forecasting_config_validated,
    build_forecasting_config,
)


# =============================================================================
# FORECASTING CONFIG TESTS
# =============================================================================


class TestForecastingConfigValidated:
    """Tests for ForecastingConfigValidated Pydantic model."""

    @pytest.fixture
    def valid_forecasting_config(self) -> Dict[str, Any]:
        """Valid configuration dictionary for testing."""
        return {
            'n_bootstrap_samples': 100,
            'ridge_alpha': 1.0,
            'random_state': 42,
            'exclude_holidays': True,
            'mature_data_cutoff_days': 50,
            'min_training_cutoff': 30
        }

    def test_valid_config_creates_instance(self, valid_forecasting_config):
        """Valid configuration should create ForecastingConfigValidated instance."""
        config = ForecastingConfigValidated(**valid_forecasting_config)

        assert config.n_bootstrap_samples == 100
        assert config.ridge_alpha == 1.0
        assert config.random_state == 42
        assert config.exclude_holidays is True
        assert config.mature_data_cutoff_days == 50
        assert config.min_training_cutoff == 30

    def test_default_exclude_holidays(self):
        """exclude_holidays should default to True."""
        config = ForecastingConfigValidated(
            n_bootstrap_samples=100,
            ridge_alpha=1.0,
            random_state=42,
            mature_data_cutoff_days=50,
            min_training_cutoff=30
        )
        assert config.exclude_holidays is True

    def test_dict_method_returns_dict(self, valid_forecasting_config):
        """dict() method should return dictionary for backward compatibility."""
        config = ForecastingConfigValidated(**valid_forecasting_config)
        result = config.dict()

        assert isinstance(result, dict)
        assert result['n_bootstrap_samples'] == 100
        assert result['ridge_alpha'] == 1.0

    # -------------------------------------------------------------------------
    # Boundary Violations - n_bootstrap_samples
    # -------------------------------------------------------------------------

    def test_bootstrap_samples_zero_fails(self, valid_forecasting_config):
        """n_bootstrap_samples = 0 should fail (gt=0 constraint)."""
        valid_forecasting_config['n_bootstrap_samples'] = 0
        with pytest.raises(ValidationError) as exc_info:
            ForecastingConfigValidated(**valid_forecasting_config)
        assert 'n_bootstrap_samples' in str(exc_info.value)

    def test_bootstrap_samples_negative_fails(self, valid_forecasting_config):
        """n_bootstrap_samples < 0 should fail."""
        valid_forecasting_config['n_bootstrap_samples'] = -10
        with pytest.raises(ValidationError):
            ForecastingConfigValidated(**valid_forecasting_config)

    def test_bootstrap_samples_exceeds_max_fails(self, valid_forecasting_config):
        """n_bootstrap_samples > 1000 should fail (le=1000 constraint)."""
        valid_forecasting_config['n_bootstrap_samples'] = 1001
        with pytest.raises(ValidationError) as exc_info:
            ForecastingConfigValidated(**valid_forecasting_config)
        assert 'n_bootstrap_samples' in str(exc_info.value)

    def test_bootstrap_samples_less_than_10_fails(self, valid_forecasting_config):
        """n_bootstrap_samples < 10 should fail custom validator."""
        valid_forecasting_config['n_bootstrap_samples'] = 5
        with pytest.raises(ValidationError) as exc_info:
            ForecastingConfigValidated(**valid_forecasting_config)
        assert 'statistical validity' in str(exc_info.value).lower()

    def test_bootstrap_samples_over_500_warns(self, valid_forecasting_config):
        """n_bootstrap_samples > 500 should issue warning."""
        valid_forecasting_config['n_bootstrap_samples'] = 600

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ForecastingConfigValidated(**valid_forecasting_config)

            # Check warning was issued
            assert len(w) >= 1
            assert 'slow' in str(w[0].message).lower()

        assert config.n_bootstrap_samples == 600

    # -------------------------------------------------------------------------
    # Boundary Violations - ridge_alpha
    # -------------------------------------------------------------------------

    def test_ridge_alpha_zero_fails(self, valid_forecasting_config):
        """ridge_alpha = 0 should fail (gt=0 constraint)."""
        valid_forecasting_config['ridge_alpha'] = 0.0
        with pytest.raises(ValidationError) as exc_info:
            ForecastingConfigValidated(**valid_forecasting_config)
        assert 'ridge_alpha' in str(exc_info.value)

    def test_ridge_alpha_negative_fails(self, valid_forecasting_config):
        """ridge_alpha < 0 should fail."""
        valid_forecasting_config['ridge_alpha'] = -1.0
        with pytest.raises(ValidationError):
            ForecastingConfigValidated(**valid_forecasting_config)

    def test_ridge_alpha_exceeds_max_fails(self, valid_forecasting_config):
        """ridge_alpha > 100 should fail (le=100 constraint)."""
        valid_forecasting_config['ridge_alpha'] = 101.0
        with pytest.raises(ValidationError):
            ForecastingConfigValidated(**valid_forecasting_config)

    def test_ridge_alpha_over_10_warns(self, valid_forecasting_config):
        """ridge_alpha > 10 should issue underfitting warning."""
        valid_forecasting_config['ridge_alpha'] = 15.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = ForecastingConfigValidated(**valid_forecasting_config)

            assert len(w) >= 1
            assert 'underfitting' in str(w[0].message).lower()

        assert config.ridge_alpha == 15.0

    # -------------------------------------------------------------------------
    # Boundary Violations - Other Fields
    # -------------------------------------------------------------------------

    def test_random_state_exceeds_max_fails(self, valid_forecasting_config):
        """random_state > 2^32-1 should fail."""
        valid_forecasting_config['random_state'] = 2**32
        with pytest.raises(ValidationError):
            ForecastingConfigValidated(**valid_forecasting_config)

    def test_random_state_negative_fails(self, valid_forecasting_config):
        """random_state < 0 should fail."""
        valid_forecasting_config['random_state'] = -1
        with pytest.raises(ValidationError):
            ForecastingConfigValidated(**valid_forecasting_config)

    def test_mature_data_cutoff_zero_fails(self, valid_forecasting_config):
        """mature_data_cutoff_days = 0 should fail (gt=0)."""
        valid_forecasting_config['mature_data_cutoff_days'] = 0
        with pytest.raises(ValidationError):
            ForecastingConfigValidated(**valid_forecasting_config)

    def test_mature_data_cutoff_exceeds_max_fails(self, valid_forecasting_config):
        """mature_data_cutoff_days > 365 should fail."""
        valid_forecasting_config['mature_data_cutoff_days'] = 400
        with pytest.raises(ValidationError):
            ForecastingConfigValidated(**valid_forecasting_config)

    def test_min_training_cutoff_too_small_fails(self, valid_forecasting_config):
        """min_training_cutoff <= 5 should fail (gt=5)."""
        valid_forecasting_config['min_training_cutoff'] = 5
        with pytest.raises(ValidationError):
            ForecastingConfigValidated(**valid_forecasting_config)

    def test_min_training_cutoff_exceeds_max_fails(self, valid_forecasting_config):
        """min_training_cutoff > 100 should fail."""
        valid_forecasting_config['min_training_cutoff'] = 101
        with pytest.raises(ValidationError):
            ForecastingConfigValidated(**valid_forecasting_config)

    # -------------------------------------------------------------------------
    # Extra Fields Prevention
    # -------------------------------------------------------------------------

    def test_extra_fields_forbidden(self, valid_forecasting_config):
        """Extra fields should raise ValidationError (extra='forbid')."""
        valid_forecasting_config['unknown_param'] = 'value'
        with pytest.raises(ValidationError) as exc_info:
            ForecastingConfigValidated(**valid_forecasting_config)
        assert 'extra fields' in str(exc_info.value).lower() or 'unknown_param' in str(exc_info.value)

    def test_validate_on_assignment(self, valid_forecasting_config):
        """Changing attribute should trigger validation (validate_assignment=True)."""
        config = ForecastingConfigValidated(**valid_forecasting_config)

        with pytest.raises(ValidationError):
            config.n_bootstrap_samples = 5  # Less than 10, should fail


# =============================================================================
# BOOTSTRAP MODEL CONFIG TESTS
# =============================================================================


class TestBootstrapModelConfigValidated:
    """Tests for BootstrapModelConfigValidated Pydantic model."""

    @pytest.fixture
    def valid_bootstrap_config(self) -> Dict[str, Any]:
        """Valid bootstrap model configuration."""
        return {
            'estimator_type': 'Ridge',
            'alpha': 1.0,
            'positive_constraint': True,
            'fit_intercept': True,
            'normalize': False
        }

    def test_valid_config_creates_instance(self, valid_bootstrap_config):
        """Valid configuration should create BootstrapModelConfigValidated instance."""
        config = BootstrapModelConfigValidated(**valid_bootstrap_config)

        assert config.estimator_type == 'Ridge'
        assert config.alpha == 1.0
        assert config.positive_constraint is True
        assert config.fit_intercept is True
        assert config.normalize is False

    def test_default_values(self):
        """Default values should be applied correctly."""
        config = BootstrapModelConfigValidated(alpha=1.0)

        assert config.estimator_type == 'Ridge'  # Default
        assert config.positive_constraint is True  # Default
        assert config.fit_intercept is True  # Default
        assert config.normalize is False  # Default

    def test_estimator_type_lasso(self, valid_bootstrap_config):
        """estimator_type='Lasso' should be valid."""
        valid_bootstrap_config['estimator_type'] = 'Lasso'
        config = BootstrapModelConfigValidated(**valid_bootstrap_config)
        assert config.estimator_type == 'Lasso'

    def test_estimator_type_elasticnet(self, valid_bootstrap_config):
        """estimator_type='ElasticNet' should be valid."""
        valid_bootstrap_config['estimator_type'] = 'ElasticNet'
        config = BootstrapModelConfigValidated(**valid_bootstrap_config)
        assert config.estimator_type == 'ElasticNet'

    def test_invalid_estimator_type_fails(self, valid_bootstrap_config):
        """Invalid estimator_type should fail (Literal constraint)."""
        valid_bootstrap_config['estimator_type'] = 'LinearRegression'
        with pytest.raises(ValidationError):
            BootstrapModelConfigValidated(**valid_bootstrap_config)

    def test_alpha_zero_fails(self, valid_bootstrap_config):
        """alpha = 0 should fail (gt=0 constraint)."""
        valid_bootstrap_config['alpha'] = 0.0
        with pytest.raises(ValidationError):
            BootstrapModelConfigValidated(**valid_bootstrap_config)

    def test_alpha_exceeds_max_fails(self, valid_bootstrap_config):
        """alpha > 100 should fail (le=100 constraint)."""
        valid_bootstrap_config['alpha'] = 101.0
        with pytest.raises(ValidationError):
            BootstrapModelConfigValidated(**valid_bootstrap_config)

    def test_extra_fields_forbidden(self, valid_bootstrap_config):
        """Extra fields should raise ValidationError."""
        valid_bootstrap_config['extra_param'] = True
        with pytest.raises(ValidationError):
            BootstrapModelConfigValidated(**valid_bootstrap_config)


# =============================================================================
# CROSS VALIDATION CONFIG TESTS
# =============================================================================


class TestCrossValidationConfigValidated:
    """Tests for CrossValidationConfigValidated Pydantic model."""

    @pytest.fixture
    def valid_cv_config(self) -> Dict[str, Any]:
        """Valid cross-validation configuration."""
        return {
            'start_cutoff': 40,
            'end_cutoff': 100,
            'validation_method': 'expanding_window',
            'n_splits': 5
        }

    def test_valid_config_creates_instance(self, valid_cv_config):
        """Valid configuration should create CrossValidationConfigValidated instance."""
        config = CrossValidationConfigValidated(**valid_cv_config)

        assert config.start_cutoff == 40
        assert config.end_cutoff == 100
        assert config.validation_method == 'expanding_window'
        assert config.n_splits == 5

    def test_default_end_cutoff_none(self):
        """end_cutoff should default to None."""
        config = CrossValidationConfigValidated(
            start_cutoff=40,
            validation_method='expanding_window',
            n_splits=5
        )
        assert config.end_cutoff is None

    def test_default_validation_method(self):
        """validation_method should default to 'expanding_window'."""
        config = CrossValidationConfigValidated(
            start_cutoff=40,
            n_splits=5
        )
        assert config.validation_method == 'expanding_window'

    def test_rolling_window_method_valid(self, valid_cv_config):
        """validation_method='rolling_window' should be valid."""
        valid_cv_config['validation_method'] = 'rolling_window'
        config = CrossValidationConfigValidated(**valid_cv_config)
        assert config.validation_method == 'rolling_window'

    def test_start_cutoff_too_small_fails(self, valid_cv_config):
        """start_cutoff <= 5 should fail (gt=5 constraint)."""
        valid_cv_config['start_cutoff'] = 5
        with pytest.raises(ValidationError):
            CrossValidationConfigValidated(**valid_cv_config)

    def test_start_cutoff_exceeds_max_fails(self, valid_cv_config):
        """start_cutoff > 200 should fail (le=200 constraint)."""
        valid_cv_config['start_cutoff'] = 201
        with pytest.raises(ValidationError):
            CrossValidationConfigValidated(**valid_cv_config)

    def test_end_cutoff_less_than_start_fails(self, valid_cv_config):
        """end_cutoff <= start_cutoff should fail custom validator."""
        valid_cv_config['start_cutoff'] = 50
        valid_cv_config['end_cutoff'] = 40  # Less than start
        with pytest.raises(ValidationError) as exc_info:
            CrossValidationConfigValidated(**valid_cv_config)
        assert 'end_cutoff must be greater than start_cutoff' in str(exc_info.value)

    def test_end_cutoff_equal_to_start_fails(self, valid_cv_config):
        """end_cutoff == start_cutoff should fail custom validator."""
        valid_cv_config['start_cutoff'] = 50
        valid_cv_config['end_cutoff'] = 50  # Equal to start
        with pytest.raises(ValidationError) as exc_info:
            CrossValidationConfigValidated(**valid_cv_config)
        assert 'end_cutoff must be greater than start_cutoff' in str(exc_info.value)

    def test_n_splits_negative_fails(self, valid_cv_config):
        """n_splits < 0 should fail (ge=0 constraint)."""
        valid_cv_config['n_splits'] = -1
        with pytest.raises(ValidationError):
            CrossValidationConfigValidated(**valid_cv_config)

    def test_n_splits_exceeds_max_fails(self, valid_cv_config):
        """n_splits > 50 should fail (le=50 constraint)."""
        valid_cv_config['n_splits'] = 51
        with pytest.raises(ValidationError):
            CrossValidationConfigValidated(**valid_cv_config)

    def test_invalid_validation_method_fails(self, valid_cv_config):
        """Invalid validation_method should fail (Literal constraint)."""
        valid_cv_config['validation_method'] = 'k_fold'
        with pytest.raises(ValidationError):
            CrossValidationConfigValidated(**valid_cv_config)

    def test_extra_fields_forbidden(self, valid_cv_config):
        """Extra fields should raise ValidationError."""
        valid_cv_config['extra_param'] = True
        with pytest.raises(ValidationError):
            CrossValidationConfigValidated(**valid_cv_config)


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTION TESTS
# =============================================================================


class TestBackwardCompatibilityFunctions:
    """Tests for backward compatibility functions."""

    @pytest.fixture
    def valid_config_kwargs(self) -> Dict[str, Any]:
        """Valid kwargs for config building functions."""
        return {
            'n_bootstrap_samples': 100,
            'ridge_alpha': 1.0,
            'random_state': 42,
            'exclude_holidays': True,
            'mature_data_cutoff_days': 50,
            'min_training_cutoff': 30
        }

    def test_build_forecasting_config_validated_returns_model(self, valid_config_kwargs):
        """build_forecasting_config_validated should return Pydantic model."""
        result = build_forecasting_config_validated(**valid_config_kwargs)

        assert isinstance(result, ForecastingConfigValidated)
        assert result.n_bootstrap_samples == 100

    def test_build_forecasting_config_validated_raises_on_invalid(self):
        """build_forecasting_config_validated should raise ValueError on invalid config."""
        with pytest.raises(ValueError) as exc_info:
            build_forecasting_config_validated(
                n_bootstrap_samples=-10,  # Invalid
                ridge_alpha=1.0,
                random_state=42,
                mature_data_cutoff_days=50,
                min_training_cutoff=30
            )
        assert 'validation failed' in str(exc_info.value).lower()

    def test_build_forecasting_config_returns_dict(self, valid_config_kwargs):
        """build_forecasting_config (legacy) should return dictionary."""
        result = build_forecasting_config(**valid_config_kwargs)

        assert isinstance(result, dict)
        assert result['n_bootstrap_samples'] == 100
        assert result['ridge_alpha'] == 1.0

    def test_build_forecasting_config_raises_on_invalid(self):
        """build_forecasting_config (legacy) should raise ValueError on invalid config."""
        with pytest.raises(ValueError) as exc_info:
            build_forecasting_config(
                n_bootstrap_samples=-10,  # Invalid
                ridge_alpha=1.0,
                random_state=42,
                mature_data_cutoff_days=50,
                min_training_cutoff=30
            )
        assert 'validation failed' in str(exc_info.value).lower()

    def test_build_forecasting_config_with_defaults(self):
        """build_forecasting_config should apply defaults correctly."""
        result = build_forecasting_config(
            n_bootstrap_samples=100,
            ridge_alpha=1.0,
            random_state=42,
            mature_data_cutoff_days=50,
            min_training_cutoff=30
            # exclude_holidays not provided, should default to True
        )

        assert result['exclude_holidays'] is True


# =============================================================================
# EDGE CASES AND INTEGRATION TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_forecasting_config_boundary_valid_bootstrap(self):
        """Boundary value: n_bootstrap_samples = 10 should pass."""
        config = ForecastingConfigValidated(
            n_bootstrap_samples=10,  # Minimum valid
            ridge_alpha=1.0,
            random_state=42,
            mature_data_cutoff_days=50,
            min_training_cutoff=30
        )
        assert config.n_bootstrap_samples == 10

    def test_forecasting_config_boundary_max_bootstrap(self):
        """Boundary value: n_bootstrap_samples = 1000 should pass."""
        config = ForecastingConfigValidated(
            n_bootstrap_samples=1000,  # Maximum valid
            ridge_alpha=1.0,
            random_state=42,
            mature_data_cutoff_days=50,
            min_training_cutoff=30
        )
        assert config.n_bootstrap_samples == 1000

    def test_cv_config_end_cutoff_just_above_start(self):
        """end_cutoff = start_cutoff + 1 should be valid."""
        config = CrossValidationConfigValidated(
            start_cutoff=40,
            end_cutoff=41,  # Just above start
            n_splits=5
        )
        assert config.end_cutoff == 41

    def test_bootstrap_config_small_alpha(self):
        """Very small alpha (0.001) should be valid."""
        config = BootstrapModelConfigValidated(
            alpha=0.001,
            estimator_type='Ridge'
        )
        assert config.alpha == 0.001

    def test_forecasting_config_zero_random_state(self):
        """random_state = 0 should be valid (ge=0)."""
        config = ForecastingConfigValidated(
            n_bootstrap_samples=100,
            ridge_alpha=1.0,
            random_state=0,  # Boundary: minimum
            mature_data_cutoff_days=50,
            min_training_cutoff=30
        )
        assert config.random_state == 0

    def test_cv_config_zero_splits(self):
        """n_splits = 0 should be valid (ge=0 constraint)."""
        config = CrossValidationConfigValidated(
            start_cutoff=40,
            n_splits=0
        )
        assert config.n_splits == 0
