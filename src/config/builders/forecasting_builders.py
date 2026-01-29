"""
Forecasting Stage Configuration Builders

Extracted from config_builder.py for single-responsibility compliance.
Handles all forecasting stage configurations: core forecasting, cross-validation,
bootstrap models, benchmarks, and validation frameworks.

Usage:
    from src.config.forecasting_builders import build_forecasting_stage_config
    config = build_forecasting_stage_config(bootstrap_samples=1000)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from src.config.types.product_config import get_default_feature_config
from src.config.types.forecasting_config import (
    ForecastingConfig, CrossValidationConfig, BootstrapModelConfig,
    BenchmarkModelConfig, ValidationFrameworkConfig, ForecastingStageConfig
)


# Module exports (including private functions for testing)
__all__ = [
    # Public API
    'build_forecasting_stage_config',
    # Private helpers (for unit testing)
    '_build_forecasting_core_config',
    '_build_cv_config',
    '_build_bootstrap_model_config',
    '_build_validation_framework_config',
    '_get_competitive_features',
    '_build_sign_correction_configs',
    '_build_model_configs',
    '_build_validation_configs',
    '_build_feature_and_sign_configs',
]


# =============================================================================
# FORECASTING CONFIGURATION HELPERS
# =============================================================================


def _build_forecasting_core_config(
    bootstrap_samples: int,
    ridge_alpha: float,
    random_state: int,
    min_training_cutoff: int
) -> ForecastingConfig:
    """Build core forecasting configuration.

    Parameters
    ----------
    bootstrap_samples : int
        Number of bootstrap samples for uncertainty quantification
    ridge_alpha : float
        Ridge regression regularization parameter
    random_state : int
        Random seed for reproducibility
    min_training_cutoff : int
        Minimum observations required before forecasting

    Returns
    -------
    ForecastingConfig
        Core forecasting configuration
    """
    mature_data_cutoff_days = 50  # Business rule: exclude recent incomplete data
    return ForecastingConfig({
        'n_bootstrap_samples': bootstrap_samples,
        'ridge_alpha': ridge_alpha,
        'random_state': random_state,
        'exclude_holidays': True,
        'mature_data_cutoff_days': mature_data_cutoff_days,
        'min_training_cutoff': min_training_cutoff
    })


def _build_cv_config(
    start_cutoff: int,
    end_cutoff: Optional[int],
    validation_method: str
) -> CrossValidationConfig:
    """Build cross-validation configuration.

    Parameters
    ----------
    start_cutoff : int
        First observation index for cross-validation
    end_cutoff : Optional[int]
        Last observation index (None = use full dataset)
    validation_method : str
        Cross-validation method (e.g., 'expanding_window')

    Returns
    -------
    CrossValidationConfig
        Cross-validation configuration
    """
    return CrossValidationConfig({
        'start_cutoff': start_cutoff,
        'end_cutoff': end_cutoff,
        'validation_method': validation_method,
        'n_splits': 0  # Expanding window (continuous)
    })


def _build_bootstrap_model_config(
    ridge_alpha: float,
    bootstrap_samples: int
) -> BootstrapModelConfig:
    """Build Bootstrap Ridge regression model configuration.

    Parameters
    ----------
    ridge_alpha : float
        Ridge regression regularization parameter
    bootstrap_samples : int
        Number of bootstrap estimators

    Returns
    -------
    BootstrapModelConfig
        Bootstrap model configuration
    """
    return BootstrapModelConfig({
        'estimator_type': 'Ridge',
        'alpha': ridge_alpha,
        'positive_constraint': True,  # Business constraint: sales >= 0
        'fit_intercept': True,
        'n_estimators': bootstrap_samples,
        'normalize': False  # Features already preprocessed
    })


def _build_validation_framework_config(
    enable_detailed_validation: bool
) -> ValidationFrameworkConfig:
    """Build mathematical validation framework configuration.

    Parameters
    ----------
    enable_detailed_validation : bool
        Whether to enable detailed comparison output

    Returns
    -------
    ValidationFrameworkConfig
        Validation framework configuration
    """
    return ValidationFrameworkConfig({
        'tolerance_r2': 1e-6,
        'tolerance_mape': 1e-4,
        'tolerance_prediction': 1e-6,
        'enable_detailed_comparison': enable_detailed_validation,
        'reference_results_path': str(Path(__file__).parent.parent.parent / 'outputs' / 'results')
    })


def _get_competitive_features() -> List[str]:
    """Return list of competitive features for sign correction.

    Derives from ProductFeatureConfig (Single Source of Truth), filtering
    to competitor-only features.

    Returns
    -------
    List[str]
        List of competitor feature names
    """
    config = get_default_feature_config()
    return [f for f in config.candidate_features if 'competitor' in f.lower()]


def _build_sign_correction_configs(
    model_features: List[str],
    benchmark_features: List[str],
    competitive_features: List[str],
    decay_rate: float
) -> tuple:
    """Build sign correction configurations for model and benchmark.

    Creates boolean masks indicating which features need sign correction
    (competitor features should have negative coefficients for economic
    theory compliance).

    Parameters
    ----------
    model_features : List[str]
        Feature names used in the main model
    benchmark_features : List[str]
        Feature names used in the benchmark model
    competitive_features : List[str]
        List of competitive feature names requiring sign correction
    decay_rate : float
        Temporal weight decay rate

    Returns
    -------
    tuple
        (model_sign_correction_config, benchmark_sign_correction_config)
    """
    model_correction_mask = np.array([f in competitive_features for f in model_features])
    benchmark_correction_mask = np.array([False] * len(benchmark_features))

    model_sign_correction_config = {
        'sign_correction_mask': model_correction_mask,
        'decay_rate': decay_rate
    }

    benchmark_sign_correction_config = {
        'sign_correction_mask': benchmark_correction_mask,
        'decay_rate': decay_rate
    }

    return model_sign_correction_config, benchmark_sign_correction_config


# =============================================================================
# STAGE-BASED FORECASTING HELPERS
# =============================================================================


def _build_model_configs(
    bootstrap_samples: int,
    ridge_alpha: float,
    random_state: int,
    min_training_cutoff: int
) -> Dict[str, Any]:
    """Build model-related configs: forecasting, bootstrap, benchmark.

    Parameters
    ----------
    bootstrap_samples : int
        Number of bootstrap samples
    ridge_alpha : float
        Ridge regularization parameter
    random_state : int
        Random seed for reproducibility
    min_training_cutoff : int
        Minimum observations required

    Returns
    -------
    Dict[str, Any]
        Model configuration dict with forecasting, bootstrap, benchmark configs
    """
    return {
        'forecasting_config': _build_forecasting_core_config(
            bootstrap_samples, ridge_alpha, random_state, min_training_cutoff
        ),
        'bootstrap_model_config': _build_bootstrap_model_config(ridge_alpha, bootstrap_samples),
        'benchmark_model_config': BenchmarkModelConfig({
            'method': 'rolling_average',
            'window_size': None,
            'seasonal_period': None
        })
    }


def _build_validation_configs(
    start_cutoff: int,
    end_cutoff: Optional[int],
    validation_method: str,
    enable_detailed_validation: bool
) -> Dict[str, Any]:
    """Build validation-related configs: cv, validation framework.

    Parameters
    ----------
    start_cutoff : int
        First observation index for CV
    end_cutoff : Optional[int]
        Last observation index (None = full dataset)
    validation_method : str
        Cross-validation method
    enable_detailed_validation : bool
        Enable detailed comparison output

    Returns
    -------
    Dict[str, Any]
        Validation configuration dict
    """
    return {
        'cv_config': _build_cv_config(start_cutoff, end_cutoff, validation_method),
        'validation_config': _build_validation_framework_config(enable_detailed_validation)
    }


def _build_feature_and_sign_configs() -> Dict[str, Any]:
    """Build feature specifications and sign correction configs.

    Returns
    -------
    Dict[str, Any]
        Feature and sign correction configuration dict
    """
    business_filter_config = {
        'analysis_start_date': '2022-04-01',
        'temporal_weight_decay_rate': 0.98,
        'remove_incomplete_final_obs': True
    }

    competitive_features = _get_competitive_features()
    model_features = ['prudential_rate_current', 'competitor_mid_t2', 'competitor_top5_t3']
    benchmark_features = ['sales_target_contract_t5']

    model_sign_config, benchmark_sign_config = _build_sign_correction_configs(
        model_features, benchmark_features, competitive_features,
        business_filter_config['temporal_weight_decay_rate']
    )

    return {
        'business_filter_config': business_filter_config,
        'competitive_features': competitive_features,
        'performance_monitoring_config': {'progress_reporting_interval': 25},
        'model_features': model_features,
        'benchmark_features': benchmark_features,
        'target_variable': 'sales_target_current',
        'model_sign_correction_config': model_sign_config,
        'benchmark_sign_correction_config': benchmark_sign_config
    }


# =============================================================================
# MAIN FORECASTING STAGE CONFIG BUILDER
# =============================================================================


def build_forecasting_stage_config(
    version: int = 6,
    bootstrap_samples: int = 1000,
    ridge_alpha: float = 1.0,
    random_state: int = 42,
    min_training_cutoff: int = 30,
    start_cutoff: int = 30,
    end_cutoff: Optional[int] = None,
    validation_method: str = 'expanding_window',
    enable_detailed_validation: bool = True
) -> ForecastingStageConfig:
    """Build comprehensive forecasting stage configuration.

    Consolidates ALL forecasting parameters into single source of truth.

    Returns
    -------
    ForecastingStageConfig
        Complete forecasting stage configuration
    """
    config_dict: Dict[str, Any] = {}

    # Model configs (forecasting, bootstrap, benchmark)
    config_dict.update(_build_model_configs(
        bootstrap_samples, ridge_alpha, random_state, min_training_cutoff
    ))

    # Validation configs (cv, validation framework)
    config_dict.update(_build_validation_configs(
        start_cutoff, end_cutoff, validation_method, enable_detailed_validation
    ))

    # Feature and sign correction configs
    config_dict.update(_build_feature_and_sign_configs())

    return ForecastingStageConfig(config_dict)
