"""
Visualization Test Data Fixtures for pytest-mpl Image Regression Testing.

This module provides test data fixtures for visualization baseline testing.
Uses synthetic data to ensure reproducible, deterministic plot outputs.

Design Principles:
- Deterministic data generation (fixed random seeds)
- Minimal data sufficient to exercise all plot elements
- Realistic structure matching actual module inputs
- Reusable across all visualization test modules

Usage:
    from tests.fixtures.visualization_data import (
        get_aic_results_fixture,
        get_bootstrap_results_fixture,
        get_analysis_results_fixture,
    )
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd


# =============================================================================
# SEED MANAGEMENT (Deterministic Output)
# =============================================================================

_DEFAULT_SEED = 42


def _reset_seed(seed: int = _DEFAULT_SEED) -> None:
    """Reset random seed for reproducible data generation."""
    np.random.seed(seed)


# =============================================================================
# CORE DATA FIXTURES
# =============================================================================


def get_aic_results_fixture(
    n_models: int = 20,
    seed: int = _DEFAULT_SEED
) -> pd.DataFrame:
    """
    Generate AIC results DataFrame for model comparison visualizations.

    Simulates the output of AIC-based feature selection with realistic
    model metrics and feature combinations.

    Args:
        n_models: Number of models to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: features, aic, r_squared, n_features,
                               economically_valid, coefficients
    """
    _reset_seed(seed)

    # Feature pool for combinations
    feature_pool = [
        'prudential_rate_current', 'competitor_mid_t1', 'competitor_mid_t2',
        'competitor_top5_t3', 'DGS5', 'VIX', 'Spread', 'treasury_10y',
        'market_volatility', 'economic_indicator'
    ]

    models = []
    for i in range(n_models):
        # Generate feature combination
        n_feat = np.random.randint(1, 5)
        selected = np.random.choice(feature_pool, size=n_feat, replace=False)
        feature_str = ' + '.join(sorted(selected))

        # Generate realistic metrics (lower AIC = better, correlated with R2)
        base_aic = 250 - n_feat * 3 + np.random.normal(0, 10)
        r_squared = min(0.85, max(0.15, 0.4 + n_feat * 0.08 + np.random.normal(0, 0.1)))

        # Generate coefficients
        coefficients = {'const': np.random.normal(15, 2)}
        for feat in selected:
            if 'competitor' in feat:
                coefficients[feat] = np.random.uniform(-0.5, 0.1)  # Usually negative
            elif 'prudential' in feat:
                coefficients[feat] = np.random.uniform(0.1, 0.8)   # Usually positive
            else:
                coefficients[feat] = np.random.normal(0, 0.3)

        models.append({
            'features': feature_str,
            'aic': base_aic,
            'r_squared': r_squared,
            'n_features': n_feat,
            'economically_valid': np.random.random() > 0.3,
            'coefficients': coefficients
        })

    df = pd.DataFrame(models)
    if df.empty:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'features', 'aic', 'r_squared', 'n_features',
            'economically_valid', 'coefficients'
        ])
    return df.sort_values('aic').reset_index(drop=True)


def get_bootstrap_results_fixture(
    n_models: int = 10,
    n_bootstrap_samples: int = 50,
    seed: int = _DEFAULT_SEED
) -> List[Dict[str, Any]]:
    """
    Generate bootstrap stability results for visualization.

    Creates both dictionary-format and NamedTuple-format results
    to test handling of different input types.

    Args:
        n_models: Number of models to include
        n_bootstrap_samples: Number of bootstrap samples per model
        seed: Random seed for reproducibility

    Returns:
        List of bootstrap result dictionaries
    """
    _reset_seed(seed)

    # Define NamedTuple for structured results
    BootstrapResult = namedtuple(
        'BootstrapResult',
        ['model_features', 'original_aic', 'bootstrap_samples',
         'stability_metrics', 'confidence_intervals']
    )

    feature_pool = [
        'prudential_rate_current', 'competitor_mid_t1', 'competitor_mid_t2',
        'competitor_top5_t3', 'DGS5', 'VIX'
    ]

    results = []
    for i in range(n_models):
        # Generate feature combination
        n_feat = np.random.randint(1, 4)
        selected = np.random.choice(feature_pool, size=n_feat, replace=False)
        feature_str = ' + '.join(sorted(selected))

        # Original AIC
        original_aic = 240 - n_feat * 5 + np.random.normal(0, 8)

        # Generate bootstrap samples
        aic_samples = np.random.normal(original_aic, 5, n_bootstrap_samples)
        bootstrap_samples = [{'aic': aic, 'r_squared': 0.5 + np.random.normal(0, 0.1)}
                            for aic in aic_samples]

        # Stability metrics
        aic_mean = np.mean(aic_samples)
        aic_std = np.std(aic_samples)
        stability_metrics = {
            'aic_mean': aic_mean,
            'aic_std': aic_std,
            'aic_cv': aic_std / abs(aic_mean) if aic_mean != 0 else 0,
            'successful_fit_rate': 0.85 + np.random.uniform(0, 0.15)
        }

        # Confidence intervals
        ci_90 = (np.percentile(aic_samples, 5), np.percentile(aic_samples, 95))
        ci_95 = (np.percentile(aic_samples, 2.5), np.percentile(aic_samples, 97.5))
        ci_99 = (np.percentile(aic_samples, 0.5), np.percentile(aic_samples, 99.5))

        # Alternate between dict and NamedTuple formats
        if i % 2 == 0:
            results.append(BootstrapResult(
                model_features=feature_str,
                original_aic=original_aic,
                bootstrap_samples=bootstrap_samples,
                stability_metrics=stability_metrics,
                confidence_intervals={90: ci_90, 95: ci_95, 99: ci_99}
            ))
        else:
            results.append({
                'features': feature_str,
                'aic_stability_cv': stability_metrics['aic_cv'],
                'median_aic': np.median(aic_samples),
                'successful_fits': int(stability_metrics['successful_fit_rate'] * 100)
            })

    return results


def get_information_criteria_results_fixture(
    n_models: int = 10,
    seed: int = _DEFAULT_SEED
) -> List[Any]:
    """
    Generate information criteria results for robustness visualization.

    Args:
        n_models: Number of models to include
        seed: Random seed for reproducibility

    Returns:
        List of information criteria result NamedTuples
    """
    _reset_seed(seed)

    ICResult = namedtuple(
        'ICResult',
        ['model_features', 'criteria_values', 'ranking_positions', 'robustness_score']
    )

    feature_pool = [
        'prudential_rate_current', 'competitor_mid_t1', 'competitor_mid_t2',
        'DGS5', 'VIX', 'Spread'
    ]

    results = []
    for i in range(n_models):
        n_feat = np.random.randint(1, 4)
        selected = np.random.choice(feature_pool, size=n_feat, replace=False)
        feature_str = ' + '.join(sorted(selected))

        # Base score influenced by feature count
        base = 250 - n_feat * 5 + np.random.normal(0, 10)

        criteria_values = {
            'aic': base,
            'bic': base + n_feat * 2,  # BIC penalizes complexity more
            'hqic': base + n_feat * 1.5,
            'caic': base + n_feat * 2.5
        }

        # Rankings (will be computed relative to other models)
        ranking_positions = {
            'aic': i + 1,
            'bic': i + 1,
            'hqic': i + 1,
            'caic': i + 1
        }

        robustness_score = 0.5 + np.random.uniform(0, 0.5)

        results.append(ICResult(
            model_features=feature_str,
            criteria_values=criteria_values,
            ranking_positions=ranking_positions,
            robustness_score=robustness_score
        ))

    return results


def get_coefficient_stability_fixture(
    n_models: int = 8,
    seed: int = _DEFAULT_SEED
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Generate coefficient stability analysis data.

    Args:
        n_models: Number of models to include
        seed: Random seed for reproducibility

    Returns:
        Nested dictionary: model_features -> feature_name -> stability_stats
    """
    _reset_seed(seed)

    feature_pool = [
        'prudential_rate_current', 'competitor_mid_t1', 'competitor_mid_t2',
        'DGS5', 'VIX'
    ]

    stability = {}
    for i in range(n_models):
        n_feat = np.random.randint(2, 4)
        selected = np.random.choice(feature_pool, size=n_feat, replace=False)
        model_key = ' + '.join(sorted(selected))

        model_stability = {}
        for feat in selected:
            mean_coef = np.random.normal(0, 0.5)
            std_coef = abs(np.random.normal(0, 0.1))

            model_stability[feat] = {
                'mean': mean_coef,
                'std': std_coef,
                'cv': std_coef / abs(mean_coef) if mean_coef != 0 else 0,
                'sign_consistency': 0.7 + np.random.uniform(0, 0.3)
            }

        stability[model_key] = model_stability

    return stability


# =============================================================================
# COMPOSITE FIXTURES (Analysis Results)
# =============================================================================


def get_analysis_results_fixture(seed: int = _DEFAULT_SEED) -> Dict[str, Any]:
    """
    Generate complete analysis results for dashboard visualizations.

    Combines all component fixtures into a single dictionary matching
    the structure expected by visualization modules.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dictionary with all analysis result components
    """
    aic_results = get_aic_results_fixture(n_models=20, seed=seed)
    bootstrap_results = get_bootstrap_results_fixture(n_models=10, seed=seed)
    ic_results = get_information_criteria_results_fixture(n_models=10, seed=seed)

    # Build final model selection
    best_model = aic_results.iloc[0]
    final_model = {
        'selected_model': {
            'features': best_model['features'],
            'aic': best_model['aic'],
            'r_squared': best_model['r_squared'],
            'n_features': best_model['n_features']
        },
        'selection_criteria': 'AIC with economic constraints',
        'validation_status': 'PASSED'
    }

    # Build metadata
    metadata = {
        'analysis_id': f'test_analysis_{seed}',
        'dataset_info': {
            'total_observations': 167,
            'date_range': '2020-01-01 to 2023-06-30'
        },
        'bootstrap_config': {
            'n_samples': 100,
            'block_size': 8
        },
        'information_criteria_config': {
            'criteria': ['aic', 'bic', 'hqic', 'caic']
        },
        'precision_tolerance': 1e-12
    }

    return {
        'aic_results': aic_results,
        'final_model': final_model,
        'metadata': metadata,
        'bootstrap_results': {
            'block_bootstrap_results': bootstrap_results,
            'coefficient_stability_analysis': get_coefficient_stability_fixture(seed=seed)
        },
        'enhanced_metrics': {
            'information_criteria_analysis': ic_results
        }
    }


def get_economically_valid_models_fixture(
    aic_results: pd.DataFrame,
    seed: int = _DEFAULT_SEED
) -> pd.DataFrame:
    """
    Filter AIC results to economically valid models only.

    Args:
        aic_results: Full AIC results DataFrame
        seed: Random seed (for consistency)

    Returns:
        DataFrame filtered to economically_valid == True
    """
    if 'economically_valid' in aic_results.columns:
        return aic_results[aic_results['economically_valid'] == True].copy()
    return aic_results.copy()


def get_final_model_fixture(
    aic_results: pd.DataFrame
) -> Dict[str, Any]:
    """
    Extract final model selection from AIC results.

    Args:
        aic_results: AIC results DataFrame (sorted by AIC)

    Returns:
        Dictionary with selected model details
    """
    if aic_results.empty:
        return {}

    best = aic_results.iloc[0]
    return {
        'aic': best['aic'],
        'r_squared': best['r_squared'],
        'features': best['features'],
        'n_features': best['n_features']
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'get_aic_results_fixture',
    'get_bootstrap_results_fixture',
    'get_information_criteria_results_fixture',
    'get_coefficient_stability_fixture',
    'get_analysis_results_fixture',
    'get_economically_valid_models_fixture',
    'get_final_model_fixture',
]
