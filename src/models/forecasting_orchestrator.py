"""
Forecasting Pipeline Orchestrator - Time Series Bootstrap Ridge Regression.

This module provides orchestration functions that coordinate bootstrap forecasting
operations following the canonical architecture pattern established by feature
selection orchestration.

Key Functions:
- run_benchmark_forecasting: Orchestrate benchmark forecasting workflow
- run_bootstrap_ridge_forecasting: Orchestrate model forecasting workflow
- run_forecasting_pipeline: Main coordinating wrapper for complete analysis

Design Principles:
- Single responsibility: Workflow orchestration only
- Immutable composition: Combines atomic functions without side effects
- Comprehensive error handling and progress reporting
- Zero regression from existing notebook implementation

Architectural Pattern:
    notebooks → orchestrator → atomic_ops + atomic_models → results

Mathematical Equivalence Target:
- Model R²: 0.782598, MAPE: 12.74%
- Benchmark R²: 0.575437, MAPE: 16.40%
- 129 forecasts with 100 bootstrap samples each
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.utils import resample

# Atomic operations imports
from src.data.forecasting_atomic_ops import (
    prepare_cutoff_data_complete,
    extract_test_target_contract_date_atomic
)
from src.models.forecasting_atomic_models import (
    fit_bootstrap_ensemble_atomic,
    predict_bootstrap_ensemble_atomic,
    calculate_prediction_error_atomic
)
from src.models.forecasting_atomic_results import (
    calculate_performance_metrics_atomic
)
from src.models.forecasting_atomic_validation import (
    validate_bootstrap_predictions_atomic
)


def _validate_features_not_empty(features: List[str]) -> None:
    """
    Validate that the feature list is not empty.

    Parameters
    ----------
    features : List[str]
        Feature columns to check

    Raises
    ------
    ValueError
        If feature list is empty
    """
    if not features or len(features) == 0:
        raise ValueError(
            f"Feature list is empty. "
            f"Business Impact: Cannot proceed with forecasting without features. "
            f"Action Required: Provide at least one feature column."
        )


def _validate_features_exist_in_dataframe(
    df: pd.DataFrame, features: List[str]
) -> None:
    """
    Validate that all required features exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate against
    features : List[str]
        Feature columns that must exist

    Raises
    ------
    KeyError
        If any features are missing from the dataframe
    """
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise KeyError(
            f"Missing required features in dataset: {missing_features}. "
            f"Available columns: {list(df.columns)[:10]}... "
            f"Business Impact: Cannot proceed with forecasting without required features. "
            f"Action Required: Verify feature names and dataset completeness."
        )


def _validate_target_variable(df: pd.DataFrame, target_variable: str) -> None:
    """
    Validate that target variable exists in the dataframe.

    Delegates to canonical validator in src.validation.input_validators.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate against
    target_variable : str
        Target variable name to check

    Raises
    ------
    KeyError
        If target variable is missing from the dataframe
    """
    if not target_variable:
        return  # Empty target is allowed (original behavior)

    from src.validation.input_validators import validate_target_variable
    validate_target_variable(
        target_variable,
        df=df,
        mode="raise",
        require_numeric=False,
        exception_type=KeyError,
    )


def _validate_cutoff_bounds(
    n_obs: int, start_cutoff: int, end_cutoff: int
) -> None:
    """
    Validate cutoff bounds against dataset size.

    Parameters
    ----------
    n_obs : int
        Number of observations in the dataset
    start_cutoff : int
        Starting cutoff index
    end_cutoff : int
        Ending cutoff index

    Raises
    ------
    ValueError
        If end_cutoff exceeds dataset size or insufficient forecasts for R2
    """
    if end_cutoff > n_obs:
        raise ValueError(
            f"end_cutoff ({end_cutoff}) exceeds dataset size ({n_obs}). "
            f"Business Impact: Cannot forecast beyond available data. "
            f"Action Required: Reduce end_cutoff to at most {n_obs}."
        )

    # Note: start_cutoff >= end_cutoff is valid - just means zero forecasts

    n_forecasts = max(0, end_cutoff - start_cutoff)
    if n_forecasts == 1:
        raise ValueError(
            f"R² calculation requires at least 2 forecasts, got {n_forecasts}. "
            f"Business Impact: Cannot calculate meaningful performance metrics with single forecast. "
            f"Action Required: Use end_cutoff - start_cutoff >= 2."
        )


def validate_forecasting_inputs(
    df: pd.DataFrame,
    start_cutoff: int,
    end_cutoff: int,
    features: List[str],
    target_variable: str = None
) -> None:
    """
    Validate inputs for forecasting operations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate
    start_cutoff : int
        Starting cutoff index
    end_cutoff : int
        Ending cutoff index
    features : List[str]
        Feature columns to check
    target_variable : str, optional
        Target variable to check

    Raises
    ------
    ValueError
        If validation fails with clear business context
    KeyError
        If required columns are missing
    """
    _validate_features_not_empty(features)
    _validate_features_exist_in_dataframe(df, features)
    _validate_target_variable(df, target_variable)
    _validate_cutoff_bounds(len(df), start_cutoff, end_cutoff)


def _process_single_benchmark_cutoff(
    df: pd.DataFrame,
    cutoff: int,
    benchmark_features: List[str],
    n_bootstrap_samples: int,
    results: Dict[str, Any]
) -> bool:
    """Process single cutoff for benchmark forecasting. Returns True on success."""
    y_true_test = extract_test_target_contract_date_atomic(df, cutoff)
    forecast_date = str(df.iloc[cutoff]['date'])[:10]

    df_cutoff = df[:cutoff]
    if len(df_cutoff) == 0:
        results['errors'].append(f"Cutoff {cutoff}: No data available")
        return False

    sample_data = df_cutoff[benchmark_features].iloc[-1]

    bootstrap_predictions = []
    for i in range(n_bootstrap_samples):
        resampled_data = resample(sample_data, replace=True, random_state=i)
        bootstrap_predictions.append(resampled_data.mean())
    bootstrap_predictions = np.array(bootstrap_predictions)

    benchmark_prediction = bootstrap_predictions.mean()
    error_metrics = calculate_prediction_error_atomic(y_true_test, benchmark_prediction)

    results['dates'].append(forecast_date)
    results['y_true'].append(y_true_test)
    results['y_predict'].append(benchmark_prediction)
    results['abs_pct_error'].append(error_metrics['percentage_error'] / 100)
    results['bootstrap_predictions'][forecast_date] = bootstrap_predictions
    results['cutoffs'].append(cutoff)
    return True


def _initialize_forecasting_results() -> Dict[str, Any]:
    """
    Initialize empty results dictionary for forecasting operations.

    Returns
    -------
    Dict[str, Any]
        Empty results structure with all required keys
    """
    return {
        'dates': [],
        'y_true': [],
        'y_predict': [],
        'abs_pct_error': [],
        'bootstrap_predictions': {},
        'cutoffs': [],
        'errors': []
    }


def _report_progress(
    cutoff_idx: int,
    n_forecasts: int,
    abs_pct_errors: List[float],
    progress_interval: int
) -> None:
    """
    Report progress at specified intervals during forecasting.

    Parameters
    ----------
    cutoff_idx : int
        Current cutoff index (0-based)
    n_forecasts : int
        Total number of forecasts
    abs_pct_errors : List[float]
        List of absolute percentage errors so far
    progress_interval : int
        Report every N forecasts
    """
    if (cutoff_idx + 1) % progress_interval == 0:
        progress = (cutoff_idx + 1) / n_forecasts * 100
        current_mape = np.mean(abs_pct_errors) * 100
        print(f"   Progress: {progress:.1f}% ({cutoff_idx + 1}/{n_forecasts} forecasts, MAPE: {current_mape:.2f}%)")


def _finalize_forecasting_results(results: Dict[str, Any], success_count: int) -> None:
    """Calculate final performance metrics for forecasting results."""
    if success_count > 0:
        metrics = calculate_performance_metrics_atomic(
            y_true=np.array(results['y_true']),
            y_pred=np.array(results['y_predict']),
            sample_weights=None
        )
        results['metrics'] = metrics
        results['n_forecasts'] = success_count
    else:
        results['metrics'] = {'r2_score': 0.0, 'mape': 100.0}
        results['n_forecasts'] = 0


def run_benchmark_forecasting(
    df: pd.DataFrame,
    start_cutoff: int,
    end_cutoff: int,
    benchmark_features: List[str],
    n_bootstrap_samples: int,
    random_state: int = 42,
    progress_interval: int = 25
) -> Dict[str, Any]:
    """
    Orchestrate benchmark forecasting using bootstrap resampling.

    This function coordinates the benchmark forecasting workflow by extracting
    the last feature value at each cutoff and generating bootstrap predictions
    through resampling. Follows the exact algorithm from original notebook cell 17.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset with temporal structure
    start_cutoff : int
        First observation index for forecasting
    end_cutoff : int
        Last observation index for forecasting (exclusive)
    benchmark_features : List[str]
        Feature columns to use for benchmark (typically ['sales_target_contract_t5'])
    n_bootstrap_samples : int
        Number of bootstrap samples per forecast (typically 100)
    random_state : int, default=42
        Random seed for reproducibility
    progress_interval : int, default=25
        Report progress every N forecasts

    Returns
    -------
    Dict[str, Any]
        Benchmark forecasting results with structure matching original notebook.

    Business Context
    ----------------
    Benchmark provides baseline performance threshold for model comparison.
    Uses simple persistence with bootstrap uncertainty quantification.
    Expected performance: R² ≈ 0.575, MAPE ≈ 16.4%
    """
    validate_forecasting_inputs(df=df, start_cutoff=start_cutoff,
                                end_cutoff=end_cutoff, features=benchmark_features)

    results = _initialize_forecasting_results()
    n_forecasts = end_cutoff - start_cutoff
    success_count = 0

    for cutoff_idx, cutoff in enumerate(range(start_cutoff, end_cutoff)):
        try:
            if _process_single_benchmark_cutoff(df, cutoff, benchmark_features,
                                                n_bootstrap_samples, results):
                success_count += 1
            _report_progress(cutoff_idx, n_forecasts, results['abs_pct_error'], progress_interval)
        except Exception as e:
            results['errors'].append(f"Cutoff {cutoff}: {str(e)}")
            print(f"Benchmark forecast failed at cutoff {cutoff}: {e}")

    _finalize_forecasting_results(results, success_count)
    return results


def _process_single_ridge_cutoff(
    df: pd.DataFrame,
    cutoff: int,
    model_features: List[str],
    target_variable: str,
    sign_correction_config: Dict[str, Any],
    bootstrap_config: Dict[str, Any],
    forecasting_config: Dict[str, Any],
    results: Dict[str, Any]
) -> bool:
    """Process single cutoff for Ridge forecasting. Returns True on success."""
    prepared_data = prepare_cutoff_data_complete(
        df=df, cutoff=cutoff, feature_columns=model_features,
        target_column=target_variable, config=sign_correction_config
    )

    X_train, y_train = prepared_data['X_train'], prepared_data['y_train']
    X_test, y_test = prepared_data['X_test'], prepared_data['y_test']
    weights = prepared_data['weights']
    forecast_date = str(df.iloc[cutoff]['date'])[:10]

    bootstrap_models = fit_bootstrap_ensemble_atomic(
        X=X_train, y=np.log(y_train), weights=weights, config=bootstrap_config
    )

    bootstrap_predictions = np.exp(predict_bootstrap_ensemble_atomic(
        bootstrap_models, X_test
    ))

    bootstrap_validation = validate_bootstrap_predictions_atomic(
        bootstrap_predictions=bootstrap_predictions, y_true=y_test,
        config={'n_bootstrap_samples': forecasting_config['n_bootstrap_samples'],
                'positive_constraint': bootstrap_config['positive_constraint'],
                'reasonable_multiple': 10.0, 'min_cv': 0.001}
    )
    if not all(bootstrap_validation.values()):
        print(f"[ERROR] Bootstrap validation failed at cutoff {cutoff}")

    ensemble_prediction = np.mean(bootstrap_predictions)
    error_metrics = calculate_prediction_error_atomic(y_test, ensemble_prediction)

    results['dates'].append(forecast_date)
    results['y_true'].append(y_test)
    results['y_predict'].append(ensemble_prediction)
    results['abs_pct_error'].append(error_metrics['percentage_error'] / 100)
    results['bootstrap_predictions'][forecast_date] = bootstrap_predictions
    results['cutoffs'].append(cutoff)
    return True


def run_bootstrap_ridge_forecasting(
    df: pd.DataFrame,
    start_cutoff: int,
    end_cutoff: int,
    model_features: List[str],
    target_variable: str,
    sign_correction_config: Dict[str, Any],
    bootstrap_config: Dict[str, Any],
    forecasting_config: Dict[str, Any],
    progress_interval: int = 25
) -> Dict[str, Any]:
    """
    Orchestrate Bootstrap Ridge regression forecasting workflow.

    This function coordinates the complete Bootstrap Ridge forecasting by preparing
    data at each cutoff, fitting bootstrap ensembles, generating predictions, and
    validating results. Follows the exact algorithm from original notebook cell 19.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset with temporal structure
    start_cutoff : int
        First observation index for forecasting
    end_cutoff : int
        Last observation index for forecasting (exclusive)
    model_features : List[str]
        Feature columns for Ridge regression
    target_variable : str
        Target column name for prediction
    sign_correction_config : Dict[str, Any]
        Configuration for economic theory sign corrections
    bootstrap_config : Dict[str, Any]
        Bootstrap model configuration (alpha, positive constraint, etc.)
    forecasting_config : Dict[str, Any]
        General forecasting configuration (n_bootstrap_samples, random_state, etc.)
    progress_interval : int, default=25
        Report progress every N forecasts

    Returns
    -------
    Dict[str, Any]
        Model forecasting results with structure matching original notebook.

    Business Context
    ----------------
    Bootstrap Ridge uses competitive intelligence features to forecast sales.
    Bootstrap ensemble provides uncertainty quantification.
    Expected performance: R² ≈ 0.783, MAPE ≈ 12.7%
    """
    validate_forecasting_inputs(df=df, start_cutoff=start_cutoff, end_cutoff=end_cutoff,
                                features=model_features, target_variable=target_variable)

    results = _initialize_forecasting_results()
    n_forecasts = end_cutoff - start_cutoff
    success_count = 0

    for cutoff_idx, cutoff in enumerate(range(start_cutoff, end_cutoff)):
        try:
            if _process_single_ridge_cutoff(df, cutoff, model_features, target_variable,
                                           sign_correction_config, bootstrap_config,
                                           forecasting_config, results):
                success_count += 1
            _report_progress(cutoff_idx, n_forecasts, results['abs_pct_error'], progress_interval)
        except Exception as e:
            results['errors'].append(f"Cutoff {cutoff}: {str(e)}")
            print(f"Model forecast failed at cutoff {cutoff}: {e}")

    _finalize_forecasting_results(results, success_count)
    return results


def _extract_pipeline_config(forecasting_config: Dict[str, Any],
                             n_observations: int) -> Dict[str, Any]:
    """Extract and prepare pipeline configuration parameters."""
    cv_config = forecasting_config['cv_config']
    return {
        'cv_config': cv_config,
        'bootstrap_config': forecasting_config['bootstrap_model_config'],
        'performance_config': forecasting_config['performance_monitoring_config'],
        'model_features': forecasting_config.get('model_features',
            ['prudential_rate_current', 'competitor_mid_t2', 'competitor_top5_t3']),
        'benchmark_features': forecasting_config.get('benchmark_features', ['sales_target_contract_t5']),
        'target_variable': forecasting_config.get('target_variable', 'sales_target_current'),
        'start_cutoff': cv_config['start_cutoff'],
        'end_cutoff': cv_config['end_cutoff'] or n_observations,
        'n_bootstrap_samples': forecasting_config['forecasting_config']['n_bootstrap_samples'],
        'random_state': forecasting_config['forecasting_config']['random_state']
    }


def _print_phase_results(phase_name: str, results: Dict[str, Any]) -> None:
    """Print phase completion results."""
    print(f"\n[COMPLETE] {phase_name} Complete:")
    print(f"   Forecasts: {results['n_forecasts']}")
    print(f"   R2: {results['metrics']['r2_score']:.6f}")
    print(f"   MAPE: {results['metrics']['mape']:.2f}%")


def _calculate_comparison_metrics(benchmark_results: Dict[str, Any],
                                  model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate performance comparison between benchmark and model."""
    mape_improvement = ((benchmark_results['metrics']['mape'] - model_results['metrics']['mape']) /
                        benchmark_results['metrics']['mape']) * 100
    r2_improvement = ((model_results['metrics']['r2_score'] - benchmark_results['metrics']['r2_score']) /
                      max(benchmark_results['metrics']['r2_score'], 0.001)) * 100
    return {
        'mape_improvement_pct': mape_improvement,
        'r2_improvement_pct': r2_improvement,
        'model_outperforms': model_results['metrics']['mape'] < benchmark_results['metrics']['mape']
    }


def run_forecasting_pipeline(
    df: pd.DataFrame,
    forecasting_config: Dict[str, Any],
    model_sign_correction_config: Dict[str, Any],
    benchmark_sign_correction_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main orchestrator for complete forecasting pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Complete filtered dataset ready for forecasting
    forecasting_config : Dict[str, Any]
        Complete forecasting configuration from config_builder
    model_sign_correction_config : Dict[str, Any]
        Sign correction configuration for model features
    benchmark_sign_correction_config : Dict[str, Any]
        Sign correction configuration for benchmark features

    Returns
    -------
    Dict[str, Any]
        Complete pipeline results with benchmark, model, and comparison.

    Business Context
    ----------------
    Provides complete forecasting analysis with baseline comparison.
    """
    config = _extract_pipeline_config(forecasting_config, len(df))

    print("=" * 80 + "\nFORECASTING PIPELINE ORCHESTRATION\n" + "=" * 80)
    print(f"Dataset: {len(df)} observations, Forecasting range: {config['start_cutoff']} to {config['end_cutoff']-1}")
    print(f"Total forecasts: {config['end_cutoff'] - config['start_cutoff']}, Bootstrap samples: {config['n_bootstrap_samples']}")

    print("\n[PHASE 1] Benchmark Forecasting\n" + "-" * 80)
    benchmark_results = run_benchmark_forecasting(
        df=df, start_cutoff=config['start_cutoff'], end_cutoff=config['end_cutoff'],
        benchmark_features=config['benchmark_features'], n_bootstrap_samples=config['n_bootstrap_samples'],
        random_state=config['random_state'],
        progress_interval=config['performance_config']['progress_reporting_interval']
    )
    _print_phase_results("Benchmark", benchmark_results)

    print("\n[PHASE 2] Bootstrap Ridge Forecasting\n" + "-" * 80)
    model_results = run_bootstrap_ridge_forecasting(
        df=df, start_cutoff=config['start_cutoff'], end_cutoff=config['end_cutoff'],
        model_features=config['model_features'], target_variable=config['target_variable'],
        sign_correction_config=model_sign_correction_config,
        bootstrap_config=config['bootstrap_config'],
        forecasting_config=forecasting_config['forecasting_config'],
        progress_interval=config['performance_config']['progress_reporting_interval']
    )
    _print_phase_results("Model", model_results)

    comparison = _calculate_comparison_metrics(benchmark_results, model_results)
    print(f"\n[RESULTS] Performance Comparison:")
    print(f"   MAPE improvement: {comparison['mape_improvement_pct']:.1f}%")
    print(f"   R2 improvement: {comparison['r2_improvement_pct']:.1f}%")
    print(f"   Model status: {'Outperforms benchmark' if comparison['model_outperforms'] else 'Similar to benchmark'}")
    print("\n" + "=" * 80 + "\nPIPELINE COMPLETE\n" + "=" * 80)

    return {
        'benchmark_results': benchmark_results, 'model_results': model_results, 'comparison': comparison,
        'config_used': {
            'model_features': config['model_features'], 'benchmark_features': config['benchmark_features'],
            'target_variable': config['target_variable'], 'n_bootstrap_samples': config['n_bootstrap_samples'],
            'start_cutoff': config['start_cutoff'], 'end_cutoff': config['end_cutoff']
        }
    }
