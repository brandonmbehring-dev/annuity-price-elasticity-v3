"""
Out-of-Sample Evaluation Framework for Feature Selection.

This module provides comprehensive out-of-sample performance evaluation
to address the complete absence of generalization validation identified
in the mathematical analysis report. Essential for production readiness
assessment and model reliability validation.

Key Functions:
- evaluate_temporal_generalization: Train/test split performance assessment
- run_time_series_cross_validation: Rolling origin validation
- calculate_generalization_metrics: Performance gap analysis
- assess_production_readiness: Deployment suitability evaluation

Critical Statistical Issues Addressed:
- Issue #2: Complete Absence of Out-of-Sample Validation (SEVERITY: CRITICAL)
- Provides actual evidence of model generalization
- Prevents overfitting through proper validation methodology
- Enables confident production deployment decisions

Mathematical Foundation:
- Temporal Validation: Training [1, t-h], Testing [t-h+1, T]
- Generalization Gap: |R²_train - R²_test|
- Time Series CV: Expanding/rolling window validation
- Performance Metrics: R², MAPE, directional accuracy, residual analysis

Design Principles:
- Time-aware validation (no data leakage)
- Comprehensive performance metrics
- Statistical significance testing of performance differences
- Business-interpretable deployment recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta
import logging
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OutOfSampleResult:
    """
    Container for out-of-sample evaluation result.

    Attributes
    ----------
    model_features : str
        Features used in model
    train_period : Tuple[str, str]
        Training period (start, end)
    test_period : Tuple[str, str]
        Test period (start, end)
    train_performance : Dict[str, float]
        Training set performance metrics
    test_performance : Dict[str, float]
        Test set performance metrics
    generalization_metrics : Dict[str, float]
        Generalization gap analysis
    statistical_tests : Dict[str, Dict[str, float]]
        Significance tests for performance differences
    residual_analysis : Dict[str, Any]
        Residual diagnostic results
    production_assessment : Dict[str, Any]
        Production readiness evaluation
    predictions : Dict[str, List[float]]
        Actual predictions for further analysis
    """
    model_features: str
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]
    train_performance: Dict[str, float]
    test_performance: Dict[str, float]
    generalization_metrics: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, float]]
    residual_analysis: Dict[str, Any]
    production_assessment: Dict[str, Any]
    predictions: Dict[str, List[float]]


@dataclass
class CrossValidationResult:
    """
    Container for time series cross-validation results.

    Uses expanding window (TimeSeriesSplit) for all time series validation.

    Attributes
    ----------
    model_features : str
        Features used in model
    n_folds : int
        Number of CV folds
    fold_results : List[Dict[str, Any]]
        Performance results for each fold
    average_performance : Dict[str, float]
        Average performance across folds
    performance_stability : Dict[str, float]
        Stability metrics across folds
    overall_assessment : Dict[str, Any]
        Overall CV assessment
    """
    model_features: str
    n_folds: int
    fold_results: List[Dict[str, Any]]
    average_performance: Dict[str, float]
    performance_stability: Dict[str, float]
    overall_assessment: Dict[str, Any]


def _validate_temporal_split_data(temporal_split_data: Dict[str, pd.DataFrame]
                                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate temporal split data and return train/test DataFrames."""
    if 'train' not in temporal_split_data or 'test' not in temporal_split_data:
        raise ValueError(
            f"CRITICAL: Temporal split data missing required 'train' and 'test' keys. "
            f"Available keys: {list(temporal_split_data.keys())}. "
            f"Business impact: Cannot perform out-of-sample validation. "
            f"Required action: Ensure proper temporal data splitting."
        )

    train_data = temporal_split_data['train']
    test_data = temporal_split_data['test']

    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError(
            f"CRITICAL: Empty training or test dataset. "
            f"Train size: {len(train_data)}, Test size: {len(test_data)}. "
            f"Business impact: Cannot assess generalization. "
            f"Required action: Check temporal splitting logic."
        )

    return train_data, test_data


def evaluate_temporal_generalization(model_results: pd.DataFrame,
                                   temporal_split_data: Dict[str, pd.DataFrame],
                                   target_variable: str,
                                   models_to_evaluate: int = 15) -> List[OutOfSampleResult]:
    """
    Evaluate temporal generalization performance for top models.

    Provides comprehensive out-of-sample validation addressing Issue #2.
    """
    try:
        train_data, test_data = _validate_temporal_split_data(temporal_split_data)
        top_models = model_results.nsmallest(models_to_evaluate, 'aic')

        logger.info(f"Evaluating temporal generalization for {len(top_models)} models")

        out_of_sample_results = [
            _evaluate_single_model_generalization(
                model_row=model_row,
                train_data=train_data,
                test_data=test_data,
                target_variable=target_variable
            )
            for idx, (_, model_row) in enumerate(top_models.iterrows())
            if logger.info(f"Evaluating model {idx+1}/{len(top_models)}: {model_row['features']}") or True
        ]

        logger.info(f"Temporal generalization evaluation completed for {len(out_of_sample_results)} models")
        return out_of_sample_results

    except Exception as e:
        raise ValueError(
            f"CRITICAL: Temporal generalization evaluation failed. "
            f"Business impact: Cannot assess model production readiness. "
            f"Required action: Check data format and model specifications. "
            f"Original error: {e}"
        ) from e


def _extract_time_periods(train_data: pd.DataFrame,
                          test_data: pd.DataFrame) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Extract time period tuples from train and test datasets."""
    if 'date' in train_data.columns:
        train_period = (
            train_data['date'].min().strftime('%Y-%m-%d'),
            train_data['date'].max().strftime('%Y-%m-%d')
        )
        test_period = (
            test_data['date'].min().strftime('%Y-%m-%d'),
            test_data['date'].max().strftime('%Y-%m-%d')
        )
    else:
        train_period = (f"Period 1", f"Period {len(train_data)}")
        test_period = (f"Period {len(train_data)+1}", f"Period {len(train_data)+len(test_data)}")
    return train_period, test_period


def _fit_and_predict(formula: str, train_data: pd.DataFrame,
                     test_data: pd.DataFrame, target_variable: str
                     ) -> Tuple[pd.Series, np.ndarray, pd.Series, np.ndarray]:
    """Fit model on training data and generate predictions for both sets."""
    import statsmodels.formula.api as smf

    train_model = smf.ols(formula, data=train_data).fit()
    train_predictions = train_model.fittedvalues
    train_actual = train_data[target_variable]
    test_predictions = train_model.predict(test_data)
    test_actual = test_data[target_variable]
    return train_actual, train_predictions, test_actual, test_predictions


def _evaluate_single_model_generalization(model_row: pd.Series,
                                        train_data: pd.DataFrame,
                                        test_data: pd.DataFrame,
                                        target_variable: str) -> OutOfSampleResult:
    """
    Evaluate generalization for single model specification.

    Orchestrates model fitting, evaluation, and result aggregation.
    """
    features = model_row['features']
    formula = f"{target_variable} ~ {features.replace(' + ', ' + ')}"

    train_actual, train_predictions, test_actual, test_predictions = _fit_and_predict(
        formula, train_data, test_data, target_variable
    )

    train_performance = _calculate_performance_metrics(train_actual, train_predictions, "training")
    test_performance = _calculate_performance_metrics(test_actual, test_predictions, "test")
    generalization_metrics = _calculate_generalization_metrics(train_performance, test_performance)

    statistical_tests = _perform_generalization_statistical_tests(
        train_actual, train_predictions, test_actual, test_predictions
    )

    test_residuals = test_actual - test_predictions
    residual_analysis = _analyze_test_residuals(test_residuals, test_predictions)

    production_assessment = _assess_production_readiness(
        train_performance, test_performance, generalization_metrics, residual_analysis
    )

    train_period, test_period = _extract_time_periods(train_data, test_data)

    return OutOfSampleResult(
        model_features=features,
        train_period=train_period,
        test_period=test_period,
        train_performance=train_performance,
        test_performance=test_performance,
        generalization_metrics=generalization_metrics,
        statistical_tests=statistical_tests,
        residual_analysis=residual_analysis,
        production_assessment=production_assessment,
        predictions={
            'train_actual': train_actual.tolist(),
            'train_predictions': train_predictions.tolist(),
            'test_actual': test_actual.tolist(),
            'test_predictions': test_predictions.tolist()
        }
    )


def _compute_core_regression_metrics(actual_valid: np.ndarray,
                                      pred_valid: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute MSE, MAE, RMSE, and R-squared."""
    mse = np.mean((actual_valid - pred_valid) ** 2)
    mae = np.mean(np.abs(actual_valid - pred_valid))
    rmse = np.sqrt(mse)

    ss_res = np.sum((actual_valid - pred_valid) ** 2)
    ss_tot = np.sum((actual_valid - np.mean(actual_valid)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return mse, mae, rmse, r_squared


def _compute_mape(actual_valid: np.ndarray, pred_valid: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error."""
    non_zero_actual = actual_valid[np.abs(actual_valid) > 1e-10]
    pred_non_zero = pred_valid[np.abs(actual_valid) > 1e-10]
    if len(non_zero_actual) > 0:
        return np.mean(np.abs((non_zero_actual - pred_non_zero) / non_zero_actual)) * 100
    return np.inf


def _compute_directional_accuracy(actual_valid: np.ndarray, pred_valid: np.ndarray) -> float:
    """Compute directional accuracy for time series."""
    if len(actual_valid) > 1:
        actual_changes = np.diff(actual_valid)
        pred_changes = np.diff(pred_valid)
        return np.mean(np.sign(actual_changes) == np.sign(pred_changes)) * 100
    return np.nan


def _calculate_performance_metrics(actual: pd.Series,
                                 predictions: Union[pd.Series, np.ndarray],
                                 dataset_type: str) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    actual_array = np.array(actual)
    predictions_array = np.array(predictions)

    valid_mask = ~(np.isnan(actual_array) | np.isnan(predictions_array))
    actual_valid = actual_array[valid_mask]
    pred_valid = predictions_array[valid_mask]

    if len(actual_valid) == 0:
        logger.warning(f"No valid predictions for {dataset_type} dataset")
        return {'error': 'no_valid_predictions'}

    mse, mae, rmse, r_squared = _compute_core_regression_metrics(actual_valid, pred_valid)
    mape = _compute_mape(actual_valid, pred_valid)
    directional_accuracy = _compute_directional_accuracy(actual_valid, pred_valid)
    correlation = np.corrcoef(actual_valid, pred_valid)[0, 1] if len(actual_valid) > 1 else np.nan

    return {
        'r_squared': r_squared,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'directional_accuracy': directional_accuracy,
        'n_observations': len(actual_valid),
        'dataset_type': dataset_type
    }


def _compute_metric_gaps(train_performance: Dict[str, float],
                          test_performance: Dict[str, float]) -> Dict[str, float]:
    """Compute absolute and relative gaps for comparison metrics."""
    generalization_metrics = {}
    metrics_to_compare = ['r_squared', 'mape', 'mae', 'correlation']

    for metric in metrics_to_compare:
        if metric in train_performance and metric in test_performance:
            train_val = train_performance[metric]
            test_val = test_performance[metric]

            if not (np.isnan(train_val) or np.isnan(test_val)):
                gap = train_val - test_val
                generalization_metrics[f'{metric}_gap'] = gap

                if abs(train_val) > 1e-10:
                    relative_gap = (gap / train_val) * 100
                    generalization_metrics[f'{metric}_relative_gap_pct'] = relative_gap

    return generalization_metrics


def _assess_generalization_quality(r2_relative_gap: float) -> Tuple[str, int]:
    """Determine generalization quality based on R² degradation percentage."""
    if r2_relative_gap <= 5:
        return "EXCELLENT", 95
    elif r2_relative_gap <= 10:
        return "GOOD", 80
    elif r2_relative_gap <= 20:
        return "ACCEPTABLE", 65
    elif r2_relative_gap <= 40:
        return "CONCERNING", 40
    return "POOR", 20


def _calculate_generalization_metrics(train_performance: Dict[str, float],
                                    test_performance: Dict[str, float]) -> Dict[str, float]:
    """Calculate generalization gap metrics."""
    generalization_metrics = _compute_metric_gaps(train_performance, test_performance)

    r2_gap = generalization_metrics.get('r_squared_gap', 0)
    r2_relative_gap = generalization_metrics.get('r_squared_relative_gap_pct', 0)

    generalization_quality, generalization_score = _assess_generalization_quality(r2_relative_gap)

    generalization_metrics.update({
        'generalization_quality': generalization_quality,
        'generalization_score': generalization_score,
        'primary_gap_r2': r2_gap,
        'primary_gap_r2_pct': r2_relative_gap
    })

    return generalization_metrics


def _test_residual_means(train_residuals: np.ndarray,
                          test_residuals: np.ndarray) -> Dict[str, float]:
    """Test if residual means are significantly different from zero."""
    train_mean_test = stats.ttest_1samp(train_residuals, 0)
    test_mean_test = stats.ttest_1samp(test_residuals, 0)
    return {
        'train_mean': np.mean(train_residuals),
        'train_tstat': train_mean_test.statistic,
        'train_pval': train_mean_test.pvalue,
        'test_mean': np.mean(test_residuals),
        'test_tstat': test_mean_test.statistic,
        'test_pval': test_mean_test.pvalue
    }


def _test_variance_equality(train_residuals: np.ndarray,
                             test_residuals: np.ndarray) -> Dict[str, float]:
    """Test for equal variances between train and test residuals."""
    levene_stat, levene_pval = stats.levene(train_residuals, test_residuals)
    return {
        'levene_statistic': levene_stat,
        'levene_pvalue': levene_pval,
        'train_var': np.var(train_residuals),
        'test_var': np.var(test_residuals)
    }


def _test_distribution_similarity(train_residuals: np.ndarray,
                                   test_residuals: np.ndarray) -> Dict[str, Any]:
    """Test distribution similarity using Kolmogorov-Smirnov test."""
    ks_stat, ks_pval = stats.ks_2samp(train_residuals, test_residuals)
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'distributions_similar': ks_pval > 0.05
    }


def _perform_generalization_statistical_tests(train_actual: pd.Series,
                                            train_predictions: np.ndarray,
                                            test_actual: pd.Series,
                                            test_predictions: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Perform statistical tests for generalization performance differences."""
    statistical_tests = {}

    try:
        train_residuals = np.array(train_actual) - np.array(train_predictions)
        test_residuals = np.array(test_actual) - np.array(test_predictions)
        train_residuals = train_residuals[~np.isnan(train_residuals)]
        test_residuals = test_residuals[~np.isnan(test_residuals)]

        if len(train_residuals) > 0 and len(test_residuals) > 0:
            statistical_tests['residual_mean_tests'] = _test_residual_means(
                train_residuals, test_residuals
            )

            if len(train_residuals) > 1 and len(test_residuals) > 1:
                statistical_tests['variance_equality_test'] = _test_variance_equality(
                    train_residuals, test_residuals
                )

            statistical_tests['distribution_similarity_test'] = _test_distribution_similarity(
                train_residuals, test_residuals
            )

    except Exception as e:
        logger.warning(f"Statistical tests failed: {e}")
        statistical_tests['tests_failed'] = True

    return statistical_tests


def _compute_residual_basic_stats(residuals_clean: np.ndarray) -> Dict[str, float]:
    """Compute basic statistical measures for residuals."""
    return {
        'mean': np.mean(residuals_clean),
        'std': np.std(residuals_clean),
        'min': np.min(residuals_clean),
        'max': np.max(residuals_clean),
        'median': np.median(residuals_clean),
        'skewness': stats.skew(residuals_clean) if len(residuals_clean) > 2 else np.nan,
        'kurtosis': stats.kurtosis(residuals_clean) if len(residuals_clean) > 2 else np.nan
    }


def _run_normality_tests(residuals_clean: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Run normality tests on residuals."""
    from scipy.stats import shapiro, jarque_bera

    results = {}
    if len(residuals_clean) <= 50:
        sw_stat, sw_pval = shapiro(residuals_clean)
        results['normality_shapiro'] = {
            'statistic': sw_stat,
            'pvalue': sw_pval,
            'normal_at_5pct': sw_pval > 0.05
        }

    jb_stat, jb_pval = jarque_bera(residuals_clean)
    results['normality_jarque_bera'] = {
        'statistic': jb_stat,
        'pvalue': jb_pval,
        'normal_at_5pct': jb_pval > 0.05
    }
    return results


def _check_heteroscedasticity(residuals_clean: np.ndarray,
                               predictions_clean: np.ndarray) -> Dict[str, Any]:
    """Check for heteroscedasticity in residuals."""
    correlation_res_fitted = np.corrcoef(np.abs(residuals_clean), predictions_clean)[0, 1]
    return {
        'residual_fitted_correlation': correlation_res_fitted,
        'potential_heteroscedasticity': abs(correlation_res_fitted) > 0.3
    }


def _analyze_test_residuals(test_residuals: pd.Series,
                          test_predictions: np.ndarray) -> Dict[str, Any]:
    """Analyze test set residuals for model diagnostics."""
    residual_analysis = {}

    try:
        residuals_array = np.array(test_residuals)
        predictions_array = np.array(test_predictions)

        valid_mask = ~(np.isnan(residuals_array) | np.isnan(predictions_array))
        residuals_clean = residuals_array[valid_mask]
        predictions_clean = predictions_array[valid_mask]

        if len(residuals_clean) > 0:
            residual_analysis['basic_stats'] = _compute_residual_basic_stats(residuals_clean)

            if len(residuals_clean) > 3:
                residual_analysis.update(_run_normality_tests(residuals_clean))

            if len(predictions_clean) > 1:
                residual_analysis['heteroscedasticity'] = _check_heteroscedasticity(
                    residuals_clean, predictions_clean
                )

    except Exception as e:
        logger.warning(f"Residual analysis failed: {e}")
        residual_analysis['analysis_failed'] = True

    return residual_analysis


def _check_test_r2(test_r2: float, min_test_r2: float,
                   assessment: Dict[str, Any]) -> bool:
    """Check if test R² meets minimum threshold."""
    if test_r2 >= min_test_r2:
        assessment['strengths'].append(f"Good test R² ({test_r2:.3f} ≥ {min_test_r2})")
        return True
    assessment['primary_concerns'].append(f"Low test R² ({test_r2:.3f} < {min_test_r2})")
    return False


def _check_test_mape(test_mape: float, max_test_mape: float,
                     assessment: Dict[str, Any]) -> bool:
    """Check if test MAPE is within acceptable bounds."""
    if test_mape <= max_test_mape:
        assessment['strengths'].append(f"Acceptable MAPE ({test_mape:.1f}% ≤ {max_test_mape}%)")
        return True
    assessment['primary_concerns'].append(f"High MAPE ({test_mape:.1f}% > {max_test_mape}%)")
    return False


def _check_generalization(r2_gap_pct: float, max_r2_degradation: float,
                          assessment: Dict[str, Any]) -> bool:
    """Check if generalization gap is acceptable."""
    if r2_gap_pct <= max_r2_degradation:
        assessment['strengths'].append(
            f"Good generalization ({r2_gap_pct:.1f}% degradation ≤ {max_r2_degradation}%)"
        )
        return True
    assessment['primary_concerns'].append(
        f"Poor generalization ({r2_gap_pct:.1f}% degradation > {max_r2_degradation}%)"
    )
    return False


def _check_residual_diagnostics(residual_analysis: Dict[str, Any],
                                 assessment: Dict[str, Any]) -> bool:
    """Check residual diagnostics for issues."""
    residual_issues = []
    if 'basic_stats' in residual_analysis:
        residual_mean = residual_analysis['basic_stats'].get('mean', 0)
        if abs(residual_mean) > 0.1:
            residual_issues.append(f"biased residuals (mean={residual_mean:.3f})")

    if 'heteroscedasticity' in residual_analysis:
        if residual_analysis['heteroscedasticity'].get('potential_heteroscedasticity', False):
            residual_issues.append("potential heteroscedasticity")

    if residual_issues:
        assessment['primary_concerns'].extend(residual_issues)
        return False
    assessment['strengths'].append("Clean residual diagnostics")
    return True


def _determine_confidence_level(passed_checks: int, total_checks: int,
                                 assessment: Dict[str, Any]) -> None:
    """Determine production readiness and confidence level."""
    if passed_checks == total_checks:
        assessment['production_ready'] = True
        assessment['confidence_level'] = 'HIGH'
        assessment['recommendations'].append("Model ready for production deployment")
    elif passed_checks >= total_checks * 0.75:
        assessment['production_ready'] = True
        assessment['confidence_level'] = 'MODERATE'
        assessment['recommendations'].append("Model acceptable for production with monitoring")
    elif passed_checks >= total_checks * 0.5:
        assessment['production_ready'] = False
        assessment['confidence_level'] = 'LOW'
        assessment['recommendations'].append("Address major concerns before production")
    else:
        assessment['production_ready'] = False
        assessment['confidence_level'] = 'VERY LOW'
        assessment['recommendations'].append("Model requires significant improvement")


def _add_specific_recommendations(assessment: Dict[str, Any], test_r2: float,
                                   test_mape: float, r2_gap_pct: float,
                                   min_test_r2: float, max_test_mape: float,
                                   max_r2_degradation: float) -> None:
    """Add specific recommendations based on failing checks."""
    if not assessment['production_ready']:
        if test_r2 < min_test_r2:
            assessment['recommendations'].append("Consider additional features or model complexity")
        if test_mape > max_test_mape:
            assessment['recommendations'].append("Investigate prediction accuracy issues")
        if r2_gap_pct > max_r2_degradation:
            assessment['recommendations'].append("Address overfitting through regularization or more data")


def _build_summary_metrics(passed_checks: int, total_checks: int, test_r2: float,
                            test_mape: float, generalization_quality: str,
                            r2_gap_pct: float) -> Dict[str, Any]:
    """Build summary metrics dictionary for production assessment."""
    return {
        'checks_passed': passed_checks,
        'total_checks': total_checks,
        'pass_rate': passed_checks / total_checks if total_checks > 0 else 0,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'generalization_quality': generalization_quality,
        'r2_degradation_pct': r2_gap_pct
    }


def _assess_production_readiness(train_performance: Dict[str, float],
                               test_performance: Dict[str, float],
                               generalization_metrics: Dict[str, float],
                               residual_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Assess model production readiness based on out-of-sample performance."""
    assessment = {
        'production_ready': False, 'confidence_level': 'LOW',
        'primary_concerns': [], 'strengths': [], 'recommendations': []
    }

    try:
        test_r2 = test_performance.get('r_squared', 0)
        test_mape = test_performance.get('mape', np.inf)
        generalization_quality = generalization_metrics.get('generalization_quality', 'UNKNOWN')
        r2_gap_pct = generalization_metrics.get('r_squared_relative_gap_pct', np.inf)
        min_test_r2, max_test_mape, max_r2_degradation = 0.3, 25, 30

        performance_checks = [
            _check_test_r2(test_r2, min_test_r2, assessment),
            _check_test_mape(test_mape, max_test_mape, assessment),
            _check_generalization(r2_gap_pct, max_r2_degradation, assessment),
            _check_residual_diagnostics(residual_analysis, assessment)
        ]

        passed_checks, total_checks = sum(performance_checks), len(performance_checks)
        _determine_confidence_level(passed_checks, total_checks, assessment)
        _add_specific_recommendations(
            assessment, test_r2, test_mape, r2_gap_pct,
            min_test_r2, max_test_mape, max_r2_degradation
        )
        assessment['summary_metrics'] = _build_summary_metrics(
            passed_checks, total_checks, test_r2, test_mape, generalization_quality, r2_gap_pct
        )

    except Exception as e:
        logger.warning(f"Production readiness assessment failed: {e}")
        assessment['assessment_failed'] = True

    return assessment


def run_time_series_cross_validation(model_results: pd.DataFrame,
                                   data: pd.DataFrame,
                                   target_variable: str,
                                   n_splits: int = 5,
                                   min_train_size: Optional[int] = None,
                                   models_to_evaluate: int = 5) -> List[CrossValidationResult]:
    """
    Run time series cross-validation for robust performance assessment.

    Uses expanding window (TimeSeriesSplit) for all time series validation.
    This ensures each fold uses all historical data up to the validation point,
    which is appropriate for time series data where we want to use all available
    history for training.

    Parameters
    ----------
    model_results : pd.DataFrame
        DataFrame with model results and AIC scores
    data : pd.DataFrame
        Training data for cross-validation
    target_variable : str
        Name of target variable
    n_splits : int, default=5
        Number of time series splits
    min_train_size : Optional[int], default=None
        Minimum training size (defaults to max(50, 30% of data))
    models_to_evaluate : int, default=5
        Number of top models to evaluate

    Returns
    -------
    List[CrossValidationResult]
        Cross-validation results for each model
    """
    try:
        from sklearn.model_selection import TimeSeriesSplit

        if min_train_size is None:
            min_train_size = max(50, int(len(data) * 0.3))

        tscv = TimeSeriesSplit(n_splits=n_splits)
        top_models = model_results.nsmallest(models_to_evaluate, 'aic')

        cv_results = [
            _run_single_model_time_series_cv(
                model_row=model_row,
                data=data,
                target_variable=target_variable,
                tscv=tscv
            )
            for idx, (_, model_row) in enumerate(top_models.iterrows())
            if logger.info(f"Running CV for model {idx+1}/{len(top_models)}: {model_row['features']}") or True
        ]

        return cv_results

    except Exception as e:
        raise ValueError(
            f"CRITICAL: Time series cross-validation failed. "
            f"Business impact: Cannot assess model stability across time periods. "
            f"Required action: Check data format and CV configuration. "
            f"Original error: {e}"
        ) from e


def _evaluate_cv_fold(fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray,
                      data: pd.DataFrame, formula: str, target_variable: str,
                      features: str) -> Dict[str, Any]:
    """Evaluate a single CV fold and return result dictionary."""
    import statsmodels.formula.api as smf

    try:
        train_fold = data.iloc[train_idx]
        test_fold = data.iloc[test_idx]
        fold_model = smf.ols(formula, data=train_fold).fit()
        fold_predictions = fold_model.predict(test_fold)
        fold_actual = test_fold[target_variable]
        fold_performance = _calculate_performance_metrics(
            fold_actual, fold_predictions, f"fold_{fold_idx}"
        )
        return {
            'fold': fold_idx,
            'train_size': len(train_fold),
            'test_size': len(test_fold),
            'performance': fold_performance
        }
    except Exception as e:
        logger.warning(f"CV fold {fold_idx} failed for {features}: {e}")
        return {'fold': fold_idx, 'error': str(e), 'failed': True}


def _aggregate_cv_performance(successful_folds: List[Dict[str, Any]]
                              ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Aggregate performance metrics across successful CV folds."""
    avg_performance = {}
    performance_stability = {}

    for metric in ['r_squared', 'mape', 'mae', 'correlation']:
        values = [
            fold['performance'][metric] for fold in successful_folds
            if metric in fold['performance'] and not np.isnan(fold['performance'][metric])
        ]
        if values:
            avg_performance[metric] = np.mean(values)
            performance_stability[f'{metric}_std'] = np.std(values)
            mean_val = np.mean(values)
            performance_stability[f'{metric}_cv'] = np.std(values) / mean_val if mean_val > 0 else np.inf

    return avg_performance, performance_stability


def _assess_cv_quality(r2_cv: float, avg_r2: float) -> str:
    """Determine CV quality based on R² coefficient of variation and average."""
    if r2_cv < 0.1 and avg_r2 > 0.4:
        return "EXCELLENT"
    elif r2_cv < 0.2 and avg_r2 > 0.3:
        return "GOOD"
    elif r2_cv < 0.3 and avg_r2 > 0.2:
        return "ACCEPTABLE"
    return "POOR"


def _build_cv_overall_assessment(successful_folds: List[Dict[str, Any]],
                                  fold_results: List[Dict[str, Any]],
                                  avg_performance: Dict[str, float],
                                  performance_stability: Dict[str, float]) -> Dict[str, Any]:
    """Build overall CV assessment dictionary."""
    if not successful_folds:
        return {
            'cv_quality': 'FAILED',
            'successful_folds': 0,
            'total_folds': len(fold_results),
            'success_rate': 0.0
        }

    r2_cv = performance_stability.get('r_squared_cv', np.inf)
    avg_r2 = avg_performance.get('r_squared', 0)

    return {
        'cv_quality': _assess_cv_quality(r2_cv, avg_r2),
        'successful_folds': len(successful_folds),
        'total_folds': len(fold_results),
        'success_rate': len(successful_folds) / len(fold_results),
        'average_r2': avg_r2,
        'r2_stability_cv': r2_cv
    }


def _run_single_model_time_series_cv(model_row: pd.Series,
                                   data: pd.DataFrame,
                                   target_variable: str,
                                   tscv: Any) -> CrossValidationResult:
    """
    Run time series CV for single model using expanding window.

    Orchestrates fold evaluation and result aggregation.

    Parameters
    ----------
    model_row : pd.Series
        Row from model results with features
    data : pd.DataFrame
        Training data
    target_variable : str
        Name of target variable
    tscv : TimeSeriesSplit
        Time series cross-validator (expanding window)

    Returns
    -------
    CrossValidationResult
        Validation results for this model
    """
    features = model_row['features']
    formula = f"{target_variable} ~ {features.replace(' + ', ' + ')}"

    fold_results = [
        _evaluate_cv_fold(fold_idx, train_idx, test_idx, data, formula,
                          target_variable, features)
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data))
    ]

    successful_folds = [f for f in fold_results if 'failed' not in f]
    avg_performance, performance_stability = _aggregate_cv_performance(successful_folds)
    overall_assessment = _build_cv_overall_assessment(
        successful_folds, fold_results, avg_performance, performance_stability
    )

    return CrossValidationResult(
        model_features=features,
        n_folds=len(fold_results),
        fold_results=fold_results,
        average_performance=avg_performance,
        performance_stability=performance_stability,
        overall_assessment=overall_assessment
    )