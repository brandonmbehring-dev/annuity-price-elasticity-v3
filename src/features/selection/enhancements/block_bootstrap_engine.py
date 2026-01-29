"""
Block Bootstrap Engine for Time Series Feature Selection.

This module addresses Issue #4 from the mathematical analysis report:
Time Series Bootstrap Violations where standard i.i.d. bootstrap is used
on time-dependent sales data, violating fundamental bootstrap assumptions
and leading to invalid confidence intervals.

Key Functions:
- run_block_bootstrap_stability: Time series-appropriate bootstrap resampling
- create_temporal_blocks: Contiguous time block generation
- calculate_block_bootstrap_ci: Valid confidence intervals for time series
- compare_bootstrap_methods: i.i.d. vs block bootstrap comparison

Critical Statistical Issues Addressed:
- Issue #4: Time Series Bootstrap Violations (SEVERITY: HIGH)
- Independence Violation: Standard bootstrap assumes i.i.d. observations
- Temporal Structure Destroyed: Random sampling breaks time ordering
- Invalid Confidence Intervals: Coverage rates much lower than nominal
- Biased Standard Errors: Underestimate uncertainty in autocorrelated data

Mathematical Foundation:
- Block Bootstrap: Sample contiguous blocks [t, t+1, ..., t+b-1]
- Block Size Selection: b = 4 weeks for weekly data (preserves short-term dependencies)
- Overlap Allowance: Blocks can overlap to maintain sample size
- Temporal Preservation: Maintains autocorrelation structure within blocks

Design Principles:
- Time series-appropriate resampling methodology
- Configurable block sizes based on data frequency
- Comparison with standard bootstrap to show improvement
- Integration with existing stability analysis framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BlockBootstrapResult:
    """
    Container for block bootstrap analysis result.

    Attributes
    ----------
    model_features : str
        Feature combination tested
    block_size : int
        Block size used for resampling
    n_bootstrap_samples : int
        Number of bootstrap samples generated
    bootstrap_aics : List[float]
        AIC values from bootstrap samples
    bootstrap_r_squareds : List[float]
        R² values from bootstrap samples
    bootstrap_coefficients : List[Dict[str, float]]
        Coefficient estimates from each bootstrap sample
    confidence_intervals : Dict[str, Tuple[float, float]]
        Confidence intervals for model parameters
    stability_metrics : Dict[str, float]
        Stability assessment metrics
    successful_fits : int
        Number of successful bootstrap fits
    total_attempts : int
        Total bootstrap attempts made
    temporal_structure_preserved : bool
        Whether temporal dependencies maintained
    comparison_with_standard : Optional[Dict[str, Any]]
        Comparison with standard i.i.d. bootstrap if available
    """
    model_features: str
    block_size: int
    n_bootstrap_samples: int
    bootstrap_aics: List[float]
    bootstrap_r_squareds: List[float]
    bootstrap_coefficients: List[Dict[str, float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    stability_metrics: Dict[str, float]
    successful_fits: int
    total_attempts: int
    temporal_structure_preserved: bool
    comparison_with_standard: Optional[Dict[str, Any]] = None


def _validate_block_bootstrap_data(data: pd.DataFrame, block_size: int) -> None:
    """Validate data has sufficient observations for block bootstrap."""
    if len(data) < block_size:
        raise ValueError(
            f"CRITICAL: Insufficient data for block bootstrap. "
            f"Need at least {block_size} observations, got {len(data)}. "
            f"Business impact: Cannot preserve temporal structure. "
            f"Required action: Use smaller block size or more data."
        )


def _sort_data_by_date(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Sort data chronologically by date column."""
    if date_column in data.columns:
        return data.sort_values(date_column).reset_index(drop=True)
    logger.warning("No date column found - assuming data is chronologically sorted")
    return data.reset_index(drop=True)


def _create_single_block(data_sorted: pd.DataFrame, start_idx: int,
                         block_size: int, date_column: str) -> Dict[str, Any]:
    """Create a single temporal block with metadata."""
    end_idx = start_idx + block_size
    block = data_sorted.iloc[start_idx:end_idx].copy()

    block_start_date = (block.iloc[0][date_column] if date_column in block.columns
                        else f"Period_{start_idx}")
    block_end_date = (block.iloc[-1][date_column] if date_column in block.columns
                      else f"Period_{end_idx-1}")

    return {
        'block': block,
        'start_index': start_idx,
        'end_index': end_idx - 1,
        'start_date': block_start_date,
        'end_date': block_end_date,
        'temporal_span': f"{block_start_date} to {block_end_date}"
    }


def _create_overlapping_blocks(data_sorted: pd.DataFrame, block_size: int,
                               date_column: str) -> List[Dict[str, Any]]:
    """Create overlapping temporal blocks starting at each position."""
    max_start = len(data_sorted) - block_size
    return [
        _create_single_block(data_sorted, start_idx, block_size, date_column)
        for start_idx in range(max_start + 1)
    ]


def _create_non_overlapping_blocks(data_sorted: pd.DataFrame, block_size: int,
                                   date_column: str) -> List[Dict[str, Any]]:
    """Create non-overlapping sequential temporal blocks."""
    blocks = []
    for start_idx in range(0, len(data_sorted), block_size):
        end_idx = min(start_idx + block_size, len(data_sorted))
        if end_idx - start_idx >= block_size:
            blocks.append(_create_single_block(data_sorted, start_idx, block_size, date_column))
    return blocks


def create_temporal_blocks(data: pd.DataFrame,
                         block_size: int = 4,
                         n_blocks_needed: int = None,
                         overlap_allowed: bool = True,
                         date_column: str = 'date') -> List[pd.DataFrame]:
    """Create contiguous temporal blocks for block bootstrap resampling."""
    _validate_block_bootstrap_data(data, block_size)
    data_sorted = _sort_data_by_date(data, date_column)

    if overlap_allowed:
        available_blocks = _create_overlapping_blocks(data_sorted, block_size, date_column)
    else:
        available_blocks = _create_non_overlapping_blocks(data_sorted, block_size, date_column)

    if len(available_blocks) == 0:
        raise ValueError(
            f"CRITICAL: No valid blocks created. "
            f"Block size {block_size} too large for data length {len(data)}. "
            f"Business impact: Cannot perform block bootstrap. "
            f"Required action: Reduce block size or increase data."
        )

    logger.info(f"Created {len(available_blocks)} temporal blocks "
                f"({'overlapping' if overlap_allowed else 'non-overlapping'}) "
                f"of size {block_size}")

    return [block_info['block'] for block_info in available_blocks]


def _analyze_single_model_bootstrap(model_row: pd.Series,
                                    target_variable: str,
                                    temporal_blocks: List[pd.DataFrame],
                                    data: pd.DataFrame,
                                    n_bootstrap_samples: int,
                                    confidence_levels: List[float],
                                    block_size: int,
                                    compare_with_standard: bool) -> BlockBootstrapResult:
    """Run bootstrap analysis for a single model row."""
    features = model_row['features']
    formula = f"{target_variable} ~ {features.replace(' + ', ' + ')}"

    block_result = _run_single_model_block_bootstrap(
        formula=formula,
        temporal_blocks=temporal_blocks,
        original_data=data,
        n_bootstrap_samples=n_bootstrap_samples,
        confidence_levels=confidence_levels,
        block_size=block_size
    )

    if compare_with_standard:
        standard_comparison = _compare_with_standard_bootstrap(
            formula=formula,
            data=data,
            n_bootstrap_samples=min(n_bootstrap_samples, 200),
            block_bootstrap_result=block_result
        )
        block_result.comparison_with_standard = standard_comparison

    return block_result


def run_block_bootstrap_stability(model_results: pd.DataFrame,
                                data: pd.DataFrame,
                                target_variable: str,
                                n_bootstrap_samples: int = 1000,
                                block_size: int = 4,
                                models_to_analyze: int = 15,
                                confidence_levels: List[float] = [90, 95, 99],
                                compare_with_standard: bool = True,
                                random_seed: Optional[int] = None) -> List[BlockBootstrapResult]:
    """Run block bootstrap stability analysis for time series feature selection."""
    try:
        if random_seed is not None:
            np.random.seed(random_seed)

        top_models = model_results.nsmallest(models_to_analyze, 'aic')
        temporal_blocks = create_temporal_blocks(data, block_size=block_size, overlap_allowed=True)

        logger.info(f"Running block bootstrap analysis on {len(top_models)} models "
                    f"with {n_bootstrap_samples} samples, block size {block_size}")

        bootstrap_results = []
        for idx, (_, model_row) in enumerate(top_models.iterrows()):
            logger.info(f"Analyzing model {idx+1}/{len(top_models)}: {model_row['features']}")
            result = _analyze_single_model_bootstrap(
                model_row, target_variable, temporal_blocks, data,
                n_bootstrap_samples, confidence_levels, block_size, compare_with_standard
            )
            bootstrap_results.append(result)

        logger.info(f"Block bootstrap analysis completed for {len(bootstrap_results)} models")
        return bootstrap_results

    except Exception as e:
        raise ValueError(
            f"CRITICAL: Block bootstrap analysis failed. "
            f"Business impact: Cannot assess time series model stability. "
            f"Required action: Check data format and model specifications. "
            f"Original error: {e}"
        ) from e


def _create_bootstrap_sample(temporal_blocks: List[pd.DataFrame],
                            n_blocks_needed: int, original_n: int) -> pd.DataFrame:
    """Create a single bootstrap sample from temporal blocks."""
    selected_blocks = np.random.choice(len(temporal_blocks), size=n_blocks_needed, replace=True)
    bootstrap_sample = pd.concat(
        [temporal_blocks[block_idx] for block_idx in selected_blocks], ignore_index=True
    )
    if len(bootstrap_sample) > original_n:
        bootstrap_sample = bootstrap_sample.iloc[:original_n]
    return bootstrap_sample


def _fit_bootstrap_model(formula: str, bootstrap_sample: pd.DataFrame) -> Optional[Tuple[float, float, Dict]]:
    """Fit model on bootstrap sample and return results or None on failure."""
    import statsmodels.formula.api as smf
    try:
        model = smf.ols(formula, data=bootstrap_sample).fit()
        return model.aic, model.rsquared, dict(model.params)
    except (ValueError, np.linalg.LinAlgError) as e:
        # Expected failures: singular matrix, insufficient data, perfect collinearity
        logger.debug(f"Bootstrap model fit failed: {e}")
        return None
    except KeyError as e:
        # Formula references non-existent column
        logger.warning(f"Bootstrap model formula error: {e}")
        return None


def _run_bootstrap_iterations(formula: str, temporal_blocks: List[pd.DataFrame],
                              original_n: int, n_blocks_needed: int,
                              n_bootstrap_samples: int) -> Tuple[List, List, List, int]:
    """Run all bootstrap iterations and collect results."""
    bootstrap_aics, bootstrap_r2s, bootstrap_coefficients = [], [], []
    successful_fits = 0

    for sample_idx in range(n_bootstrap_samples):
        bootstrap_sample = _create_bootstrap_sample(temporal_blocks, n_blocks_needed, original_n)
        result = _fit_bootstrap_model(formula, bootstrap_sample)
        if result is not None:
            aic, r2, coeffs = result
            bootstrap_aics.append(aic)
            bootstrap_r2s.append(r2)
            bootstrap_coefficients.append(coeffs)
            successful_fits += 1
        else:
            logger.debug(f"Bootstrap sample {sample_idx} failed")

    return bootstrap_aics, bootstrap_r2s, bootstrap_coefficients, successful_fits


def _run_single_model_block_bootstrap(formula: str,
                                    temporal_blocks: List[pd.DataFrame],
                                    original_data: pd.DataFrame,
                                    n_bootstrap_samples: int,
                                    confidence_levels: List[float],
                                    block_size: int) -> BlockBootstrapResult:
    """Run block bootstrap for single model specification."""
    original_n = len(original_data)
    n_blocks_needed = int(np.ceil(original_n / block_size))

    bootstrap_aics, bootstrap_r2s, bootstrap_coefficients, successful_fits = _run_bootstrap_iterations(
        formula, temporal_blocks, original_n, n_blocks_needed, n_bootstrap_samples
    )

    if successful_fits < n_bootstrap_samples * 0.5:
        warnings.warn(
            f"Block bootstrap had low success rate: {successful_fits}/{n_bootstrap_samples} "
            f"({successful_fits/n_bootstrap_samples:.1%}). Results may be unreliable."
        )

    confidence_intervals = _calculate_bootstrap_confidence_intervals(bootstrap_coefficients, confidence_levels)
    stability_metrics = _calculate_block_bootstrap_stability_metrics(
        bootstrap_aics, bootstrap_r2s, bootstrap_coefficients
    )

    return BlockBootstrapResult(
        model_features=formula.split('~')[1].strip(),
        block_size=block_size,
        n_bootstrap_samples=n_bootstrap_samples,
        bootstrap_aics=bootstrap_aics,
        bootstrap_r_squareds=bootstrap_r2s,
        bootstrap_coefficients=bootstrap_coefficients,
        confidence_intervals=confidence_intervals,
        stability_metrics=stability_metrics,
        successful_fits=successful_fits,
        total_attempts=n_bootstrap_samples,
        temporal_structure_preserved=True
    )


def _get_all_coefficient_names(bootstrap_coefficients: List[Dict[str, float]]) -> set:
    """Extract all unique coefficient names from bootstrap results."""
    all_names = set()
    for coef_dict in bootstrap_coefficients:
        all_names.update(coef_dict.keys())
    return all_names


def _extract_coefficient_values(bootstrap_coefficients: List[Dict[str, float]],
                                coef_name: str) -> np.ndarray:
    """Extract non-NaN values for a specific coefficient."""
    values = [
        coef_dict[coef_name] for coef_dict in bootstrap_coefficients
        if coef_name in coef_dict and not np.isnan(coef_dict[coef_name])
    ]
    return np.array(values) if values else np.array([])


def _compute_percentile_ci(coef_array: np.ndarray, conf_level: float) -> Tuple[float, float]:
    """Compute percentile-based confidence interval for a coefficient."""
    alpha = (100 - conf_level) / 100
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    return np.percentile(coef_array, lower_percentile), np.percentile(coef_array, upper_percentile)


def _calculate_bootstrap_confidence_intervals(bootstrap_coefficients: List[Dict[str, float]],
                                            confidence_levels: List[float]) -> Dict[str, Tuple[float, float]]:
    """Calculate bootstrap confidence intervals for model coefficients."""
    if not bootstrap_coefficients:
        return {}

    all_coef_names = _get_all_coefficient_names(bootstrap_coefficients)
    confidence_intervals = {}

    for coef_name in all_coef_names:
        coef_array = _extract_coefficient_values(bootstrap_coefficients, coef_name)
        if len(coef_array) > 0:
            for conf_level in confidence_levels:
                lower_ci, upper_ci = _compute_percentile_ci(coef_array, conf_level)
                confidence_intervals[f"{coef_name}_{conf_level}pct"] = (lower_ci, upper_ci)

    return confidence_intervals


def _compute_metric_stability(values: List[float], prefix: str) -> Dict[str, float]:
    """Compute stability metrics (mean, median, std, cv) for a metric."""
    if not values:
        return {}
    arr = np.array(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {}
    mean_val = np.mean(arr)
    return {
        f'{prefix}_mean': mean_val,
        f'{prefix}_median': np.median(arr),
        f'{prefix}_std': np.std(arr),
        f'{prefix}_cv': np.std(arr) / mean_val if mean_val != 0 else np.inf
    }


def _compute_coefficient_stability(bootstrap_coefficients: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute coefficient variation metrics across all coefficients."""
    if not bootstrap_coefficients:
        return {}

    all_coef_names = _get_all_coefficient_names(bootstrap_coefficients)
    coef_cvs = []

    for coef_name in all_coef_names:
        coef_array = _extract_coefficient_values(bootstrap_coefficients, coef_name)
        if len(coef_array) > 1:
            coef_mean = np.mean(coef_array)
            if abs(coef_mean) > 1e-10:
                coef_cvs.append(abs(np.std(coef_array) / coef_mean))

    if coef_cvs:
        return {'coefficient_cv_mean': np.mean(coef_cvs), 'coefficient_cv_max': np.max(coef_cvs)}
    return {}


def _assess_overall_stability(aic_cv: float, r2_cv: float) -> Tuple[str, int]:
    """Determine stability assessment and score from CV values."""
    if aic_cv < 0.002 and r2_cv < 0.05:
        return "HIGHLY_STABLE", 95
    elif aic_cv < 0.005 and r2_cv < 0.1:
        return "STABLE", 85
    elif aic_cv < 0.01 and r2_cv < 0.2:
        return "MODERATE", 65
    elif aic_cv < 0.02 and r2_cv < 0.3:
        return "UNSTABLE", 40
    return "HIGHLY_UNSTABLE", 20


def _calculate_block_bootstrap_stability_metrics(bootstrap_aics: List[float],
                                               bootstrap_r2s: List[float],
                                               bootstrap_coefficients: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate stability metrics for block bootstrap results."""
    stability_metrics = {}
    stability_metrics.update(_compute_metric_stability(bootstrap_aics, 'aic'))
    stability_metrics.update(_compute_metric_stability(bootstrap_r2s, 'r2'))
    stability_metrics.update(_compute_coefficient_stability(bootstrap_coefficients))

    aic_cv = stability_metrics.get('aic_cv', np.inf)
    r2_cv = stability_metrics.get('r2_cv', np.inf)
    assessment, score = _assess_overall_stability(aic_cv, r2_cv)
    stability_metrics['stability_assessment'] = assessment
    stability_metrics['stability_score'] = score

    return stability_metrics


def _run_standard_bootstrap(formula: str, data: pd.DataFrame,
                           n_bootstrap_samples: int) -> Tuple[List[float], List[float]]:
    """Run standard i.i.d. bootstrap and return AIC and R2 lists."""
    import statsmodels.formula.api as smf
    standard_aics, standard_r2s = [], []

    for _ in range(n_bootstrap_samples):
        try:
            bootstrap_indices = np.random.choice(len(data), size=len(data), replace=True)
            bootstrap_sample = data.iloc[bootstrap_indices]
            model = smf.ols(formula, data=bootstrap_sample).fit()
            standard_aics.append(model.aic)
            standard_r2s.append(model.rsquared)
        except (ValueError, np.linalg.LinAlgError) as e:
            # Expected failures: singular matrix, insufficient data, perfect collinearity
            logger.debug(f"Standard bootstrap model fit failed: {e}")
            continue
        except KeyError as e:
            # Formula references non-existent column
            logger.warning(f"Standard bootstrap formula error: {e}")
            continue

    return standard_aics, standard_r2s


def _build_comparison_results(standard_aic_cv: float, standard_r2_cv: float,
                              block_aic_cv: float, block_r2_cv: float) -> Dict[str, Any]:
    """Build comparison results dictionary from CV values."""
    return {
        'standard_bootstrap_aic_cv': standard_aic_cv,
        'standard_bootstrap_r2_cv': standard_r2_cv,
        'block_bootstrap_aic_cv': block_aic_cv,
        'block_bootstrap_r2_cv': block_r2_cv,
        'aic_cv_improvement': (standard_aic_cv - block_aic_cv) / standard_aic_cv if standard_aic_cv > 0 else 0,
        'r2_cv_improvement': (standard_r2_cv - block_r2_cv) / standard_r2_cv if standard_r2_cv > 0 else 0,
        'block_bootstrap_superior': block_aic_cv < standard_aic_cv and block_r2_cv < standard_r2_cv,
        'interpretation': (
            "Block bootstrap shows better stability (lower CV) than standard bootstrap"
            if block_aic_cv < standard_aic_cv else
            "Standard bootstrap shows similar or better stability (unexpected for time series)"
        )
    }


def _compare_with_standard_bootstrap(formula: str,
                                   data: pd.DataFrame,
                                   n_bootstrap_samples: int,
                                   block_bootstrap_result: BlockBootstrapResult) -> Dict[str, Any]:
    """Compare block bootstrap with standard i.i.d. bootstrap."""
    try:
        standard_aics, standard_r2s = _run_standard_bootstrap(formula, data, n_bootstrap_samples)

        standard_aic_cv = np.std(standard_aics) / np.mean(standard_aics) if standard_aics else np.inf
        standard_r2_cv = np.std(standard_r2s) / np.mean(standard_r2s) if standard_r2s else np.inf
        block_aic_cv = block_bootstrap_result.stability_metrics.get('aic_cv', np.inf)
        block_r2_cv = block_bootstrap_result.stability_metrics.get('r2_cv', np.inf)

        return _build_comparison_results(standard_aic_cv, standard_r2_cv, block_aic_cv, block_r2_cv)

    except Exception as e:
        logger.warning(f"Standard bootstrap comparison failed: {e}")
        return {'comparison_failed': True, 'error': str(e)}


def _aggregate_block_size_metrics(block_results: List[BlockBootstrapResult]) -> Dict[str, Any]:
    """Aggregate stability metrics across bootstrap results."""
    return {
        'average_aic_cv': np.mean([r.stability_metrics.get('aic_cv', np.inf) for r in block_results]),
        'average_r2_cv': np.mean([r.stability_metrics.get('r2_cv', np.inf) for r in block_results]),
        'average_stability_score': np.mean([r.stability_metrics.get('stability_score', 0) for r in block_results]),
        'models_tested': len(block_results)
    }


def _test_single_block_size(top_models: pd.DataFrame, data: pd.DataFrame,
                            target_variable: str, block_size: int,
                            n_bootstrap_samples: int, models_to_test: int) -> Dict[str, Any]:
    """Test a single block size and return metrics or error."""
    try:
        logger.info(f"Testing block size: {block_size}")
        block_results = run_block_bootstrap_stability(
            model_results=top_models, data=data, target_variable=target_variable,
            n_bootstrap_samples=n_bootstrap_samples, block_size=block_size,
            models_to_analyze=models_to_test, compare_with_standard=False
        )
        return _aggregate_block_size_metrics(block_results)
    except Exception as e:
        logger.warning(f"Block size {block_size} analysis failed: {e}")
        return {'error': str(e), 'analysis_failed': True}


def _determine_best_block_size(block_size_analysis: Dict[int, Dict]) -> Dict[str, Any]:
    """Determine the best block size from analysis results."""
    valid_results = {bs: m for bs, m in block_size_analysis.items() if 'error' not in m}
    if not valid_results:
        return {'recommended_block_size': None, 'analysis_summary': {}}

    best_block_size = max(valid_results.keys(), key=lambda bs: valid_results[bs]['average_stability_score'])
    return {
        'recommended_block_size': best_block_size,
        'analysis_summary': {
            'block_sizes_tested': list(valid_results.keys()),
            'best_block_size': best_block_size,
            'best_stability_score': valid_results[best_block_size]['average_stability_score'],
            'recommendation': f"Block size {best_block_size} provides optimal balance of "
                              "temporal structure preservation and bootstrap efficiency"
        }
    }


def assess_block_size_sensitivity(model_results: pd.DataFrame,
                                data: pd.DataFrame,
                                target_variable: str,
                                block_sizes: List[int] = [2, 4, 6, 8],
                                n_bootstrap_samples: int = 200,
                                models_to_test: int = 3) -> Dict[str, Any]:
    """Assess sensitivity to block size selection for optimal parameter tuning."""
    top_models = model_results.nsmallest(models_to_test, 'aic')

    block_size_analysis = {
        block_size: _test_single_block_size(
            top_models, data, target_variable, block_size, n_bootstrap_samples, models_to_test
        )
        for block_size in block_sizes
    }

    result = {'block_size_analysis': block_size_analysis}
    result.update(_determine_best_block_size(block_size_analysis))
    return result