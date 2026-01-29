"""
Bootstrap Stability Analysis Engine for Feature Selection.

Enhanced version with integrated visualization capabilities for comprehensive
bootstrap analysis and stability assessment.

Key Functions:
- run_bootstrap_stability: Main atomic function for bootstrap analysis
- run_core_bootstrap_analysis: Enhanced analysis with visualization integration
- calculate_bootstrap_metrics: Single model bootstrap evaluation
- assess_model_stability: Stability classification and interpretation
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Dict, List, Any, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import types - Fail fast with clear error if imports fail
from src.features.selection_types import (
    BootstrapAnalysisConfig,
    BootstrapResult,
    AICResult
)


def _run_single_model_bootstrap(data: pd.DataFrame, model_row: pd.Series,
                                target_variable: str, n_samples: int,
                                idx: int) -> BootstrapResult:
    """Run bootstrap for a single model with error context."""
    model_features = model_row['features'].split(' + ')
    try:
        return calculate_bootstrap_metrics(
            data=data, model_features=model_features, target_variable=target_variable,
            original_aic=model_row['aic'], original_r2=model_row['r_squared'], n_samples=n_samples
        )
    except ValueError as e:
        raise ValueError(f"Bootstrap failed for model {idx + 1} ({' + '.join(model_features)}): {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error for model {idx + 1} ({' + '.join(model_features)}): {e}") from e


def run_bootstrap_stability(data: pd.DataFrame, valid_models_df: pd.DataFrame,
                           config: BootstrapAnalysisConfig,
                           target_variable: str) -> List[BootstrapResult]:
    """
    Run bootstrap stability analysis for top models.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis dataset
    valid_models_df : pd.DataFrame
        Valid models from AIC evaluation
    config : BootstrapAnalysisConfig
        Bootstrap configuration
    target_variable : str
        Target variable name

    Returns
    -------
    List[BootstrapResult]
        Bootstrap results
    """
    if not config.get('enabled', True):
        return []

    n_samples = config.get('n_samples', 100)
    models_to_analyze = min(config.get('models_to_analyze', 10), len(valid_models_df))

    return [_run_single_model_bootstrap(data, valid_models_df.iloc[idx], target_variable, n_samples, idx)
            for idx in range(models_to_analyze)]


def _generate_bootstrap_seeds(
    n_samples: int, base_seed: int = None
) -> np.ndarray:
    """
    Generate independent random seeds for proper bootstrap sampling.

    Single responsibility: Random seed generation only.
    Follows UNIFIED_CODING_STANDARDS.md with focused randomization logic.

    Parameters
    ----------
    n_samples : int
        Number of bootstrap samples needed
    base_seed : int, optional
        Master seed for reproducible random generation.
        Defaults to DEFAULT_RANDOM_SEED from constants.

    Returns
    -------
    np.ndarray
        Array of independent random seeds for bootstrap sampling
    """
    from src.config.constants import DEFAULT_RANDOM_SEED

    if base_seed is None:
        base_seed = DEFAULT_RANDOM_SEED

    # This fixes the critical bug of using sequential seeds (0, 1, 2, ...) which creates
    # predictable and correlated bootstrap samples that violate bootstrap theory
    rng = np.random.RandomState(base_seed)
    independent_seeds = rng.randint(0, 2**31-1, size=n_samples)
    return independent_seeds


def _run_bootstrap_sampling(data: pd.DataFrame,
                           formula: str,
                           independent_seeds: np.ndarray,
                           n_samples: int) -> Tuple[List[float], List[float], int]:
    """
    Execute bootstrap sampling and model fitting loop.

    Single responsibility: Bootstrap sampling and model fitting only.
    Follows UNIFIED_CODING_STANDARDS.md with focused sampling logic.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis dataset for bootstrap sampling
    formula : str
        OLS formula string for model fitting
    independent_seeds : np.ndarray
        Array of independent random seeds
    n_samples : int
        Number of bootstrap samples to generate

    Returns
    -------
    Tuple[List[float], List[float], int]
        (bootstrap_aics, bootstrap_r2_values, successful_fits)
    """
    bootstrap_aics = []
    bootstrap_r2_values = []
    successful_fits = 0

    for i in range(n_samples):
        try:
            # Bootstrap sample with replacement using independent random seeds
            # This ensures truly random, uncorrelated bootstrap samples
            bootstrap_sample = data.sample(n=len(data), replace=True, random_state=independent_seeds[i])

            # Fit model on bootstrap sample
            model = smf.ols(formula, data=bootstrap_sample).fit()

            # Check if model fit was successful (statsmodels OLS doesn't have .converged attribute)
            if hasattr(model, 'aic') and not (np.isnan(model.aic) or np.isinf(model.aic)):
                bootstrap_aics.append(model.aic)
                bootstrap_r2_values.append(model.rsquared_adj)
                successful_fits += 1

        except (ValueError, np.linalg.LinAlgError) as e:
            # Expected failures: singular matrix, insufficient data, perfect collinearity
            logger.debug(f"Bootstrap model fit {i+1}/{n_samples} failed: {e}")
            continue
        except KeyError as e:
            # Formula references non-existent column in bootstrap sample
            logger.warning(f"Bootstrap formula error in sample {i+1}: {e}")
            continue

    return bootstrap_aics, bootstrap_r2_values, successful_fits


def _compute_single_stability_coefficient(values: List[float],
                                          original_value: float) -> float:
    """
    Compute stability coefficient for a single metric using relative measures.

    Parameters
    ----------
    values : List[float]
        Bootstrap values for the metric
    original_value : float
        Original model value for reference scaling

    Returns
    -------
    float
        Stability coefficient (coefficient of variation or robust alternative)
    """
    mean_val = np.mean(values)

    if abs(mean_val) > 1e-10:
        return np.std(values) / abs(mean_val)

    if abs(original_value) > 1e-10:
        return np.std(values) / abs(original_value)

    # Both mean and original near zero - use robust IQR-based measure
    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    median_val = np.median(values)
    return iqr / abs(median_val) if abs(median_val) > 1e-10 else float('inf')


def _calculate_stability_coefficients(bootstrap_aics: List[float],
                                    bootstrap_r2_values: List[float],
                                    original_aic: float,
                                    original_r2: float) -> Tuple[float, float]:
    """
    Calculate AIC and R-squared stability coefficients using relative measures.

    Orchestrator delegating to single-metric stability calculation.

    Parameters
    ----------
    bootstrap_aics : List[float]
        Bootstrap AIC values from successful fits
    bootstrap_r2_values : List[float]
        Bootstrap R-squared values from successful fits
    original_aic : float
        Original model AIC score for reference scaling
    original_r2 : float
        Original model R-squared score for reference scaling

    Returns
    -------
    Tuple[float, float]
        (aic_stability_coefficient, r2_stability_coefficient)
    """
    aic_stability = _compute_single_stability_coefficient(bootstrap_aics, original_aic)
    r2_stability = _compute_single_stability_coefficient(bootstrap_r2_values, original_r2)
    return aic_stability, r2_stability


def _calculate_confidence_intervals(bootstrap_aics: List[float],
                                  bootstrap_r2_values: List[float]) -> Dict[str, Dict[str, float]]:
    """
    Calculate confidence intervals for bootstrap metrics.

    Single responsibility: Confidence interval calculation only.
    Follows UNIFIED_CODING_STANDARDS.md with focused statistical computation.

    Parameters
    ----------
    bootstrap_aics : List[float]
        Bootstrap AIC values from successful fits
    bootstrap_r2_values : List[float]
        Bootstrap R² values from successful fits

    Returns
    -------
    Dict[str, Dict[str, float]]
        Confidence intervals for each level: {level: {aic_lower, aic_upper, r2_lower, r2_upper}}
    """
    confidence_intervals = {}

    # Implement confidence levels from config: [50, 70, 90]
    confidence_levels = [50, 70, 90]
    for level in confidence_levels:
        alpha = (100 - level) / 2  # Two-tailed alpha
        lower_percentile = alpha
        upper_percentile = 100 - alpha

        confidence_intervals[str(level)] = {
            'aic_lower': np.percentile(bootstrap_aics, lower_percentile),
            'aic_upper': np.percentile(bootstrap_aics, upper_percentile),
            'r2_lower': np.percentile(bootstrap_r2_values, lower_percentile),
            'r2_upper': np.percentile(bootstrap_r2_values, upper_percentile)
        }

    return confidence_intervals


def _validate_bootstrap_sample_count(bootstrap_aics: List[float],
                                     n_samples: int,
                                     formula: str) -> None:
    """
    Validate that sufficient bootstrap samples succeeded.

    Parameters
    ----------
    bootstrap_aics : List[float]
        Successfully computed bootstrap AIC values
    n_samples : int
        Total bootstrap attempts
    formula : str
        Model formula for error context

    Raises
    ------
    ValueError
        If insufficient successful fits
    """
    min_required = max(30, int(n_samples * 0.5))
    if len(bootstrap_aics) < min_required:
        raise ValueError(
            f"Bootstrap analysis failed: only {len(bootstrap_aics)} successful fits out of {n_samples} attempts. "
            f"Need at least {min_required} successful fits for reliable stability analysis. "
            f"Check model formula: {formula}"
        )


def _build_bootstrap_result(model_features: List[str],
                            bootstrap_aics: List[float],
                            bootstrap_r2_values: List[float],
                            original_aic: float,
                            original_r2: float,
                            aic_stability: float,
                            r2_stability: float,
                            confidence_intervals: Dict[str, Dict[str, float]],
                            successful_fits: int,
                            n_samples: int,
                            stability_assessment: str) -> BootstrapResult:
    """Construct BootstrapResult from computed metrics."""
    return BootstrapResult(
        model_name="Model 1",  # Updated by caller
        model_features=' + '.join(model_features),
        bootstrap_aics=bootstrap_aics,
        bootstrap_r2_values=bootstrap_r2_values,
        original_aic=original_aic,
        original_r2=original_r2,
        aic_stability_coefficient=aic_stability,
        r2_stability_coefficient=r2_stability,
        confidence_intervals=confidence_intervals,
        successful_fits=successful_fits,
        total_attempts=n_samples,
        stability_assessment=stability_assessment
    )


def calculate_bootstrap_metrics(data: pd.DataFrame, model_features: List[str],
                               target_variable: str, original_aic: float,
                               original_r2: float, n_samples: int = 100) -> BootstrapResult:
    """Calculate bootstrap metrics for a single model. Orchestrates helper functions."""
    formula = f"{target_variable} ~ {' + '.join(model_features)}"

    try:
        independent_seeds = _generate_bootstrap_seeds(n_samples)
        bootstrap_aics, bootstrap_r2_values, successful_fits = _run_bootstrap_sampling(
            data=data, formula=formula, independent_seeds=independent_seeds, n_samples=n_samples
        )
        _validate_bootstrap_sample_count(bootstrap_aics, n_samples, formula)

        aic_stability, r2_stability = _calculate_stability_coefficients(
            bootstrap_aics, bootstrap_r2_values, original_aic, original_r2
        )
        stability_assessment = assess_model_stability(aic_stability, r2_stability, successful_fits, n_samples)
        confidence_intervals = _calculate_confidence_intervals(bootstrap_aics, bootstrap_r2_values)

        return _build_bootstrap_result(
            model_features, bootstrap_aics, bootstrap_r2_values, original_aic, original_r2,
            aic_stability, r2_stability, confidence_intervals, successful_fits, n_samples, stability_assessment
        )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error in bootstrap analysis: {str(e)}") from e


def assess_model_stability(aic_cv: float,
                         r2_cv: float,
                         successful_fits: int,
                         total_attempts: int) -> str:
    """
    Assess model stability with business interpretation.

    Parameters
    ----------
    aic_cv : float
        AIC coefficient of variation
    r2_cv : float
        R² coefficient of variation
    successful_fits : int
        Number of successful bootstrap fits
    total_attempts : int
        Total bootstrap attempts

    Returns
    -------
    str
        Stability assessment: "STABLE", "MODERATE", "UNSTABLE", or "FAILED"
    """
    # Check fit success rate
    success_rate = successful_fits / total_attempts if total_attempts > 0 else 0.0

    if success_rate < 0.5:
        return "FAILED"

    # Stability thresholds based on coefficient of variation
    if aic_cv < 0.005 and r2_cv < 0.1:
        return "STABLE"
    elif aic_cv < 0.01 and r2_cv < 0.2:
        return "MODERATE"
    else:
        return "UNSTABLE"


def _validate_bootstrap_config(config: Dict[str, Any],
                               valid_models_df: pd.DataFrame) -> Tuple[int, int]:
    """
    Validate bootstrap configuration and extract parameters.

    Parameters
    ----------
    config : Dict[str, Any]
        Bootstrap configuration dictionary
    valid_models_df : pd.DataFrame
        Valid models from AIC evaluation

    Returns
    -------
    Tuple[int, int]
        (n_samples, models_to_analyze)

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")

    n_samples = config.get('n_samples', 100)
    models_to_analyze = min(config.get('models_to_analyze', 10), len(valid_models_df))

    if n_samples < 5:
        raise ValueError(f"Insufficient bootstrap samples: {n_samples}. Need at least 5 for stability analysis.")

    if models_to_analyze < 1:
        raise ValueError(f"No models to analyze. Available: {len(valid_models_df)}")

    return n_samples, models_to_analyze


def _print_bootstrap_summary(bootstrap_results: List[BootstrapResult]) -> None:
    """
    Print summary of bootstrap stability analysis results.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results to summarize
    """
    stability_summary: Dict[str, int] = {}
    for result in bootstrap_results:
        assessment = result.stability_assessment
        stability_summary[assessment] = stability_summary.get(assessment, 0) + 1

    print(f"\nBootstrap Stability Analysis Results:")
    print(f"  Models analyzed: {len(bootstrap_results)}")
    print(f"  Stability Assessment Summary:")
    for assessment, count in stability_summary.items():
        print(f"    {assessment}: {count} models")


def _run_bootstrap_with_viz(bootstrap_results: List[BootstrapResult], config: Dict[str, Any],
                           create_visualizations: bool) -> Optional[Dict[str, plt.Figure]]:
    """Create visualizations if requested, handling errors gracefully."""
    if not create_visualizations:
        return None
    try:
        viz = _create_bootstrap_visualizations(bootstrap_results, config)
        print("SUCCESS: Core bootstrap visualizations created")
        return viz
    except Exception as e:
        print(f"WARNING: Bootstrap visualization creation failed: {str(e)}")
        return {}


def run_core_bootstrap_analysis(data: pd.DataFrame, valid_models_df: pd.DataFrame,
                               config: Dict[str, Any], target_variable: str,
                               create_visualizations: bool = True) -> Tuple[List[BootstrapResult], Optional[Dict[str, plt.Figure]]]:
    """Run core bootstrap stability analysis with integrated visualizations."""
    if not config.get('enabled', True):
        return ([], {}) if create_visualizations else ([], None)

    n_samples, models_to_analyze = _validate_bootstrap_config(config, valid_models_df)
    print(f"Starting core bootstrap stability analysis...")
    print(f"  Models to analyze: {models_to_analyze}")
    print(f"  Bootstrap samples per model: {n_samples}")

    try:
        bootstrap_results = run_bootstrap_stability(
            data=data, valid_models_df=valid_models_df, config=config, target_variable=target_variable
        )
    except Exception as e:
        raise RuntimeError(f"Bootstrap stability analysis failed: {str(e)}") from e

    if not bootstrap_results:
        raise RuntimeError("No bootstrap results generated - check configuration and data")

    _print_bootstrap_summary(bootstrap_results)
    visualizations = _run_bootstrap_with_viz(bootstrap_results, config, create_visualizations)
    print("SUCCESS: Core bootstrap analysis complete")
    return bootstrap_results, visualizations


def _prepare_bootstrap_viz_data(bootstrap_results: List[BootstrapResult],
                               n_models_display: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare bootstrap data for visualization.

    Single responsibility: Data preparation for visualization only.
    Follows UNIFIED_CODING_STANDARDS.md with focused data processing.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results
    n_models_display : int
        Number of models to include in visualization

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        (bootstrap_dataframe, models_order_list)
    """
    # Convert bootstrap results to DataFrame for visualization
    bootstrap_viz_data = []
    for i, result in enumerate(bootstrap_results[:n_models_display]):
        for j, aic_val in enumerate(result.bootstrap_aics):
            bootstrap_viz_data.append({
                'model': f'Model {i+1}',
                'model_features': result.model_features,
                'bootstrap_aic': aic_val,
                'bootstrap_r2': result.bootstrap_r2_values[j] if j < len(result.bootstrap_r2_values) else np.nan,
                'original_aic': result.original_aic,
                'original_r2': result.original_r2,
                'stability_assessment': result.stability_assessment
            })

    bootstrap_df = pd.DataFrame(bootstrap_viz_data)
    models_order = [f'Model {i+1}' for i in range(n_models_display)]

    if bootstrap_df.empty:
        raise ValueError("No visualization data available")

    return bootstrap_df, models_order


def _get_stability_color(stability: str) -> Any:
    """
    Get color based on stability assessment.

    Parameters
    ----------
    stability : str
        Stability classification: STABLE, MODERATE, or UNSTABLE

    Returns
    -------
    Any
        Seaborn color tuple
    """
    if stability == 'STABLE':
        return sns.color_palette("deep")[2]  # Green
    elif stability == 'MODERATE':
        return sns.color_palette("deep")[1]  # Orange
    return sns.color_palette("deep")[3]  # Red for unstable


def _render_violin_kde(ax: plt.Axes, model_data: np.ndarray, i: int,
                       stability: str, original_aic: float) -> None:
    """
    Render KDE violin shape for a single model.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on
    model_data : np.ndarray
        Bootstrap AIC values for this model
    i : int
        Model index (vertical position)
    stability : str
        Stability assessment string
    original_aic : float
        Original model AIC for marker
    """
    color = _get_stability_color(stability)

    p1, p99 = np.percentile(model_data, [2.5, 97.5])
    x_range = np.linspace(p1, p99, 100)

    kde = stats.gaussian_kde(model_data)
    density = kde(x_range)
    density_scaled = density / density.max() * 0.35

    ax.fill_between(x_range, i, i + density_scaled,
                   color=color, alpha=0.7, label=stability if i == 0 else "")
    ax.plot(x_range, i + density_scaled, color=color, linewidth=1.5)

    ax.axhline(y=i, color='#CCCCCC', linewidth=0.5, alpha=0.8)
    ax.scatter(original_aic, i, color='black', s=60, zorder=10,
              edgecolors='white', linewidth=1.5)

    median_val = np.median(model_data)
    ax.plot([median_val, median_val], [i-0.1, i+0.3],
           color='darkblue', linewidth=2.5, zorder=5)


def _format_violin_axes(ax: plt.Axes, bootstrap_df: pd.DataFrame,
                        models_order: List[str]) -> None:
    """
    Apply formatting to violin plot axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to format
    bootstrap_df : pd.DataFrame
        Bootstrap data for y-tick labels
    models_order : List[str]
        Ordered list of model names
    """
    ax.set_xlabel('AIC Score (Lower is Better)')
    ax.set_ylabel('Model Ranking')
    ax.set_title(f'Bootstrap AIC Distribution Analysis - Top {len(models_order)} Models',
                 fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels([f'{model}\n({bootstrap_df[bootstrap_df["model"]==model]["stability_assessment"].iloc[0]})'
                       for model in models_order])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3)
    ax.grid(False, axis='y')


def _create_violin_plot(bootstrap_df: pd.DataFrame, models_order: List[str]) -> plt.Figure:
    """
    Create enhanced violin plot visualization for bootstrap AIC distributions.

    Orchestrator delegating to KDE rendering and formatting helpers.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Bootstrap data prepared for visualization
    models_order : List[str]
        Ordered list of model names

    Returns
    -------
    plt.Figure
        Violin plot figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    for i, model in enumerate(models_order):
        model_data = bootstrap_df[bootstrap_df['model'] == model]['bootstrap_aic'].values
        stability = bootstrap_df[bootstrap_df['model'] == model]['stability_assessment'].iloc[0]
        original_aic = bootstrap_df[bootstrap_df['model'] == model]['original_aic'].iloc[0]

        if len(model_data) > 5:
            _render_violin_kde(ax, model_data, i, stability, original_aic)

    _format_violin_axes(ax, bootstrap_df, models_order)
    plt.tight_layout()
    return fig


def _prepare_boxplot_data(bootstrap_df: pd.DataFrame,
                          models_order: List[str]) -> Tuple[List[np.ndarray], List[str], List[Any]]:
    """
    Prepare data arrays for boxplot visualization.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Bootstrap data prepared for visualization
    models_order : List[str]
        Ordered list of model names

    Returns
    -------
    Tuple[List[np.ndarray], List[str], List[Any]]
        (boxplot_data, model_labels, colors)
    """
    boxplot_data = []
    model_labels = []
    colors = []

    for model in models_order:
        model_data = bootstrap_df[bootstrap_df['model'] == model]['bootstrap_aic'].values
        stability = bootstrap_df[bootstrap_df['model'] == model]['stability_assessment'].iloc[0]

        if len(model_data) > 5:
            boxplot_data.append(model_data)
            features = bootstrap_df[bootstrap_df['model'] == model]['model_features'].iloc[0]
            short_features = features[:35] + "..." if len(features) > 35 else features
            model_labels.append(f'{model}\n{short_features}\n({stability})')
            colors.append(_get_stability_color(stability))

    return boxplot_data, model_labels, colors


def _render_boxplot(ax: plt.Axes, boxplot_data: List[np.ndarray],
                    model_labels: List[str], colors: List[Any]) -> None:
    """
    Render boxplot with styling on given axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to render on
    boxplot_data : List[np.ndarray]
        Data arrays for each box
    model_labels : List[str]
        Labels for each box
    colors : List[Any]
        Colors for each box
    """
    box_plot = ax.boxplot(boxplot_data, tick_labels=model_labels, patch_artist=True,
                          orientation='vertical', boxprops=dict(linewidth=1.2),
                          medianprops=dict(linewidth=2, color='darkblue'),
                          whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Models (Ranked by Original AIC)')
    ax.set_ylabel('Bootstrap AIC Scores (Lower is Better)')
    ax.set_title(f'Bootstrap AIC Distribution - Top {len(boxplot_data)} Models',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _create_bootstrap_boxplot(bootstrap_df: pd.DataFrame, models_order: List[str],
                            fig_width: int) -> plt.Figure:
    """
    Create enhanced boxplot visualization for bootstrap AIC distributions.

    Orchestrator delegating to data preparation and rendering helpers.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Bootstrap data prepared for visualization
    models_order : List[str]
        Ordered list of model names
    fig_width : int
        Figure width for the plot

    Returns
    -------
    plt.Figure
        Boxplot figure
    """
    fig, ax = plt.subplots(figsize=(fig_width, 10))

    boxplot_data, model_labels, colors = _prepare_boxplot_data(bootstrap_df, models_order)

    if boxplot_data:
        _render_boxplot(ax, boxplot_data, model_labels, colors)
        plt.tight_layout()

    return fig


def _create_bootstrap_visualizations(bootstrap_results: List[BootstrapResult],
                                   config: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """Create bootstrap visualizations (violin + boxplot) with error handling."""
    if not bootstrap_results:
        raise ValueError("No bootstrap results available for visualization")

    n_models_display = min(config.get('n_models_display', 15), len(bootstrap_results))
    fig_width = config.get('fig_width', 16)
    visualizations: Dict[str, plt.Figure] = {}

    try:
        bootstrap_df, models_order = _prepare_bootstrap_viz_data(bootstrap_results, n_models_display)
        try:
            visualizations['violin_plot'] = _create_violin_plot(bootstrap_df, models_order)
        except Exception as e:
            print(f"WARNING: Violin plot creation failed: {str(e)}")
        try:
            visualizations['boxplot'] = _create_bootstrap_boxplot(bootstrap_df, models_order, fig_width)
        except Exception as e:
            print(f"WARNING: Boxplot creation failed: {str(e)}")
    except Exception as e:
        print(f"ERROR: Bootstrap visualization preparation failed: {str(e)}")
        raise

    return visualizations