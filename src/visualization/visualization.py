"""
Charts and analysis outputs for clean_v0 pipeline.

This module handles AIC feature selection visualization and model performance charts
for RILA price elasticity analysis. Functions follow clean atomic function
patterns (20-50 lines each) with comprehensive error handling.

Following CODING_STANDARDS.md principles:
- Single responsibility functions (20-50 lines max)
- Full type hints with TypedDict configurations
- Immutable operations (return new DataFrames)
- Explicit error handling with clear messages
- AIC visualization with economic interpretation plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

# Defensive imports for optional visualization dependencies
try:
    from src.config.pipeline_config import VisualizationConfig
    VISUALIZATION_CONFIG_AVAILABLE = True
except ImportError:
    VISUALIZATION_CONFIG_AVAILABLE = False
    VisualizationConfig = None


def _validate_aic_results_for_comparison(
    aic_results: pd.DataFrame
) -> None:
    """
    Validate AIC results have required columns and data for comparison plot.

    Parameters
    ----------
    aic_results : pd.DataFrame
        Results from calculate_aic_scores

    Raises
    ------
    ValueError
        If AIC results are empty or missing required columns
    """
    if len(aic_results) == 0:
        raise ValueError(
            "CRITICAL: AIC results are empty - no data to plot. "
            "Business impact: Cannot visualize feature selection analysis. "
            "Required action: Verify AIC feature selection completed successfully and produced results."
        )

    required_cols = ['features', 'aic_score', 'r_squared', 'n_features']
    missing_cols = [col for col in required_cols if col not in aic_results.columns]
    if missing_cols:
        raise ValueError(
            f"CRITICAL: Missing required columns: {missing_cols}. "
            f"Business impact: Cannot create model comparison plots. "
            f"Available columns: {list(aic_results.columns)}. "
            f"Required action: Verify AIC analysis produced complete results with all required metrics."
        )


def _filter_valid_models_for_comparison(
    aic_results: pd.DataFrame,
    max_models: int
) -> pd.DataFrame:
    """
    Filter to valid converged models and select top N by AIC score.

    Parameters
    ----------
    aic_results : pd.DataFrame
        Results from calculate_aic_scores
    max_models : int
        Maximum number of models to return

    Returns
    -------
    pd.DataFrame
        Filtered and sorted plot data

    Raises
    ------
    ValueError
        If no valid converged models found
    """
    plot_data = aic_results[
        (aic_results['model_converged'] == True) &
        (~np.isinf(aic_results['aic_score']))
    ].nsmallest(max_models, 'aic_score').copy()

    if len(plot_data) == 0:
        raise ValueError(
            "CRITICAL: No valid converged models found for plotting. "
            "Business impact: All AIC models either failed to converge or have infinite scores. "
            "Required action: Review model specification and data quality for convergence issues."
        )

    return plot_data


def _create_aic_scores_bar_plot(
    ax: plt.Axes,
    plot_data: pd.DataFrame,
    title: str
) -> None:
    """
    Create bar plot of AIC scores for top models.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    plot_data : pd.DataFrame
        Filtered AIC results
    title : str
        Base title for the plot
    """
    ax.bar(range(len(plot_data)), plot_data['aic_score'],
           color='steelblue', alpha=0.7)
    ax.set_title(f"{title} - Top {len(plot_data)} Models by AIC")
    ax.set_xlabel("Model Rank")
    ax.set_ylabel("AIC Score (lower is better)")
    ax.grid(True, alpha=0.3)


def _create_rsquared_feature_scatter(
    ax: plt.Axes,
    plot_data: pd.DataFrame
) -> None:
    """
    Create scatter plot of R-squared vs feature count colored by AIC.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    plot_data : pd.DataFrame
        Filtered AIC results with n_features, r_squared, aic_score columns
    """
    scatter = ax.scatter(plot_data['n_features'], plot_data['r_squared'],
                        c=plot_data['aic_score'], cmap='viridis_r',
                        alpha=0.7, s=60)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("R-squared")
    ax.set_title("Model Performance: R² vs Feature Count (colored by AIC)")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='AIC Score')


def plot_aic_model_comparison(
    aic_results: pd.DataFrame,
    max_models: int = 20,
    title: str = "AIC Model Comparison"
) -> plt.Figure:
    """
    Create bar plot comparing AIC scores across different feature combinations.

    Parameters
    ----------
    aic_results : pd.DataFrame
        Results from calculate_aic_scores with columns: features, aic_score, r_squared
    max_models : int
        Maximum number of models to display (default: 20)
    title : str
        Plot title (default: "AIC Model Comparison")

    Returns
    -------
    plt.Figure
        Matplotlib figure object with AIC comparison plot

    Raises
    ------
    ValueError
        If AIC results are empty or missing required columns

    Examples
    --------
    >>> fig = plot_aic_model_comparison(aic_results, max_models=15)
    >>> plt.show()
    """
    _validate_aic_results_for_comparison(aic_results)
    plot_data = _filter_valid_models_for_comparison(aic_results, max_models)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    _create_aic_scores_bar_plot(ax1, plot_data, title)
    _create_rsquared_feature_scatter(ax2, plot_data)

    plt.tight_layout()
    return fig


def _validate_aic_results_for_progression(
    aic_results: pd.DataFrame
) -> None:
    """
    Validate AIC results are non-empty for progression plot.

    Parameters
    ----------
    aic_results : pd.DataFrame
        Complete AIC analysis results

    Raises
    ------
    ValueError
        If AIC results are empty
    """
    if len(aic_results) == 0:
        raise ValueError(
            "CRITICAL: No AIC results provided for progression plot. "
            "Business impact: Cannot visualize feature selection progression. "
            "Required action: Verify AIC feature selection completed successfully before plotting."
        )


def _compute_aic_progression_by_feature_count(
    aic_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter valid models and compute best AIC for each feature count.

    Parameters
    ----------
    aic_results : pd.DataFrame
        Complete AIC analysis results

    Returns
    -------
    pd.DataFrame
        Progression data with n_features and best aic_score columns

    Raises
    ------
    ValueError
        If no valid converged models found
    """
    valid_results = aic_results[
        (aic_results['model_converged'] == True) &
        (~np.isinf(aic_results['aic_score']))
    ].copy()

    if len(valid_results) == 0:
        raise ValueError(
            "CRITICAL: No valid models found for progression analysis. "
            "Business impact: Cannot show how feature count affects model performance. "
            "Required action: Review convergence criteria and ensure at least some models converge successfully."
        )

    progression = valid_results.groupby('n_features')['aic_score'].min().reset_index()
    return progression.sort_values('n_features')


def _create_aic_progression_line_plot(
    ax: plt.Axes,
    progression: pd.DataFrame,
    best_aic: float
) -> None:
    """
    Create line plot showing AIC progression by feature count.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    progression : pd.DataFrame
        Progression data with n_features and aic_score columns
    best_aic : float
        Best AIC score to highlight
    """
    ax.plot(progression['n_features'], progression['aic_score'],
            marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(y=best_aic, color='red', linestyle='--', alpha=0.7,
               label=f'Selected Model AIC: {best_aic:.2f}')
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Best AIC Score")
    ax.set_title("AIC Progression by Feature Count")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _create_selected_features_bar_chart(
    ax: plt.Axes,
    selected_features: List[str],
    best_aic: float
) -> None:
    """
    Create horizontal bar chart showing selected features by priority.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    selected_features : List[str]
        Features in the optimal model
    best_aic : float
        Best AIC score for title
    """
    ax.barh(range(len(selected_features)),
            [len(selected_features) - i for i in range(len(selected_features))],
            color='lightcoral', alpha=0.7)
    ax.set_yticks(range(len(selected_features)))
    ax.set_yticklabels([f.replace('_', ' ').title() for f in selected_features])
    ax.set_xlabel("Selection Priority")
    ax.set_title(f"Selected Features (AIC: {best_aic:.2f})")
    ax.grid(True, alpha=0.3)


def plot_feature_selection_progression(
    aic_results: pd.DataFrame,
    selected_features: List[str],
    best_aic: float
) -> plt.Figure:
    """
    Visualize AIC progression showing how adding features affects model performance.

    Parameters
    ----------
    aic_results : pd.DataFrame
        Complete AIC analysis results
    selected_features : List[str]
        List of features in the optimal model
    best_aic : float
        Best AIC score achieved

    Returns
    -------
    plt.Figure
        Matplotlib figure showing feature selection progression

    Examples
    --------
    >>> fig = plot_feature_selection_progression(results, features, aic_score)
    >>> plt.show()
    """
    _validate_aic_results_for_progression(aic_results)
    progression = _compute_aic_progression_by_feature_count(aic_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    _create_aic_progression_line_plot(ax1, progression, best_aic)
    _create_selected_features_bar_chart(ax2, selected_features, best_aic)

    plt.tight_layout()
    return fig


def _extract_coefficient_averages(
    feature_coefficients: Dict[str, Dict[str, float]]
) -> Tuple[List[float], List[float]]:
    """Extract average competitor and prudential coefficients per model."""
    competitor_coeffs = []
    prudential_coeffs = []

    for features_string, coeffs in feature_coefficients.items():
        comp_coeff = [v for k, v in coeffs.items()
                     if 'competitor_' in k and k != 'Intercept']
        prud_coeff = [v for k, v in coeffs.items()
                     if 'prudential_rate' in k and k != 'Intercept']

        competitor_coeffs.append(np.mean(comp_coeff) if comp_coeff else 0)
        prudential_coeffs.append(np.mean(prud_coeff) if prud_coeff else 0)

    return competitor_coeffs, prudential_coeffs


def _plot_coefficient_bar(
    ax: plt.Axes, coeffs: List[float], expected_positive: bool, title: str
) -> None:
    """Plot coefficient bar chart with economic theory coloring."""
    colors = ['green' if (x > 0) == expected_positive else 'red' for x in coeffs[:10]]
    ax.bar(range(len(coeffs[:10])), coeffs[:10], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Model Rank")
    ax.set_ylabel("Average Coefficient Value")
    ax.grid(True, alpha=0.3)


def plot_economic_constraint_validation(
    aic_results: pd.DataFrame,
    feature_coefficients: Dict[str, Dict[str, float]]
) -> plt.Figure:
    """Visualize economic constraint compliance across models."""
    if len(aic_results) == 0 or len(feature_coefficients) == 0:
        raise ValueError(
            "CRITICAL: Insufficient data for economic constraint visualization. "
            f"AIC results: {len(aic_results)}, Coefficients: {len(feature_coefficients)}."
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    competitor_coeffs, prudential_coeffs = _extract_coefficient_averages(feature_coefficients)

    _plot_coefficient_bar(ax1, competitor_coeffs, False, "Competitor Rate Coefficients\n(Should be Negative)")
    _plot_coefficient_bar(ax2, prudential_coeffs, True, "Prudential Rate Coefficients\n(Should be Positive)")

    plt.tight_layout()
    return fig


def _validate_diagnostics_inputs(
    data: pd.DataFrame,
    selected_features: List[str],
    target_variable: str
) -> None:
    """
    Validate inputs for model diagnostics plot.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset used for modeling
    selected_features : List[str]
        Features in the optimal model
    target_variable : str
        Target variable name

    Raises
    ------
    ValueError
        If target variable or features not found in data
    """
    if target_variable not in data.columns:
        raise ValueError(
            f"CRITICAL: Target variable '{target_variable}' not found in data. "
            f"Business impact: Cannot create model diagnostic plots for validation. "
            f"Available columns: {list(data.columns)}. "
            f"Required action: Verify target variable name matches feature-engineered dataset."
        )

    missing_features = [f for f in selected_features if f not in data.columns]
    if missing_features:
        raise ValueError(
            f"CRITICAL: Missing features in data: {missing_features}. "
            f"Business impact: Cannot create complete diagnostic plots for selected model. "
            f"Available columns: {list(data.columns)}. "
            f"Required action: Verify all selected features were properly engineered in the pipeline."
        )


def _prepare_clean_model_data(
    data: pd.DataFrame,
    selected_features: List[str],
    target_variable: str
) -> pd.DataFrame:
    """
    Prepare clean data for diagnostics by selecting columns and dropping NAs.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset used for modeling
    selected_features : List[str]
        Features in the optimal model
    target_variable : str
        Target variable name

    Returns
    -------
    pd.DataFrame
        Clean model data with no missing values

    Raises
    ------
    ValueError
        If insufficient data after cleaning
    """
    model_data = data[selected_features + [target_variable]].dropna()

    if len(model_data) < 10:
        raise ValueError(
            f"CRITICAL: Insufficient data after cleaning: {len(model_data)} observations (minimum: 10). "
            f"Business impact: Too few observations for reliable diagnostic analysis. "
            f"Required action: Review data quality and missing value handling in upstream pipeline."
        )

    return model_data


def _create_feature_correlation_heatmap(
    ax: plt.Axes,
    model_data: pd.DataFrame,
    selected_features: List[str]
) -> None:
    """
    Create correlation heatmap for selected features.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    model_data : pd.DataFrame
        Clean model data
    selected_features : List[str]
        Features to include in correlation matrix
    """
    corr_data = model_data[selected_features].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                ax=ax, square=True, fmt='.2f')
    ax.set_title("Feature Correlation Matrix")


def _create_target_vs_top_feature_scatter(
    ax: plt.Axes,
    model_data: pd.DataFrame,
    selected_features: List[str],
    target_variable: str
) -> None:
    """
    Create scatter plot of target vs top feature.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    model_data : pd.DataFrame
        Clean model data
    selected_features : List[str]
        Features in the optimal model (first feature used)
    target_variable : str
        Target variable name
    """
    if len(selected_features) > 0:
        top_feature = selected_features[0]
        ax.scatter(model_data[top_feature], model_data[target_variable],
                   alpha=0.6, color='steelblue')
        ax.set_xlabel(f"{top_feature.replace('_', ' ').title()}")
        ax.set_ylabel(f"{target_variable.replace('_', ' ').title()}")
        ax.set_title(f"Target vs {top_feature}")
        ax.grid(True, alpha=0.3)


def _create_feature_variance_bar_chart(
    ax: plt.Axes,
    model_data: pd.DataFrame,
    selected_features: List[str]
) -> None:
    """
    Create horizontal bar chart showing feature variance.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    model_data : pd.DataFrame
        Clean model data
    selected_features : List[str]
        Features to analyze
    """
    feature_vars = model_data[selected_features].var().sort_values(ascending=True)
    ax.barh(range(len(feature_vars)), feature_vars.values, color='lightgreen', alpha=0.7)
    ax.set_yticks(range(len(feature_vars)))
    ax.set_yticklabels([f.replace('_', ' ').title() for f in feature_vars.index])
    ax.set_xlabel("Feature Variance")
    ax.set_title("Feature Variance Analysis")
    ax.grid(True, alpha=0.3)


def _create_target_distribution_histogram(
    ax: plt.Axes,
    model_data: pd.DataFrame,
    target_variable: str
) -> None:
    """
    Create histogram of target variable distribution.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    model_data : pd.DataFrame
        Clean model data
    target_variable : str
        Target variable name
    """
    ax.hist(model_data[target_variable], bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel(f"{target_variable.replace('_', ' ').title()}")
    ax.set_ylabel("Frequency")
    ax.set_title("Target Variable Distribution")
    ax.grid(True, alpha=0.3)


def plot_model_diagnostics(
    data: pd.DataFrame,
    selected_features: List[str],
    target_variable: str
) -> plt.Figure:
    """
    Create diagnostic plots for the selected AIC model.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset used for modeling
    selected_features : List[str]
        Features in the optimal model
    target_variable : str
        Target variable name

    Returns
    -------
    plt.Figure
        Model diagnostic plots including residuals and feature relationships

    Examples
    --------
    >>> fig = plot_model_diagnostics(df, features, 'sales_target_current')
    >>> plt.show()
    """
    _validate_diagnostics_inputs(data, selected_features, target_variable)
    model_data = _prepare_clean_model_data(data, selected_features, target_variable)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    _create_feature_correlation_heatmap(axes[0, 0], model_data, selected_features)
    _create_target_vs_top_feature_scatter(axes[0, 1], model_data, selected_features, target_variable)
    _create_feature_variance_bar_chart(axes[1, 0], model_data, selected_features)
    _create_target_distribution_histogram(axes[1, 1], model_data, target_variable)

    plt.tight_layout()
    return fig


def _create_summary_text_section(
    ax_summary: plt.Axes,
    best_aic: float,
    selected_features: List[str],
    metadata: Dict[str, Any]
) -> None:
    """
    Create summary statistics text box for AIC report.

    Parameters
    ----------
    ax_summary : plt.Axes
        Axes for the summary text section
    best_aic : float
        Best AIC score achieved
    selected_features : List[str]
        Features selected by AIC analysis
    metadata : Dict[str, Any]
        Selection metadata with counts and performance metrics
    """
    ax_summary.axis('off')

    summary_text = f"""
AIC FEATURE SELECTION SUMMARY REPORT
=====================================

OPTIMAL MODEL PERFORMANCE:
• Best AIC Score: {best_aic:.2f} (lower is better)
• Selected Features: {len(selected_features)} features
• R-squared: {metadata.get('best_r_squared', 'N/A'):.3f}
• Observations: {metadata.get('n_observations', 'N/A'):,}

MODEL SELECTION STATISTICS:
• Total Models Tested: {metadata.get('total_models_tested', 'N/A'):,}
• Valid Models: {metadata.get('valid_models', 'N/A'):,}
• Success Rate: {100 * metadata.get('valid_models', 0) / max(metadata.get('total_models_tested', 1), 1):.1f}%

SELECTED FEATURES:
{chr(10).join([f'• {feat.replace("_", " ").title()}' for feat in selected_features[:8]])}
"""

    ax_summary.text(0.02, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))


def _create_aic_histogram(
    ax_hist: plt.Axes,
    valid_results: pd.DataFrame,
    best_aic: float
) -> None:
    """
    Create AIC distribution histogram.

    Parameters
    ----------
    ax_hist : plt.Axes
        Axes for the histogram
    valid_results : pd.DataFrame
        Filtered AIC results with valid (non-infinite) scores
    best_aic : float
        Best AIC score to highlight
    """
    ax_hist.hist(valid_results['aic_score'], bins=20, alpha=0.7,
                 color='steelblue', edgecolor='black')
    ax_hist.axvline(best_aic, color='red', linestyle='--', linewidth=2,
                    label=f'Selected: {best_aic:.1f}')
    ax_hist.set_xlabel('AIC Score')
    ax_hist.set_ylabel('Model Count')
    ax_hist.set_title('AIC Score Distribution')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)


def _create_feature_count_chart(
    ax_features: plt.Axes,
    aic_results: pd.DataFrame,
    selected_features: List[str]
) -> None:
    """
    Create feature count distribution chart.

    Parameters
    ----------
    ax_features : plt.Axes
        Axes for the feature count chart
    aic_results : pd.DataFrame
        Complete AIC analysis results
    selected_features : List[str]
        Features selected by AIC analysis
    """
    feature_counts = aic_results['n_features'].value_counts().sort_index()
    ax_features.bar(feature_counts.index, feature_counts.values,
                    alpha=0.7, color='lightcoral')
    ax_features.axvline(len(selected_features), color='red', linestyle='--',
                        linewidth=2, label=f'Selected: {len(selected_features)}')
    ax_features.set_xlabel('Number of Features')
    ax_features.set_ylabel('Model Count')
    ax_features.set_title('Feature Count Distribution')
    ax_features.legend()
    ax_features.grid(True, alpha=0.3)


def _create_convergence_pie(
    ax_convergence: plt.Axes,
    aic_results: pd.DataFrame
) -> None:
    """
    Create model convergence pie chart.

    Parameters
    ----------
    ax_convergence : plt.Axes
        Axes for the pie chart
    aic_results : pd.DataFrame
        Complete AIC analysis results with model_converged column
    """
    convergence_counts = aic_results['model_converged'].value_counts()
    colors = ['lightgreen' if idx else 'lightcoral' for idx in convergence_counts.index]
    ax_convergence.pie(convergence_counts.values, labels=['Converged', 'Failed'],
                       colors=colors, autopct='%1.1f%%', startangle=90)
    ax_convergence.set_title('Model Convergence Rate')


def _create_performance_tradeoff_chart(
    ax_performance: plt.Axes,
    grouped_results: pd.DataFrame
) -> None:
    """
    Create AIC vs R-squared dual-axis performance tradeoff chart.

    Parameters
    ----------
    ax_performance : plt.Axes
        Primary axes for the chart
    grouped_results : pd.DataFrame
        Aggregated results with n_features, aic_score, and r_squared columns
    """
    ax_perf_r2 = ax_performance.twinx()

    ax_performance.plot(grouped_results['n_features'], grouped_results['aic_score'],
                        marker='o', color='steelblue', linewidth=2, label='Best AIC')
    ax_perf_r2.plot(grouped_results['n_features'], grouped_results['r_squared'],
                    marker='s', color='darkgreen', linewidth=2, label='Best R²')

    ax_performance.set_xlabel('Number of Features')
    ax_performance.set_ylabel('AIC Score (blue)', color='steelblue')
    ax_perf_r2.set_ylabel('R-squared (green)', color='darkgreen')
    ax_performance.set_title('Model Performance Trade-offs: AIC vs R² by Feature Count')
    ax_performance.grid(True, alpha=0.3)

    # Combine legends from both axes
    lines1, labels1 = ax_performance.get_legend_handles_labels()
    lines2, labels2 = ax_perf_r2.get_legend_handles_labels()
    ax_performance.legend(lines1 + lines2, labels1 + labels2, loc='upper right')


def create_aic_summary_report(
    aic_results: pd.DataFrame,
    selected_features: List[str],
    best_aic: float,
    metadata: Dict[str, Any]
) -> plt.Figure:
    """Create comprehensive summary report with key AIC analysis metrics."""
    if len(aic_results) == 0:
        raise ValueError("CRITICAL: No AIC results provided for summary report.")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    valid_results = aic_results[~np.isinf(aic_results['aic_score'])]
    grouped_results = valid_results.groupby('n_features').agg({
        'aic_score': 'min', 'r_squared': 'max'
    }).reset_index()

    _create_summary_text_section(fig.add_subplot(gs[0, :]), best_aic, selected_features, metadata)
    _create_aic_histogram(fig.add_subplot(gs[1, 0]), valid_results, best_aic)
    _create_feature_count_chart(fig.add_subplot(gs[1, 1]), aic_results, selected_features)
    _create_convergence_pie(fig.add_subplot(gs[1, 2]), aic_results)
    _create_performance_tradeoff_chart(fig.add_subplot(gs[2, :]), grouped_results)

    return fig