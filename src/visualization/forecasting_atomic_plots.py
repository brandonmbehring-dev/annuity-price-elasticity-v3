"""
Atomic Plotting Operations for Time Series Forecasting - Reusable Visualization.

This module provides atomic plotting operations with single responsibility
for creating business-ready visualizations. All functions generate specific
plot types that can be composed for comprehensive analysis dashboards.

Key Design Principles:
- Atomic plot generation (single plot type per function)
- Business-ready formatting and styling
- Reusable across multiple notebooks and applications
- Consistent styling following unified standards
- No business logic mixing with visualization code

Plot Categories:
- Bootstrap uncertainty visualization (confidence bands)
- Performance comparison plots (model vs benchmark)
- Time series analysis plots (MAPE evolution, volatility)
- Export-ready plots for business presentations

Mathematical Visualization:
- Bootstrap prediction distributions
- Confidence interval bands (90% coverage typical)
- Performance metric evolution over time
- Model validation and diagnostic plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings

# Suppress specific matplotlib/seaborn warnings
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='matplotlib'
)
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='seaborn'
)
# Justification: matplotlib UserWarnings for tight_layout adjustments (cosmetic);
# seaborn FutureWarnings for API changes. Will be addressed during library upgrades.

# Set consistent plotting theme
sns.set_theme(style="whitegrid", palette="deep")


def _format_forecast_axes(axes: plt.Axes, title: str) -> None:
    """Apply standard forecast plot formatting."""
    axes.grid(True, alpha=0.3)
    axes.set_title(title, fontsize=16, pad=20)
    axes.set_xlabel("Forecast Date", fontsize=12)
    axes.set_ylabel("FlexGuard Sales", fontsize=12)
    axes.legend(fontsize=11, loc='upper left')
    plt.setp(axes.get_xticklabels(), rotation=45, ha='right')


def create_bootstrap_forecast_plot_atomic(dates: np.ndarray,
                                        y_true: np.ndarray,
                                        bootstrap_data: pd.DataFrame,
                                        title: str = "Bootstrap Forecast with Confidence Intervals",
                                        figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
    """Create bootstrap forecast plot with confidence intervals atomically."""
    figure, axes = plt.subplots(1, 1, figsize=figsize)

    dates_pd = pd.to_datetime(dates)
    axes.scatter(dates_pd, y_true, color="black", s=30, alpha=0.7, label="Actual Sales", zorder=3)

    sns.lineplot(data=bootstrap_data, x="date", y="y_bootstrap", ax=axes,
                 color="tab:blue", alpha=0.7, errorbar=("pi", 95), label="Bootstrap Forecast (95% CI)")

    _format_forecast_axes(axes, title)
    figure.tight_layout()
    return figure


def _plot_scatter_reference(axes: plt.Axes,
                           dates: np.ndarray,
                           y_true: np.ndarray,
                           label: str = "Actual Sales") -> None:
    """
    Plot scatter reference points for true values.

    Parameters
    ----------
    axes : plt.Axes
        Matplotlib axes to plot on
    dates : np.ndarray
        Dates for x-axis
    y_true : np.ndarray
        True values for y-axis
    label : str, default="Actual Sales"
        Label for legend
    """
    dates_pd = pd.to_datetime(dates)
    axes.scatter(dates_pd, y_true, color="black", s=30, alpha=0.8,
                 label=label, zorder=3)


def _plot_bootstrap_comparison(axes: plt.Axes,
                               combined_bootstrap_data: pd.DataFrame) -> None:
    """
    Plot bootstrap confidence intervals for model comparison.

    Parameters
    ----------
    axes : plt.Axes
        Matplotlib axes to plot on
    combined_bootstrap_data : pd.DataFrame
        Combined data with Model and Benchmark outputs
    """
    sns.lineplot(
        data=combined_bootstrap_data,
        x="date",
        y="y_bootstrap",
        hue="output",
        ax=axes,
        errorbar=("pi", 95),
        estimator="mean",
        alpha=0.7
    )


def _format_comparison_legend(axes: plt.Axes) -> None:
    """
    Format legend with business-friendly labels.

    Parameters
    ----------
    axes : plt.Axes
        Matplotlib axes with legend to format
    """
    handles, labels = axes.get_legend_handles_labels()
    label_mapping = {
        "Model": "Bootstrap Ridge Model",
        "Benchmark": "Rolling Average Benchmark"
    }
    business_labels = [label_mapping.get(label, label) for label in labels]
    axes.legend(handles, business_labels, fontsize=11, loc='upper left')


def create_model_benchmark_comparison_atomic(dates: np.ndarray,
                                           y_true: np.ndarray,
                                           combined_bootstrap_data: pd.DataFrame,
                                           title: str = "Model vs Benchmark Comparison",
                                           figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
    """
    Create model vs benchmark comparison plot atomically.

    Atomic Responsibility: Model comparison visualization only.
    Business Format: Comparative analysis suitable for decision-making.

    Parameters
    ----------
    dates : np.ndarray
        Forecast dates
    y_true : np.ndarray
        True values for reference
    combined_bootstrap_data : pd.DataFrame
        Combined data with Model and Benchmark outputs
    title : str, default="Model vs Benchmark Comparison"
        Plot title
    figsize : Tuple[int, int], default=(16, 6)
        Figure size

    Returns
    -------
    plt.Figure
        Model vs benchmark comparison plot

    Plot Elements
    ------------
    - True values: Reference scatter points (black)
    - Model forecasts: Bootstrap confidence intervals (blue)
    - Benchmark forecasts: Baseline confidence intervals (orange)
    - Legend: Clear differentiation between model types
    """
    figure, axes = plt.subplots(1, 1, figsize=figsize)

    _plot_scatter_reference(axes, dates, y_true)
    _plot_bootstrap_comparison(axes, combined_bootstrap_data)

    # Professional formatting
    axes.grid(True, alpha=0.3)
    axes.set_title(title, fontsize=16, pad=20)
    axes.set_xlabel("Forecast Date", fontsize=12)
    axes.set_ylabel("FlexGuard Sales", fontsize=12)

    _format_comparison_legend(axes)
    plt.setp(axes.get_xticklabels(), rotation=45, ha='right')
    figure.tight_layout()

    return figure


def _plot_mape_line(axes: plt.Axes, forecast_df: pd.DataFrame, col: str, color: str, label: str) -> None:
    """Plot a single MAPE line if column exists."""
    if col in forecast_df.columns:
        sns.lineplot(data=forecast_df, x="date", y=col, ax=axes, linewidth=3, color=color, label=label)


def create_mape_analysis_plot_atomic(forecast_df: pd.DataFrame,
                                   title: str = "MAPE Analysis Over Time",
                                   figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
    """Create MAPE analysis plot atomically - error evolution visualization."""
    figure, axes = plt.subplots(1, 1, figsize=figsize)

    _plot_mape_line(axes, forecast_df, '13_week_MAPE_model', "tab:blue", "Bootstrap Ridge Model (13-Week Average)")
    _plot_mape_line(axes, forecast_df, '13_week_MAPE_benchmark', "tab:orange", "Rolling Average Benchmark (13-Week Average)")

    axes.grid(True, alpha=0.3)
    axes.set_title(title, fontsize=16, pad=20)
    axes.set_xlabel("Forecast Date", fontsize=12)
    axes.set_ylabel("Mean Absolute Percentage Error (MAPE %)", fontsize=12)
    axes.legend(fontsize=11)
    plt.setp(axes.get_xticklabels(), rotation=45, ha='right')
    figure.tight_layout()

    return figure


def _plot_forecast_panel(axes: plt.Axes,
                        forecast_df: pd.DataFrame,
                        combined_bootstrap_data: pd.DataFrame) -> None:
    """
    Plot forecast panel with actual values and bootstrap confidence intervals.

    Parameters
    ----------
    axes : plt.Axes
        Matplotlib axes for plotting
    forecast_df : pd.DataFrame
        Complete forecast results with y_true column
    combined_bootstrap_data : pd.DataFrame
        Bootstrap data for confidence intervals
    """
    y_true = forecast_df['y_true'].values if 'y_true' in forecast_df.columns else []

    if len(y_true) > 0:
        axes.scatter(forecast_df['date'], y_true, color="black", s=20, alpha=0.7,
                     label="Actual Sales", zorder=3)

    sns.lineplot(
        data=combined_bootstrap_data,
        x="date",
        y="y_bootstrap",
        hue="output",
        ax=axes,
        errorbar=("pi", 95),
        estimator="mean",
        alpha=0.7
    )

    axes.grid(True, alpha=0.3)
    axes.set_title("FlexGuard 6Y20 Forecasting Results with Confidence Intervals", fontsize=14)
    axes.set_ylabel("FlexGuard Sales", fontsize=12)
    axes.legend(fontsize=11)


def _plot_mape_panel(axes: plt.Axes, forecast_df: pd.DataFrame) -> None:
    """
    Plot MAPE evolution panel for model and benchmark.

    Parameters
    ----------
    axes : plt.Axes
        Matplotlib axes for plotting
    forecast_df : pd.DataFrame
        Forecast data with MAPE columns
    """
    if '13_week_MAPE_model' in forecast_df.columns:
        sns.lineplot(
            data=forecast_df,
            x="date",
            y="13_week_MAPE_model",
            ax=axes,
            linewidth=4,
            color="tab:blue",
            label="Bootstrap Ridge (13-Week MAPE)"
        )

    if '13_week_MAPE_benchmark' in forecast_df.columns:
        sns.lineplot(
            data=forecast_df,
            x="date",
            y="13_week_MAPE_benchmark",
            ax=axes,
            linewidth=4,
            color="tab:orange",
            label="Benchmark (13-Week MAPE)"
        )

    axes.grid(True, alpha=0.3)
    axes.set_title("13-Week Rolling Mean Absolute Percentage Error (MAPE)", fontsize=14)
    axes.set_xlabel("Forecast Date", fontsize=12)
    axes.set_ylabel("MAPE (%)", fontsize=12)
    axes.legend(fontsize=11)


def create_comprehensive_analysis_plot_atomic(forecast_df: pd.DataFrame,
                                            combined_bootstrap_data: pd.DataFrame,
                                            title: str = "Comprehensive Forecasting Analysis",
                                            figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
    """
    Create comprehensive analysis plot atomically - multi-panel dashboard.

    Atomic Responsibility: Multi-panel analysis visualization.
    Business Format: Executive dashboard suitable for comprehensive review.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Complete forecast results
    combined_bootstrap_data : pd.DataFrame
        Bootstrap data for confidence intervals
    title : str, default="Comprehensive Forecasting Analysis"
        Main title for dashboard
    figsize : Tuple[int, int], default=(20, 12)
        Figure size for dashboard

    Returns
    -------
    plt.Figure
        Comprehensive multi-panel analysis plot

    Dashboard Panels
    ---------------
    - Top Panel: Forecast results with confidence intervals
    - Bottom Panel: MAPE analysis over time
    """
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)

    _plot_forecast_panel(axes[0], forecast_df, combined_bootstrap_data)
    _plot_mape_panel(axes[1], forecast_df)

    figure.suptitle(title, fontsize=18, y=0.98)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    figure.tight_layout()

    return figure


def _extract_volatility_metrics(volatility_metrics: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract R2 and MAPE metrics from volatility metrics dictionary.

    Parameters
    ----------
    volatility_metrics : Dict[str, float]
        Raw volatility-weighted performance metrics

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float]]
        Tuple of (r2_data, mape_data) dictionaries for plotting
    """
    r2_data = {
        'Standard Weighting': volatility_metrics.get('model_r2_standard', 0),
        'Volatility Weighting': volatility_metrics.get('model_r2_weighted', 0)
    }
    mape_data = {
        'Standard Weighting': volatility_metrics.get('model_mape_standard', 0),
        'Volatility Weighting': volatility_metrics.get('model_mape_weighted', 0)
    }
    return r2_data, mape_data


def _plot_metric_bars_with_labels(axes: plt.Axes,
                                  data: Dict[str, float],
                                  title: str,
                                  ylabel: str,
                                  value_format: str,
                                  label_offset: float) -> None:
    """
    Plot bar chart with value labels for metric comparison.

    Parameters
    ----------
    axes : plt.Axes
        Matplotlib axes for plotting
    data : Dict[str, float]
        Data dictionary with category names as keys and values as values
    title : str
        Panel title
    ylabel : str
        Y-axis label
    value_format : str
        Format string for value labels (e.g., '{:.4f}' or '{:.2f}%')
    label_offset : float
        Vertical offset for value labels above bars
    """
    bars = axes.bar(data.keys(), data.values(),
                    color=['tab:blue', 'tab:orange'], alpha=0.7)
    axes.set_title(title, fontsize=12)
    axes.set_ylabel(ylabel, fontsize=11)
    axes.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, data.values()):
        axes.text(bar.get_x() + bar.get_width()/2., bar.get_height() + label_offset,
                  value_format.format(value), ha='center', va='bottom', fontsize=10)


def create_volatility_analysis_plot_atomic(volatility_metrics: Dict[str, float],
                                         forecast_df: pd.DataFrame,
                                         title: str = "Volatility-Weighted Performance Analysis",
                                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create volatility analysis plot atomically - performance across market conditions.

    Atomic Responsibility: Volatility analysis visualization only.
    Business Format: Market condition performance analysis.

    Parameters
    ----------
    volatility_metrics : Dict[str, float]
        Volatility-weighted performance metrics
    forecast_df : pd.DataFrame
        Forecast data with volatility information
    title : str, default="Volatility-Weighted Performance Analysis"
        Plot title
    figsize : Tuple[int, int], default=(12, 8)
        Figure size

    Returns
    -------
    plt.Figure
        Volatility analysis plot

    Plot Elements
    ------------
    - Performance comparison: Standard vs volatility-weighted metrics
    - Bar chart: Clear comparison of R2 and MAPE across conditions
    """
    figure, axes = plt.subplots(2, 1, figsize=figsize)

    r2_data, mape_data = _extract_volatility_metrics(volatility_metrics)

    _plot_metric_bars_with_labels(
        axes[0], r2_data,
        title='R2 Performance: Standard vs Volatility Weighting',
        ylabel='R2 Score',
        value_format='{:.4f}',
        label_offset=0.01
    )

    _plot_metric_bars_with_labels(
        axes[1], mape_data,
        title='MAPE Performance: Standard vs Volatility Weighting',
        ylabel='MAPE (%)',
        value_format='{:.2f}%',
        label_offset=0.2
    )

    figure.suptitle(title, fontsize=14, y=0.98)
    figure.tight_layout()

    return figure


def _plot_comparison_panel(ax: plt.Axes, model_val: float, bench_val: float,
                          panel_title: str, ylabel: str, fmt: str, offset: float) -> None:
    """Plot a model vs benchmark comparison bar panel."""
    labels = ['Bootstrap Ridge\nModel', 'Rolling Average\nBenchmark']
    bars = ax.bar(labels, [model_val, bench_val], color=['tab:blue', 'tab:orange'], alpha=0.7)
    ax.set_title(panel_title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, [model_val, bench_val]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + offset, fmt.format(val),
                ha='center', va='bottom', fontsize=10)


def create_performance_summary_plot_atomic(performance_summary: Dict[str, Any],
                                         title: str = "Model Performance Summary",
                                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Create performance summary plot atomically - executive summary visualization."""
    figure, axes = plt.subplots(1, 2, figsize=figsize)

    _plot_comparison_panel(axes[0], performance_summary.get('model_r2', 0),
                          performance_summary.get('benchmark_r2', 0),
                          'R² Score Comparison', 'R² Score', '{:.4f}', 0.01)

    _plot_comparison_panel(axes[1], performance_summary.get('model_mape', 0),
                          performance_summary.get('benchmark_mape', 0),
                          'MAPE Comparison', 'MAPE (%)', '{:.2f}%', 0.3)

    figure.suptitle(title, fontsize=14, y=0.98)
    figure.tight_layout()
    return figure