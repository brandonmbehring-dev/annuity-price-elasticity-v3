#!/usr/bin/env python3
"""
RILA Price Elasticity Inference Visualization Module

Provides pixel-perfect replication of original notebook visualizations
with configuration-driven architecture for executive business reporting.

This module maintains mathematical equivalence with the original implementation
while following CODING_STANDARDS.md patterns for professional software engineering.

Key Visualizations:
1. Price Elasticity Confidence Intervals (Percentage Change)
2. Price Elasticity Confidence Intervals (Dollar Impact)

Mathematical Equivalence: All outputs must match original notebook with 1e-12 tolerance
Visual Equivalence: PNG outputs must be pixel-perfect identical to original

Author: Refactored from notebooks/01_RUNME_PE_RILA_SHORT.ipynb
Date: 2025-11-20
"""

# Core dependencies - direct imports (no defensive handling for standard/required packages)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Standard library imports
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from datetime import datetime
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


def _melt_bootstrap_data(df_bootstrap_raw: pd.DataFrame, output_type: str, current_date: str) -> pd.DataFrame:
    """Melt bootstrap data with standardized transformations."""
    df_copy = df_bootstrap_raw.copy()
    df_copy["output_type"] = output_type
    df_copy = df_copy.reset_index().rename(columns={"index": "simulation_run"})

    df_bootstrap = df_copy.melt(id_vars=["simulation_run", "output_type"])
    df_bootstrap["variable"] = 100 * df_bootstrap["variable"]
    df_bootstrap = df_bootstrap.rename(columns={"variable": "rate_change_in_basis_points"})
    df_bootstrap["prediction_date"] = current_date

    return df_bootstrap


def _create_categorical_labels(df_data: pd.DataFrame) -> List[str]:
    """Generate x_labels with exact formatting from original."""
    x_labels_1 = sorted(df_data["rate_change_in_basis_points"].unique())
    return [str(int(np.round(x))) + "bps" for x in x_labels_1]


def _apply_categorical_ordering(df: pd.DataFrame, x_labels: List[str], column_name: str = "rate_change_in_basis_points") -> pd.DataFrame:
    """Apply categorical ordering to rate change columns."""
    df = df.copy()
    df[column_name] = df[column_name].astype("string") + "bps"
    df[column_name] = pd.Categorical(df[column_name], x_labels)
    return df


def _prepare_confidence_intervals(df_output: pd.DataFrame, x_labels: List[str], output_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare confidence interval data structures."""
    # Create df_ci for melted confidence intervals
    df_ci_original = df_output.copy()
    df_ci_original["rate_change_in_basis_points"] = (df_ci_original["rate_change_in_basis_points"]/100).round(2)

    df_num = df_ci_original.melt(id_vars=["rate_change_in_basis_points"]).rename(
        columns={"variable": "range", "value": "dollar"}  # Keep original column name for consistency
    )

    df_ci = df_num.copy()
    df_ci["value"] = df_ci["dollar"]
    df_ci["output_type"] = output_type
    df_ci["rate_change_in_basis_points"] = (
        df_ci["rate_change_in_basis_points"] * 100
    ).astype("int").astype("string") + "bps"
    df_ci["rate_change_in_basis_points"] = pd.Categorical(df_ci["rate_change_in_basis_points"], x_labels)

    # Create df_ci_columns
    df_ci_columns = df_output.copy()
    df_ci_columns[["bottom", "median", "top"]] = df_ci_columns[["bottom", "median", "top"]]
    df_ci_columns["rate_change_in_basis_points"] = (
        df_ci_columns["rate_change_in_basis_points"].astype("int").astype("string") + "bps"
    )
    df_ci_columns["rate_change_in_basis_points"] = pd.Categorical(df_ci_columns["rate_change_in_basis_points"], x_labels)

    return df_ci, df_ci_columns


def prepare_visualization_data_pct(
    df_pct_change: pd.DataFrame,
    df_output_pct: pd.DataFrame,
    rate_options: np.ndarray,
    current_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare data structures for percentage change visualization with pixel-perfect equivalence.

    Parameters
    ----------
    df_pct_change : pd.DataFrame
        Bootstrap percentage change results (1000 x 19)
    df_output_pct : pd.DataFrame
        Confidence interval results for percentage changes
    rate_options : np.ndarray
        Rate scenario array (0.0 to 4.5)
    current_date : str
        Current date for file naming

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]
        (df_dist, df_ci, df_ci_columns, x_labels) - Prepared data for visualization
    """
    # Melt bootstrap distributions
    df_bootstrap = _melt_bootstrap_data(df_pct_change, "pct_change", current_date)

    # Create df_dist with categorical ordering
    df_dist = df_bootstrap.copy()
    df_dist["rate_change_in_basis_points"] = pd.to_numeric(df_dist["rate_change_in_basis_points"], errors='coerce').round().astype("int")
    df_dist = df_dist[df_dist["output_type"] == "pct_change"]
    df_dist["value"] = df_dist["value"]

    # Generate labels and apply categorical ordering
    x_labels = _create_categorical_labels(df_dist)
    df_dist = _apply_categorical_ordering(df_dist, x_labels)

    # Prepare confidence intervals
    df_ci, df_ci_columns = _prepare_confidence_intervals(df_output_pct, x_labels, "pct_change")

    return df_dist, df_ci, df_ci_columns, x_labels


def prepare_visualization_data_dollars(
    df_dollars: pd.DataFrame,
    df_output_dollar: pd.DataFrame,
    rate_options: np.ndarray,
    current_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare data structures for dollar impact visualization with pixel-perfect equivalence.

    Parameters
    ----------
    df_dollars : pd.DataFrame
        Bootstrap dollar impact results (1000 x 19)
    df_output_dollar : pd.DataFrame
        Confidence interval results for dollar impacts
    rate_options : np.ndarray
        Rate scenario array (0.0 to 4.5)
    current_date : str
        Current date for file naming

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]
        (df_dist, df_ci, df_ci_columns, x_labels) - Prepared data for visualization
    """
    # Melt bootstrap distributions
    df_bootstrap = _melt_bootstrap_data(df_dollars, "dollars", current_date)

    # Create df_dist with categorical ordering
    df_dist = df_bootstrap.copy()
    df_dist["rate_change_in_basis_points"] = pd.to_numeric(df_dist["rate_change_in_basis_points"], errors='coerce').round().astype("int")
    df_dist = df_dist[df_dist["output_type"] == "dollars"]
    df_dist["value"] = df_dist["value"]

    # Generate labels (note: different formatting for dollars - no int() wrapper)
    x_labels_1 = sorted(df_dist["rate_change_in_basis_points"].unique())
    x_labels = [str(x) + "bps" for x in x_labels_1]
    df_dist = _apply_categorical_ordering(df_dist, x_labels)

    # Prepare confidence intervals
    df_ci, df_ci_columns = _prepare_confidence_intervals(df_output_dollar, x_labels, "dollar")

    return df_dist, df_ci, df_ci_columns, x_labels


def _setup_figure_and_theme(viz_config: Dict[str, Any]) -> Tuple[plt.Figure, plt.Axes]:
    """Setup figure dimensions and seaborn theme."""
    sns.set_theme(style=viz_config['seaborn_style'], palette=viz_config['seaborn_palette'])
    figure, axes = plt.subplots(1, 1, sharex=True, sharey=False, figsize=viz_config['figure_size'])
    return figure, axes


def _extract_rate_context(rate_context: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Extract and round rate context values."""
    P = np.round(rate_context['prudential_current'], 2)
    P_lag = np.round(rate_context['prudential_lag'], 2)
    C = np.round(rate_context['competitor_current'], 2)
    C_lag = np.round(rate_context['competitor_lag'], 2)
    return P, P_lag, C, C_lag


def _create_violin_and_scatter_plots(df_dist: pd.DataFrame, df_ci: pd.DataFrame, axes: plt.Axes, viz_config: Dict[str, Any]) -> None:
    """Create violin and scatter plots on the axes."""
    sns.violinplot(
        df_dist,
        y="rate_change_in_basis_points",
        x="value",
        hue=viz_config['violin_params']['hue'],
        hue_order=viz_config['violin_params']['hue_order'],
        split=viz_config['violin_params']['split'],
        dodge=viz_config['violin_params']['dodge'],
        density_norm=viz_config['violin_params']['density_norm'],
        ax=axes,
        inner=viz_config['violin_params']['inner'],
        legend=viz_config['violin_params']['legend'],
    )

    sns.scatterplot(df_ci, y="rate_change_in_basis_points", x="value",
                   color=viz_config['line_colors']['scatter'], ax=axes)


def _add_confidence_interval_lines(df_ci_columns: pd.DataFrame, axes: plt.Axes, viz_config: Dict[str, Any]) -> None:
    """Add confidence interval lines to the plot."""
    sns.lineplot(df_ci_columns, y="rate_change_in_basis_points", x="bottom",
                ax=axes, color=viz_config['line_colors']['ci_bounds'], orient="y")
    sns.lineplot(df_ci_columns, y="rate_change_in_basis_points", x="median",
                ax=axes, color=viz_config['line_colors']['median'], orient="y")
    sns.lineplot(df_ci_columns, y="rate_change_in_basis_points", x="top",
                ax=axes, color=viz_config['line_colors']['ci_bounds'], orient="y")


def _set_title_and_labels(axes: plt.Axes, rate_context_values: Tuple[float, float, float, float],
                         current_date: str, x_labels: List[str], xlabel: str) -> None:
    """Set title, axis labels and ticks."""
    P, P_lag, C, C_lag = rate_context_values
    axes.set_title(
        f"FlexGuard \n Price Elasticity for {current_date} \n Current Prudential Cap Rate: {P:.2f} prior {P_lag:.2f}\nCurrent synethetic competitor {C:.2f} prior {C_lag:.2f}"
    )
    axes.set_yticks(ticks=x_labels, labels=x_labels)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Rate Action in Basis Points")


def _add_percentage_annotations(df_ci_columns: pd.DataFrame, viz_config: Dict[str, Any]) -> None:
    """Add percentage annotations to confidence interval points."""
    xs = df_ci_columns["bottom"]
    ys = df_ci_columns["rate_change_in_basis_points"]

    # Bottom annotations
    for x, y in zip(xs, ys):
        label = f"{int(np.round(x)):d}%"
        plt.annotate(label, (x, y), textcoords="offset points",
                    xytext=viz_config['annotation_offsets']['pct_bottom'], ha="right")

    # Median annotations
    xs = df_ci_columns["median"]
    for x, y in zip(xs, ys):
        label = f"{np.round(x, 1):.1f}%"
        plt.annotate(label, (x, y), textcoords="offset points",
                    xytext=viz_config['annotation_offsets']['pct_median'], ha="center")

    # Top annotations
    xs = df_ci_columns["top"]
    for x, y in zip(xs, ys):
        label = f"{int(np.round(x)):d}%"
        plt.annotate(label, (x, y), textcoords="offset points",
                    xytext=viz_config['annotation_offsets']['pct_top'], ha="left")


def generate_price_elasticity_visualization_pct(
    df_dist: pd.DataFrame,
    df_ci: pd.DataFrame,
    df_ci_columns: pd.DataFrame,
    x_labels: List[str],
    rate_context: Dict[str, float],
    current_date: str,
    viz_config: Dict[str, Any]
) -> plt.Figure:
    """
    Generate percentage change price elasticity visualization with pixel-perfect equivalence.

    Parameters
    ----------
    df_dist : pd.DataFrame
        Bootstrap distribution data prepared for plotting
    df_ci : pd.DataFrame
        Confidence interval data for scatter plot
    df_ci_columns : pd.DataFrame
        Confidence interval columns for line plots and annotations
    x_labels : List[str]
        Category labels for y-axis (e.g., ['0bps', '25bps', ...])
    rate_context : Dict[str, float]
        Current rate context for title generation
    current_date : str
        Current date for title
    viz_config : Dict[str, Any]
        Visualization configuration parameters

    Returns
    -------
    plt.Figure
        Generated matplotlib figure with pixel-perfect equivalence to original
    """
    figure, axes = _setup_figure_and_theme(viz_config)
    rate_context_values = _extract_rate_context(rate_context)

    _create_violin_and_scatter_plots(df_dist, df_ci, axes, viz_config)
    _add_confidence_interval_lines(df_ci_columns, axes, viz_config)
    _set_title_and_labels(axes, rate_context_values, current_date, x_labels, "Percent Change in Sales")
    _add_percentage_annotations(df_ci_columns, viz_config)

    axes.grid()
    return figure


def _add_dollar_annotations(df_ci_columns: pd.DataFrame, viz_config: Dict[str, Any]) -> None:
    """Add dollar annotations to confidence interval points."""
    xs = df_ci_columns["bottom"]
    ys = df_ci_columns["rate_change_in_basis_points"]

    # Bottom annotations
    for x, y in zip(xs, ys):
        label = f"{int(np.round(x/1e6)):d}M"
        plt.annotate(label, (x, y), textcoords="offset points",
                    xytext=viz_config['annotation_offsets']['dollar_bottom'], ha="right")

    # Median annotations
    xs = df_ci_columns["median"]
    for x, y in zip(xs, ys):
        label = f"{int(np.round(x/1e6)):d}M"
        plt.annotate(label, (x, y), textcoords="offset points",
                    xytext=viz_config['annotation_offsets']['dollar_median'], ha="center")

    # Top annotations
    xs = df_ci_columns["top"]
    for x, y in zip(xs, ys):
        label = f"{int(np.round(x/1e6)):d}M"
        plt.annotate(label, (x, y), textcoords="offset points",
                    xytext=viz_config['annotation_offsets']['dollar_top'], ha="left")


def generate_price_elasticity_visualization_dollars(
    df_dist: pd.DataFrame,
    df_ci: pd.DataFrame,
    df_ci_columns: pd.DataFrame,
    x_labels: List[str],
    rate_context: Dict[str, float],
    current_date: str,
    viz_config: Dict[str, Any]
) -> plt.Figure:
    """
    Generate dollar impact price elasticity visualization with pixel-perfect equivalence.

    Parameters
    ----------
    df_dist : pd.DataFrame
        Bootstrap distribution data prepared for plotting
    df_ci : pd.DataFrame
        Confidence interval data for scatter plot
    df_ci_columns : pd.DataFrame
        Confidence interval columns for line plots and annotations
    x_labels : List[str]
        Category labels for y-axis
    rate_context : Dict[str, float]
        Current rate context for title generation
    current_date : str
        Current date for title
    viz_config : Dict[str, Any]
        Visualization configuration parameters

    Returns
    -------
    plt.Figure
        Generated matplotlib figure for dollar impact analysis
    """
    figure, axes = _setup_figure_and_theme(viz_config)
    rate_context_values = _extract_rate_context(rate_context)

    _create_violin_and_scatter_plots(df_dist, df_ci, axes, viz_config)
    _add_confidence_interval_lines(df_ci_columns, axes, viz_config)
    _set_title_and_labels(axes, rate_context_values, current_date, x_labels, "Sales Run")
    _add_dollar_annotations(df_ci_columns, viz_config)

    axes.grid()
    return figure


def save_visualization_files(
    figure_pct: plt.Figure,
    figure_dollar: plt.Figure,
    output_dir: Union[str, Path],
    current_date: str,
    file_prefix: str = "price_elasticity_FlexGuard"
) -> Dict[str, Path]:
    """
    Save visualization PNG files exactly like original notebook.

    Parameters
    ----------
    figure_pct : plt.Figure
        Percentage change visualization figure
    figure_dollar : plt.Figure
        Dollar impact visualization figure
    output_dir : Union[str, Path]
        Output directory for PNG files
    current_date : str
        Current date for filename
    file_prefix : str
        File prefix for naming

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping visualization type to saved file path

    Business Purpose
    ---------------
    Creates presentation-ready PNG files for executive reporting and regulatory documentation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save files with exact naming convention from original
    pct_filename = f"{file_prefix}_Sample_{current_date}.png"
    dollar_filename = f"{file_prefix}_Dollars_Sample_{current_date}.png"

    pct_path = output_path / pct_filename
    dollar_path = output_path / dollar_filename

    # Save with high quality for presentations
    figure_pct.savefig(pct_path, dpi=300, bbox_inches='tight')
    figure_dollar.savefig(dollar_path, dpi=300, bbox_inches='tight')

    return {
        'percentage': pct_path,
        'dollar': dollar_path
    }


def _prepare_bootstrap_exports(df_pct_change: pd.DataFrame, df_dollars: pd.DataFrame,
                             current_time: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare bootstrap data for export."""
    df_pct_change_export = df_pct_change.copy()
    df_pct_change_export["output_type"] = "pct_change"
    df_pct_change_export = df_pct_change_export.reset_index().rename(columns={"index": "simulation_run"})

    df_dollars_export = df_dollars.copy()
    df_dollars_export["output_type"] = "dollars"
    df_dollars_export = df_dollars_export.reset_index().rename(columns={"index": "simulation_run"})

    return df_pct_change_export, df_dollars_export


def _melt_bootstrap_for_export(df_bootstrap_export: pd.DataFrame, prediction_date: Union[str, datetime]) -> pd.DataFrame:
    """Melt bootstrap data with prediction date."""
    df_bootstrap = df_bootstrap_export.melt(id_vars=["simulation_run", "output_type"])
    df_bootstrap["variable"] = 100 * df_bootstrap["variable"]
    df_bootstrap = df_bootstrap.rename(columns={"variable": "rate_change_in_basis_points"})
    df_bootstrap["prediction_date"] = prediction_date
    return df_bootstrap


def _export_basic_csv_files(output_path: Path, df_pct_change: pd.DataFrame, df_pct_change_export: pd.DataFrame,
                          df_dollars_export: pd.DataFrame, current_date: str, current_time: datetime) -> Dict[str, Path]:
    """Export basic CSV files (1-4)."""
    saved_files = {}

    # 1. Weekly raw bootstrap
    file_path = output_path / f"weekly_raw_bootstrap_{current_date}.csv"
    df_pct_change.to_csv(file_path)
    saved_files['weekly_raw_bootstrap'] = file_path

    # 2. Bootstrap distributions (combined)
    dr = pd.concat([df_pct_change_export, df_dollars_export])
    dr["prediction_date"] = current_time
    file_path = output_path / f"price_elasticity_FlexGuard_bootstrap_distributions_{current_date}.csv"
    dr.to_csv(file_path)
    saved_files['bootstrap_distributions'] = file_path

    # 3. Bootstrap distributions melt (dollars)
    df_bootstrap = _melt_bootstrap_for_export(df_dollars_export, current_time)
    file_path = output_path / f"price_elasticity_FlexGuard_bootstrap_distributions_melt_dollars_{current_date}.csv"
    df_bootstrap.to_csv(file_path)
    saved_files['bootstrap_melt_dollars'] = file_path

    # 4. Bootstrap distributions melt (percentage)
    df_bootstrap = _melt_bootstrap_for_export(df_pct_change_export, current_date)
    file_path = output_path / f"price_elasticity_FlexGuard_bootstrap_distributions_melt_{current_date}.csv"
    df_bootstrap.to_csv(file_path)
    saved_files['bootstrap_melt_pct'] = file_path

    return saved_files


def _export_confidence_interval_csv_files(output_path: Path, df_output_pct: pd.DataFrame, df_output_dollar: pd.DataFrame,
                                        df_to_bi_melt: pd.DataFrame, current_date: str) -> Dict[str, Path]:
    """Export confidence interval CSV files (5-8)."""
    saved_files = {}

    # 5. Simple percentage confidence intervals
    df_output_pct_export = df_output_pct.copy()
    df_output_pct_export["prediction_date"] = current_date
    file_path = output_path / f"sample_price_elasticity_FlexGuard_output_simple_pct_change_confidence_intervals_{current_date}.csv"
    df_output_pct_export.to_csv(file_path)
    saved_files['simple_pct_ci'] = file_path

    # 6. Simple dollar confidence intervals
    df_output_dollar_export = df_output_dollar.copy()
    df_output_dollar_export["prediction_date"] = current_date
    file_path = output_path / f"sample_price_elasticity_FlexGuard_output_simple_amount_in_dollars_confidence_intervals_{current_date}.csv"
    df_output_dollar_export.to_csv(file_path)
    saved_files['simple_dollar_ci'] = file_path

    # 7. Combined confidence intervals
    df_num = df_output_dollar.melt(id_vars=["rate_change_in_basis_points"]).rename(
        columns={"variable": "range", "value": "dollar"}
    )
    df_pct = df_output_pct.melt(id_vars=["rate_change_in_basis_points"]).rename(
        columns={"variable": "range", "value": "pct_change"}
    )
    df_to_bi = df_num.merge(
        df_pct[["rate_change_in_basis_points", "range", "pct_change"]],
        on=["rate_change_in_basis_points", "range"],
    )
    file_path = output_path / f"price_elasticity_FlexGuard_confidence_intervals_{current_date}.csv"
    df_to_bi.to_csv(file_path)
    saved_files['combined_ci'] = file_path

    # 8. Final BI export (melted)
    file_path = output_path / f"price_elasticity_FlexGuard_confidence_intervals_melt_{current_date}.csv"
    df_to_bi_melt.to_csv(file_path)
    saved_files['bi_export_melt'] = file_path

    return saved_files


def export_csv_files(
    df_pct_change: pd.DataFrame,
    df_dollars: pd.DataFrame,
    df_output_pct: pd.DataFrame,
    df_output_dollar: pd.DataFrame,
    df_to_bi_melt: pd.DataFrame,
    output_dir: Union[str, Path],
    current_date: str,
    current_time: datetime
) -> Dict[str, Path]:
    """Export all CSV files for BI team consumption with bootstrap and confidence interval data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare bootstrap exports
    df_pct_change_export, df_dollars_export = _prepare_bootstrap_exports(df_pct_change, df_dollars, current_time)

    # Export basic files (1-4)
    saved_files = _export_basic_csv_files(
        output_path, df_pct_change, df_pct_change_export, df_dollars_export, current_date, current_time
    )

    # Export confidence interval files (5-8)
    ci_files = _export_confidence_interval_csv_files(
        output_path, df_output_pct, df_output_dollar, df_to_bi_melt, current_date
    )

    saved_files.update(ci_files)
    return saved_files