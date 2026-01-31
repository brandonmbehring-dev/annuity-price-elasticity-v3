"""
Bootstrap Visualization Analysis Engine for Feature Selection.

Enhanced module for comprehensive bootstrap visualization with integrated
statistical analysis and business insights generation.

Key Functions:
- run_bootstrap_visualization_analysis: Main atomic function for visualization creation
- prepare_bootstrap_visualization_data: Data preparation for plotting
- create_aic_distribution_visualizations: AIC distribution plots and analysis
- create_stability_comparison_visualizations: Model stability comparison charts
- generate_visualization_insights: Business insights from visualization data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Defensive imports for bootstrap types
try:
    from src.features.selection_types import BootstrapResult
    BOOTSTRAP_TYPES_AVAILABLE = True
except ImportError:
    # Fallback for standalone operation
    from typing import TypedDict
    from dataclasses import dataclass
    BOOTSTRAP_TYPES_AVAILABLE = False

    @dataclass
    class BootstrapResult:
        """Fallback BootstrapResult when types module unavailable."""
        model_name: str
        model_features: str
        bootstrap_aics: List[float]
        bootstrap_r2_values: List[float]
        original_aic: float
        original_r2: float
        aic_stability_coefficient: float
        r2_stability_coefficient: float
        confidence_intervals: Dict[str, Dict[str, float]]
        successful_fits: int
        total_attempts: int
        stability_assessment: str


def prepare_bootstrap_visualization_data(bootstrap_results: List[BootstrapResult],
                                       max_models: int = 15) -> pd.DataFrame:
    """
    Prepare bootstrap results data for visualization analysis.

    Single responsibility: Data preparation only.
    Follows UNIFIED_CODING_STANDARDS.md with focused data transformation.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap stability results with AIC and R² distributions
    max_models : int, default 15
        Maximum number of models to include in visualization

    Returns
    -------
    pd.DataFrame
        Prepared visualization data with bootstrap samples in long format
    """
    if not bootstrap_results:
        raise ValueError("No bootstrap results provided for visualization data preparation")

    # Prepare bootstrap data in long format for visualization
    bootstrap_viz_data = []
    n_models = min(len(bootstrap_results), max_models)

    for i in range(n_models):
        result = bootstrap_results[i]
        model_name = f'Model {i+1}'

        # Create long-format data for each bootstrap sample
        for j, aic_val in enumerate(result.bootstrap_aics):
            bootstrap_viz_data.append({
                'model': model_name,
                'model_features': result.model_features,
                'bootstrap_aic': aic_val,
                'bootstrap_r2': result.bootstrap_r2_values[j] if j < len(result.bootstrap_r2_values) else np.nan,
                'original_aic': result.original_aic,
                'original_r2': result.original_r2,
                'stability_assessment': result.stability_assessment,
                'model_index': i
            })

    return pd.DataFrame(bootstrap_viz_data)


def _extract_original_aics(bootstrap_df: pd.DataFrame) -> Tuple[np.ndarray, List[float]]:
    """Extract model names and original AIC values from bootstrap data."""
    model_names = bootstrap_df['model'].unique()
    original_aics = [bootstrap_df[bootstrap_df['model'] == m]['original_aic'].iloc[0] for m in model_names]
    return model_names, original_aics


def _setup_aic_axis(ax: Any, bootstrap_df: pd.DataFrame, title: str, plot_type: str = 'violin') -> None:
    """Configure AIC distribution axis with violin or box plot."""
    if plot_type == 'violin':
        sns.violinplot(data=bootstrap_df, x='model', y='bootstrap_aic', ax=ax)
    else:
        sns.boxplot(data=bootstrap_df, x='model', y='bootstrap_aic', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Models (Top 15 by AIC Performance)')
    ax.set_ylabel('Bootstrap AIC Values')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def create_aic_distribution_visualizations(bootstrap_df: pd.DataFrame,
                                         config: Optional[Dict[str, Any]] = None) -> Dict[str, plt.Figure]:
    """Create AIC distribution visualizations with violin and box plots."""
    if bootstrap_df.empty:
        raise ValueError("Bootstrap visualization data is empty")

    fig_width = config.get('fig_width', 20) if config else 20
    fig_height = config.get('fig_height', 12) if config else 12
    fig_violin, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))

    model_names, original_aics = _extract_original_aics(bootstrap_df)

    _setup_aic_axis(ax1, bootstrap_df, 'Bootstrap AIC Distribution by Model (Violin Plot)', 'violin')
    ax1.scatter(range(len(model_names)), original_aics, color='red', s=100, alpha=0.8, zorder=5, label='Original AIC')
    ax1.legend()

    _setup_aic_axis(ax2, bootstrap_df, 'Bootstrap AIC Distribution by Model (Box Plot)', 'box')
    ax2.scatter(range(len(model_names)), original_aics, color='red', s=100, alpha=0.8, zorder=5, label='Original AIC')
    ax2.legend()

    plt.tight_layout()
    return {'aic_distribution': fig_violin}


def _compute_model_aic_statistics(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    """Compute AIC statistics per model for stability comparison.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Prepared bootstrap visualization data

    Returns
    -------
    pd.DataFrame
        Model-level AIC statistics with mean, std, median, and original values
    """
    model_stats = bootstrap_df.groupby('model').agg({
        'bootstrap_aic': ['mean', 'std', 'median'],
        'original_aic': 'first'
    }).round(2)

    model_stats.columns = ['mean_aic', 'std_aic', 'median_aic', 'original_aic']
    return model_stats.reset_index()


def _create_mean_vs_variability_plot(ax: Any, model_stats: pd.DataFrame) -> None:
    """Create AIC mean vs standard deviation scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object for plotting
    model_stats : pd.DataFrame
        Model-level AIC statistics
    """
    scatter = ax.scatter(model_stats['mean_aic'], model_stats['std_aic'],
                        c=model_stats['original_aic'], cmap='viridis',
                        s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('Bootstrap Mean AIC')
    ax.set_ylabel('Bootstrap Standard Deviation AIC')
    ax.set_title('Model Stability: Mean vs Variability', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add model labels
    for i, row in model_stats.iterrows():
        ax.annotate(row['model'], (row['mean_aic'], row['std_aic']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.colorbar(scatter, ax=ax, label='Original AIC')


def _create_consistency_plot(ax: Any, model_stats: pd.DataFrame) -> None:
    """Create original vs bootstrap median AIC consistency plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object for plotting
    model_stats : pd.DataFrame
        Model-level AIC statistics
    """
    ax.scatter(model_stats['original_aic'], model_stats['median_aic'],
              s=100, alpha=0.7, color='steelblue', edgecolors='black')
    ax.plot([model_stats['original_aic'].min(), model_stats['original_aic'].max()],
            [model_stats['original_aic'].min(), model_stats['original_aic'].max()],
            'r--', alpha=0.8, label='Perfect Agreement')
    ax.set_xlabel('Original AIC')
    ax.set_ylabel('Bootstrap Median AIC')
    ax.set_title('Original vs Bootstrap AIC Consistency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add model labels
    for i, row in model_stats.iterrows():
        ax.annotate(row['model'], (row['original_aic'], row['median_aic']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)


def create_stability_comparison_visualizations(bootstrap_df: pd.DataFrame,
                                             config: Optional[Dict[str, Any]] = None) -> Dict[str, plt.Figure]:
    """
    Create model stability comparison visualizations.

    Single responsibility: Stability comparison plotting only.
    Follows UNIFIED_CODING_STANDARDS.md with focused comparison analysis.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Prepared bootstrap visualization data from prepare_bootstrap_visualization_data
    config : Optional[Dict[str, Any]], default None
        Configuration for figure sizing and styling

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing stability comparison figures
    """
    if bootstrap_df.empty:
        raise ValueError("Bootstrap visualization data is empty for stability comparison")

    # Extract configuration parameters
    fig_width = config.get('fig_width', 20) if config else 20
    fig_height = config.get('fig_height', 10) if config else 10

    visualizations = {}

    # Create figure and compute statistics
    fig_ranking, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    model_stats = _compute_model_aic_statistics(bootstrap_df)

    # Generate comparison plots
    _create_mean_vs_variability_plot(ax1, model_stats)
    _create_consistency_plot(ax2, model_stats)

    plt.tight_layout()
    visualizations['stability_comparison'] = fig_ranking

    return visualizations


def _plot_r2_violin(ax: Any, r2_data: pd.DataFrame) -> None:
    """Create R2 violin plot with original values overlay."""
    sns.violinplot(data=r2_data, x='model', y='bootstrap_r2', ax=ax)
    ax.set_title('Bootstrap R² Distribution by Model', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models')
    ax.set_ylabel('Bootstrap R² Values')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    model_names = r2_data['model'].unique()
    original_r2s = [r2_data[r2_data['model'] == m]['original_r2'].iloc[0] for m in model_names]
    ax.scatter(range(len(model_names)), original_r2s,
               color='red', s=100, alpha=0.8, zorder=5, label='Original R²')
    ax.legend()


def _plot_r2_vs_aic(ax: Any, r2_data: pd.DataFrame) -> None:
    """Create R2 vs AIC scatter plot with correlation annotation."""
    ax.scatter(r2_data['bootstrap_aic'], r2_data['bootstrap_r2'], alpha=0.6, s=30, color='steelblue')
    ax.set_xlabel('Bootstrap AIC')
    ax.set_ylabel('Bootstrap R²')
    ax.set_title('AIC vs R² Relationship', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    correlation = r2_data['bootstrap_aic'].corr(r2_data['bootstrap_r2'])
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def create_r2_distribution_visualizations(bootstrap_df: pd.DataFrame,
                                        config: Optional[Dict[str, Any]] = None) -> Dict[str, plt.Figure]:
    """Create R-squared distribution visualizations for model performance analysis."""
    if bootstrap_df.empty or 'bootstrap_r2' not in bootstrap_df.columns:
        print("WARNING: No R² data available for visualization")
        return {}

    fig_width = config.get('fig_width', 20) if config else 20
    fig_height = config.get('fig_height', 10) if config else 10
    r2_data = bootstrap_df.dropna(subset=['bootstrap_r2'])

    if r2_data.empty:
        return {}

    fig_r2, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    _plot_r2_violin(ax1, r2_data)
    _plot_r2_vs_aic(ax2, r2_data)
    plt.tight_layout()

    return {'r2_distribution': fig_r2}


def _calculate_model_statistics(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate aggregated model-level statistics from bootstrap data.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Prepared bootstrap visualization data

    Returns
    -------
    pd.DataFrame
        Model-level statistics with AIC and R2 aggregations
    """
    model_stats = bootstrap_df.groupby(['model', 'model_features']).agg({
        'bootstrap_aic': ['mean', 'std', 'min', 'max', 'median'],
        'bootstrap_r2': ['mean', 'std', 'median'] if 'bootstrap_r2' in bootstrap_df.columns else lambda x: np.nan,
        'original_aic': 'first',
        'original_r2': 'first',
        'stability_assessment': 'first'
    }).round(4)

    model_stats.columns = [
        'mean_bootstrap_aic', 'std_bootstrap_aic', 'min_bootstrap_aic',
        'max_bootstrap_aic', 'median_bootstrap_aic',
        'mean_bootstrap_r2', 'std_bootstrap_r2', 'median_bootstrap_r2',
        'original_aic', 'original_r2', 'stability_assessment'
    ]
    return model_stats.reset_index()


def _calculate_stability_rankings(model_stats: pd.DataFrame) -> List[Dict[str, Any]]:
    """Calculate stability coefficients and rankings for models.

    Parameters
    ----------
    model_stats : pd.DataFrame
        Model-level statistics

    Returns
    -------
    List[Dict[str, Any]]
        Models sorted by combined stability and performance score
    """
    # Stability coefficient (inverse of coefficient of variation)
    model_stats['aic_stability_coefficient'] = 1 / (
        model_stats['std_bootstrap_aic'] / model_stats['mean_bootstrap_aic']
    )

    # Rank by stability (higher is better) and performance (lower AIC is better)
    model_stats['stability_rank'] = model_stats['aic_stability_coefficient'].rank(ascending=False)
    model_stats['performance_rank'] = model_stats['mean_bootstrap_aic'].rank(ascending=True)
    model_stats['combined_score'] = (
        model_stats['stability_rank'] + model_stats['performance_rank']
    ) / 2

    return model_stats.sort_values('combined_score').to_dict('records')


def _generate_insights_summary(model_stats: pd.DataFrame) -> str:
    """Generate human-readable insights summary from model statistics.

    Parameters
    ----------
    model_stats : pd.DataFrame
        Model-level statistics with stability coefficients

    Returns
    -------
    str
        Formatted insights summary
    """
    n_models = len(model_stats)
    best_stability = model_stats.loc[model_stats['aic_stability_coefficient'].idxmax()]
    best_performance = model_stats.loc[model_stats['mean_bootstrap_aic'].idxmin()]

    quality = 'High' if n_models >= 10 else 'Medium' if n_models >= 5 else 'Limited'

    return f"""Bootstrap Visualization Analysis reveals:
- Models Analyzed: {n_models} models with bootstrap distributions
- Most Stable Model: {best_stability['model']} (Stability Coef: {best_stability['aic_stability_coefficient']:.2f})
- Best Average Performance: {best_performance['model']} (Mean AIC: {best_performance['mean_bootstrap_aic']:.1f})
- AIC Variability Range: {model_stats['std_bootstrap_aic'].min():.2f} - {model_stats['std_bootstrap_aic'].max():.2f}
- Visualization Quality: {quality} sample coverage"""


def _build_visualization_results(
    bootstrap_df: pd.DataFrame,
    model_stats: pd.DataFrame,
    stability_ranking: List[Dict[str, Any]],
    insights_summary: str
) -> Dict[str, Any]:
    """Build final visualization insights results dictionary.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Original bootstrap data
    model_stats : pd.DataFrame
        Model-level statistics
    stability_ranking : List[Dict[str, Any]]
        Sorted stability rankings
    insights_summary : str
        Generated insights summary

    Returns
    -------
    Dict[str, Any]
        Complete visualization insights results
    """
    n_models = len(model_stats)
    return {
        'visualization_insights': insights_summary,
        'model_stability_ranking': stability_ranking[:10],
        'statistical_summary': {
            'n_models_analyzed': n_models,
            'mean_stability_coefficient': float(model_stats['aic_stability_coefficient'].mean()),
            'aic_variability_stats': {
                'mean_std': float(model_stats['std_bootstrap_aic'].mean()),
                'min_std': float(model_stats['std_bootstrap_aic'].min()),
                'max_std': float(model_stats['std_bootstrap_aic'].max())
            }
        },
        'analysis_metadata': {
            'bootstrap_samples_per_model': len(bootstrap_df) // n_models if n_models > 0 else 0,
            'visualization_coverage': 'comprehensive' if n_models >= 10 else 'partial',
            'analysis_type': 'bootstrap_visualization'
        }
    }


def generate_visualization_insights(bootstrap_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate business insights from bootstrap visualization data.

    Single responsibility: Insight generation only.
    Follows UNIFIED_CODING_STANDARDS.md with focused business analysis.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Prepared bootstrap visualization data from prepare_bootstrap_visualization_data

    Returns
    -------
    Dict[str, Any]
        Business insights and statistical summary from visualization analysis
    """
    if bootstrap_df.empty:
        return {
            'visualization_insights': 'No data available for visualization analysis',
            'model_stability_ranking': [],
            'statistical_summary': {}
        }

    # Calculate statistics and rankings
    model_stats = _calculate_model_statistics(bootstrap_df)
    stability_ranking = _calculate_stability_rankings(model_stats)
    insights_summary = _generate_insights_summary(model_stats)

    return _build_visualization_results(
        bootstrap_df, model_stats, stability_ranking, insights_summary
    )


def _display_data_summary(bootstrap_df: pd.DataFrame) -> None:
    """Display bootstrap data preparation summary.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Prepared bootstrap visualization data
    """
    n_models = bootstrap_df['model'].nunique()
    n_samples = len(bootstrap_df)

    print(f"\nBootstrap Visualization Data Summary:")
    print(f"{'Models':<15} {'Samples':<10} {'Avg per Model':<15}")
    print("-" * 45)
    print(f"{n_models:<15} {n_samples:<10} {n_samples//n_models if n_models > 0 else 0:<15}")


def _create_all_visualizations(bootstrap_df: pd.DataFrame,
                               config: Optional[Dict[str, Any]]) -> Dict[str, plt.Figure]:
    """Create all visualization figures and display them.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Prepared bootstrap visualization data
    config : Optional[Dict[str, Any]]
        Configuration for visualization styling

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of visualization figures by name
    """
    all_visualizations = {}

    try:
        # AIC distribution visualizations
        aic_viz = create_aic_distribution_visualizations(bootstrap_df, config)
        all_visualizations.update(aic_viz)

        # Stability comparison visualizations
        stability_viz = create_stability_comparison_visualizations(bootstrap_df, config)
        all_visualizations.update(stability_viz)

        # R² distribution visualizations
        r2_viz = create_r2_distribution_visualizations(bootstrap_df, config)
        all_visualizations.update(r2_viz)

        # Display visualizations
        for viz_name, fig in all_visualizations.items():
            plt.figure(fig.number)  # Make figure active
            plt.show()

        print(f"SUCCESS: Bootstrap visualizations displayed ({len(all_visualizations)} figures)")

    except Exception as e:
        print(f"WARNING: Visualization creation failed: {str(e)}")
        return {}

    return all_visualizations


def _build_analysis_results(bootstrap_df: pd.DataFrame,
                            insights: Dict[str, Any],
                            all_visualizations: Dict[str, plt.Figure],
                            max_models: int) -> Dict[str, Any]:
    """Build comprehensive analysis results dictionary.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Prepared bootstrap visualization data
    insights : Dict[str, Any]
        Generated visualization insights
    all_visualizations : Dict[str, plt.Figure]
        Dictionary of visualization figures
    max_models : int
        Maximum models configured for analysis

    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis results with metadata
    """
    return {
        'visualization_data': bootstrap_df,
        'insights': insights,
        'visualizations': all_visualizations,
        'analysis_metadata': {
            'n_models_analyzed': bootstrap_df['model'].nunique(),
            'total_bootstrap_samples': len(bootstrap_df),
            'max_models_configured': max_models,
            'analysis_complete': True
        }
    }


def run_bootstrap_visualization_analysis(bootstrap_results: List[BootstrapResult],
                                       config: Optional[Dict[str, Any]] = None,
                                       display_results: bool = True,
                                       create_visualizations: bool = True,
                                       return_detailed: bool = True) -> Dict[str, Any]:
    """Run comprehensive bootstrap visualization analysis with AIC and R2 distributions."""
    if not bootstrap_results:
        raise ValueError("No bootstrap results provided for visualization analysis")

    max_models = config.get('models_to_analyze', 15) if config else 15
    print("=== BOOTSTRAP VISUALIZATION ANALYSIS ===")

    try:
        bootstrap_df = prepare_bootstrap_visualization_data(bootstrap_results, max_models)

        if display_results:
            _display_data_summary(bootstrap_df)

        all_visualizations = {}
        if create_visualizations:
            all_visualizations = _create_all_visualizations(bootstrap_df, config)

        insights = generate_visualization_insights(bootstrap_df)

        if display_results:
            print(f"\n=== VISUALIZATION INSIGHTS ===")
            print(insights['visualization_insights'])

        print(f"SUCCESS: Bootstrap Visualization Analysis complete")

        if return_detailed:
            return _build_analysis_results(bootstrap_df, insights, all_visualizations, max_models)
        else:
            return {'visualization_data': bootstrap_df}

    except Exception as e:
        print(f"ERROR: Bootstrap visualization analysis failed: {str(e)}")
        raise RuntimeError(f"Visualization analysis failed: {str(e)}") from e


# Convenience function for notebook integration
def run_notebook_visualization_analysis(bootstrap_results: List[BootstrapResult]) -> pd.DataFrame:
    """
    Notebook-friendly visualization analysis with automatic display.

    Simplified interface for backward compatibility with existing notebook cells.
    """
    if not bootstrap_results:
        print("No bootstrap results available for visualization analysis")
        return pd.DataFrame()

    analysis_results = run_bootstrap_visualization_analysis(
        bootstrap_results=bootstrap_results,
        display_results=True,
        create_visualizations=True,
        return_detailed=False
    )

    return analysis_results.get('visualization_data', pd.DataFrame())