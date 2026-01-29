"""
README Plot Export Module

Zero-regression module for exporting plots to README documentation.
This module creates README-ready plots without modifying any existing analysis logic.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Optional, Union
import datetime

# Set professional styling for README plots
matplotlib.rcParams.update({
    'font.size': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3
})

def export_for_readme(
    figure: plt.Figure,
    category: str,
    plot_name: str,
    description: Optional[str] = None,
    version: str = "v6"
) -> str:
    """
    Export a matplotlib figure for README documentation.

    This function creates high-quality exports without modifying existing analysis.

    Parameters:
    -----------
    figure : plt.Figure
        The existing matplotlib figure to export (already created by notebooks)
    category : str
        Category for organization: 'business_intelligence', 'model_performance',
        'data_pipeline', 'competitive_analysis'
    plot_name : str
        Descriptive name for the plot file
    description : str, optional
        Description to include in metadata
    version : str
        Version identifier (default: "v6")

    Returns:
    --------
    str : Path to the exported file
    """

    # Create directory if it doesn't exist
    # Path relative to notebooks/production/{product}/ → ../../../docs/images/
    base_dir = Path("../../../docs/images")
    category_dir = base_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp and version
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"{category}_{plot_name}_{version}.png"
    filepath = category_dir / filename

    # Save the figure with professional quality settings
    figure.savefig(
        filepath,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.2,
        facecolor='white',
        edgecolor='none'
    )

    # Also create a latest version (no version in filename for README linking)
    latest_filename = f"{category}_{plot_name}_latest.png"
    latest_filepath = category_dir / latest_filename
    figure.savefig(
        latest_filepath,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.2,
        facecolor='white',
        edgecolor='none'
    )

    print(f"[OK] Exported README plot: {filepath}")
    print(f"[OK] Created latest version: {latest_filepath}")

    if description:
        print(f"   Description: {description}")

    return str(filepath)


def export_business_intelligence_plots(
    price_elasticity_fig_pct: plt.Figure,
    price_elasticity_fig_dollars: plt.Figure
) -> None:
    """
    Export key business intelligence plots for executive summary.

    Parameters:
    -----------
    price_elasticity_fig_pct : plt.Figure
        Price elasticity percentage change figure
    price_elasticity_fig_dollars : plt.Figure
        Price elasticity dollar impact figure
    """

    # Export percentage change plot for executive summary
    export_for_readme(
        price_elasticity_fig_pct,
        "business_intelligence",
        "price_elasticity_confidence_intervals_pct",
        "Strategic rate scenarios (0-450 basis points) with 95% confidence intervals"
    )

    # Export dollar impact plot for revenue planning
    export_for_readme(
        price_elasticity_fig_dollars,
        "business_intelligence",
        "price_elasticity_confidence_intervals_dollars",
        "Revenue impact projections with bootstrap uncertainty quantification"
    )


def export_model_performance_plots(
    comprehensive_fig: plt.Figure,
    bootstrap_comparison_fig: Optional[plt.Figure] = None
) -> None:
    """
    Export model performance validation plots.

    Parameters:
    -----------
    comprehensive_fig : plt.Figure
        Comprehensive forecasting analysis figure
    bootstrap_comparison_fig : plt.Figure, optional
        Bootstrap vs benchmark comparison figure
    """

    # Export comprehensive performance analysis
    export_for_readme(
        comprehensive_fig,
        "model_performance",
        "comprehensive_forecasting_analysis",
        "Bootstrap Ridge forecasting with 125 out-of-sample validation periods"
    )

    if bootstrap_comparison_fig:
        export_for_readme(
            bootstrap_comparison_fig,
            "model_performance",
            "bootstrap_vs_benchmark_comparison",
            "Model performance improvement: R² and MAPE comparison"
        )


def export_data_pipeline_plots(
    sales_competitive_fig: plt.Figure,
    data_quality_fig: Optional[plt.Figure] = None
) -> None:
    """
    Export data pipeline and architecture plots.

    Parameters:
    -----------
    sales_competitive_fig : plt.Figure
        Sales vs competitive spreads analysis figure
    data_quality_fig : plt.Figure, optional
        Data quality validation figure
    """

    # Export sales vs competitive analysis
    export_for_readme(
        sales_competitive_fig,
        "data_pipeline",
        "sales_vs_competitive_spreads",
        "FlexGuard Sales Performance vs Competitive Rate Positioning"
    )

    if data_quality_fig:
        export_for_readme(
            data_quality_fig,
            "data_pipeline",
            "data_quality_validation",
            "Pipeline data quality validation and transformation verification"
        )


def create_readme_plot_catalog() -> str:
    """
    Generate a markdown catalog of all available README plots.

    Returns:
    --------
    str : Markdown content listing all available plots
    """

    base_dir = Path("../../../docs/images")
    if not base_dir.exists():
        return "No README plots available yet."

    catalog = ["# README Plot Catalog", "", "## Available Visualizations", ""]

    categories = ["business_intelligence", "model_performance", "data_pipeline", "competitive_analysis"]

    for category in categories:
        category_dir = base_dir / category
        if category_dir.exists():
            plots = list(category_dir.glob("*_latest.png"))
            if plots:
                catalog.append(f"### {category.replace('_', ' ').title()}")
                catalog.append("")
                for plot in sorted(plots):
                    plot_name = plot.stem.replace(f"{category}_", "").replace("_latest", "")
                    catalog.append(f"- `{plot.name}` - {plot_name.replace('_', ' ').title()}")
                catalog.append("")

    catalog.append("## Usage in README")
    catalog.append("```markdown")
    catalog.append("![Plot Description](../../../docs/images/category/plot_name_latest.png)")
    catalog.append("```")

    return "\n".join(catalog)


def validate_plot_exports() -> bool:
    """
    Validate that all expected README plots exist and are accessible.

    Returns:
    --------
    bool : True if all critical plots are available
    """

    base_dir = Path("../../../docs/images")
    critical_plots = [
        "business_intelligence/business_intelligence_price_elasticity_confidence_intervals_pct_latest.png",
        "model_performance/model_performance_comprehensive_forecasting_analysis_latest.png"
    ]

    missing_plots = []
    for plot_path in critical_plots:
        full_path = base_dir / plot_path
        if not full_path.exists():
            missing_plots.append(plot_path)

    if missing_plots:
        print("[MISSING] Missing critical README plots:")
        for plot in missing_plots:
            print(f"   - {plot}")
        return False

    print("[OK] All critical README plots are available")
    return True


# Professional color palette for consistent branding
README_COLORS = {
    'primary': '#1f77b4',      # Professional blue
    'secondary': '#ff7f0e',    # Professional orange
    'success': '#2ca02c',      # Professional green
    'warning': '#d62728',      # Professional red
    'info': '#9467bd',         # Professional purple
    'neutral': '#7f7f7f',      # Professional gray
    'light': '#bcbd22',        # Professional olive
    'accent': '#17becf'        # Professional cyan
}


def apply_readme_styling(fig: plt.Figure, title: Optional[str] = None) -> plt.Figure:
    """
    Apply consistent professional styling for README plots.

    Parameters:
    -----------
    fig : plt.Figure
        Figure to style
    title : str, optional
        Title to add to the figure

    Returns:
    --------
    plt.Figure : Styled figure
    """

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    for ax in fig.axes:
        # Apply professional grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

        # Style spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#333333')

        # Style ticks
        ax.tick_params(labelsize=10, colors='#333333')

        # Style labels
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight='bold')
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig