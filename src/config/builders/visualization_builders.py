"""
Visualization Configuration Builders

Extracted from config_builder.py for single-responsibility compliance.
Handles all visualization configurations: figure sizing, seaborn styling,
violin plots, line colors, and annotation offsets.

Usage:
    from src.config.visualization_builders import build_visualization_config
    config = build_visualization_config(figure_size=(10, 15))
"""

from typing import Dict, Any

from src.config.types.pipeline_config import VisualizationConfig


# Module exports (including private functions for testing)
__all__ = [
    # Public API
    'build_visualization_config',
    # Private helpers (for unit testing)
    '_build_violin_params',
    '_build_line_colors',
    '_build_annotation_offsets',
]


# =============================================================================
# VISUALIZATION CONFIGURATION HELPERS
# =============================================================================


def _build_violin_params() -> Dict[str, Any]:
    """Build violin plot parameters for seaborn visualization.

    Returns parameters matching original notebook exactly for
    pixel-perfect equivalence in violin plot rendering.

    Returns
    -------
    Dict[str, Any]
        Violin plot configuration
    """
    return {
        'hue': True,
        'hue_order': [True, False],
        'split': True,
        'dodge': True,
        'density_norm': 'width',
        'inner': 'quartile',
        'legend': False
    }


def _build_line_colors() -> Dict[str, str]:
    """Build line color configuration for plots.

    Defines consistent colors for confidence interval bounds,
    median lines, and scatter points.

    Returns
    -------
    Dict[str, str]
        Line color mapping for CI bounds, median, and scatter
    """
    return {
        'ci_bounds': 'tab:red',
        'median': 'k',
        'scatter': 'k'
    }


def _build_annotation_offsets() -> Dict[str, tuple]:
    """Build annotation offset configuration for plot labels.

    Defines precise offsets for annotation placement on percentage
    and dollar-denominated plots, matching original notebook exactly.

    Returns
    -------
    Dict[str, tuple]
        Annotation offsets for percentage and dollar plots
    """
    return {
        # Percentage plot annotation offsets (exact from original)
        'pct_bottom': (-5, 5),
        'pct_median': (10, 20),
        'pct_top': (5, 5),
        # Dollar plot annotation offsets (exact from original)
        'dollar_bottom': (-6, 8),
        'dollar_median': (5, 25),
        'dollar_top': (6, 8)
    }


# =============================================================================
# MAIN VISUALIZATION CONFIG BUILDER
# =============================================================================


def build_visualization_config(
    figure_size: tuple = (10, 15),
    seaborn_style: str = "whitegrid",
    seaborn_palette: str = "deep",
    output_directory: str = "../../outputs/rila_6y20b/bi_team",
    file_prefix: str = "price_elasticity_FlexGuard",
    dpi: int = 300
) -> VisualizationConfig:
    """Build visualization config for centralized outputs directory.

    Default output_directory is relative to notebooks/production/rila_6y20b/.
    For other products, override this parameter when calling build_inference_stage_config().
    """
    return VisualizationConfig({
        'figure_size': figure_size,
        'seaborn_style': seaborn_style,
        'seaborn_palette': seaborn_palette,
        'violin_params': _build_violin_params(),
        'line_colors': _build_line_colors(),
        'annotation_offsets': _build_annotation_offsets(),
        'output_directory': output_directory,
        'file_prefix': file_prefix,
        'dpi': dpi
    })
