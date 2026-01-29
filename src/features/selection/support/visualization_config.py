"""
Visualization Configuration Module for Feature Selection Pipeline.

This module provides centralized configuration functions for bootstrap visualizations,
ensuring consistent styling, parameters, and business-oriented display settings
following CODING_STANDARDS.md Section 3.1 requirements.

Purpose: Centralize visualization configuration (DRY compliance)
Status: NEW (consolidation of scattered configuration)
Priority: HIGH (DRY compliance and maintainability)

Key Functions:
- build_visualization_configs(): Central configuration builder (≤50 lines)
- get_default_style_config(): Default styling configuration (≤50 lines)
- create_adaptive_layout_config(): Adaptive layout configuration (≤50 lines)
- validate_visualization_config(): Configuration validation (≤50 lines)

Mathematical Equivalence: All functions maintain identical results to original
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def _build_bootstrap_viz_config(models_to_analyze: int,
                                base_fig_width: int,
                                base_fig_height: int) -> Dict[str, Any]:
    """Build bootstrap visualization configuration.

    Parameters
    ----------
    models_to_analyze : int
        Number of models to include in visualizations
    base_fig_width : int
        Base figure width for scaling
    base_fig_height : int
        Base figure height for scaling

    Returns
    -------
    Dict[str, Any]
        Bootstrap visualization configuration
    """
    return {
        'models_to_analyze': models_to_analyze,
        'fig_width': base_fig_width,
        'fig_height': base_fig_height,
        'display_results': True,
        'create_visualizations': True,
        'return_detailed': True
    }


def _build_violin_config(models_to_analyze: int,
                         base_fig_width: int,
                         base_fig_height: int,
                         is_large_set: bool) -> Dict[str, Any]:
    """Build violin plot configuration optimized for model count.

    Parameters
    ----------
    models_to_analyze : int
        Number of models to include in visualizations
    base_fig_width : int
        Base figure width for scaling
    base_fig_height : int
        Base figure height for scaling
    is_large_set : bool
        Whether model count exceeds 10

    Returns
    -------
    Dict[str, Any]
        Violin plot configuration
    """
    return {
        'models_to_analyze': models_to_analyze,
        'fig_width': base_fig_width - 4 if is_large_set else base_fig_width - 2,
        'fig_height': base_fig_height + (2 if is_large_set else 0),
        'kde_alpha': 0.7,
        'scatter_size': 60,
        'line_width': 2.5
    }


def _build_boxplot_config(models_to_analyze: int,
                          base_fig_width: int,
                          base_fig_height: int,
                          is_large_set: bool) -> Dict[str, Any]:
    """Build boxplot configuration adapted for readability.

    Parameters
    ----------
    models_to_analyze : int
        Number of models to include in visualizations
    base_fig_width : int
        Base figure width for scaling
    base_fig_height : int
        Base figure height for scaling
    is_large_set : bool
        Whether model count exceeds 10

    Returns
    -------
    Dict[str, Any]
        Boxplot configuration
    """
    return {
        'models_to_analyze': models_to_analyze,
        'fig_width': base_fig_width + (4 if is_large_set else 0),
        'fig_height': base_fig_height,
        'rotation_angle': 45 if is_large_set else 30,
        'box_alpha': 0.7,
        'label_max_chars': 35
    }


def _build_analysis_configs(models_to_analyze: int,
                            base_fig_width: int,
                            base_fig_height: int) -> Dict[str, Dict[str, Any]]:
    """Build win rate and information ratio analysis configurations.

    Parameters
    ----------
    models_to_analyze : int
        Number of models to include in visualizations
    base_fig_width : int
        Base figure width for scaling
    base_fig_height : int
        Base figure height for scaling

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with win_rate_analysis and information_ratio configs
    """
    win_rate_config = {
        'models_to_analyze': models_to_analyze,
        'fig_width': base_fig_width,
        'fig_height': max(6, models_to_analyze // 3),
        'bar_alpha': 0.8,
        'display_percentages': True
    }

    ir_config = {
        'models_to_analyze': models_to_analyze,
        'fig_width': base_fig_width,
        'fig_height': base_fig_height + 2,
        'scatter_size': 80,
        'alpha': 0.7
    }

    return {
        'win_rate_analysis': win_rate_config,
        'information_ratio': ir_config
    }


def build_visualization_configs(models_to_analyze: int = 15,
                              base_fig_width: int = 16,
                              base_fig_height: int = 10) -> Dict[str, Dict[str, Any]]:
    """Build comprehensive visualization configuration with adaptive parameters."""
    if models_to_analyze <= 0 or models_to_analyze > 50:
        raise ValueError(
            f"CRITICAL: models_to_analyze must be 1-50, got {models_to_analyze}. "
            f"Business impact: Visualization scope constraint violated. "
            f"Required action: Use reasonable model count for clear visualization."
        )

    try:
        is_large_set = models_to_analyze > 10

        configs = {
            'bootstrap_visualization': _build_bootstrap_viz_config(
                models_to_analyze, base_fig_width, base_fig_height
            ),
            'violin_plot': _build_violin_config(
                models_to_analyze, base_fig_width, base_fig_height, is_large_set
            ),
            'boxplot': _build_boxplot_config(
                models_to_analyze, base_fig_width, base_fig_height, is_large_set
            ),
        }

        analysis_configs = _build_analysis_configs(
            models_to_analyze, base_fig_width, base_fig_height
        )
        configs.update(analysis_configs)

        return configs

    except Exception as e:
        raise ValueError(
            f"CRITICAL: Visualization configuration building failed: {e}. "
            f"Business impact: Cannot create standardized visualization parameters. "
            f"Required action: Check configuration parameters and defaults."
        ) from e


# Module-level style constants
_DEFAULT_COLORS = {
    'stable': '#2E8B57', 'moderate': '#FF8C00', 'unstable': '#DC143C',
    'default': '#4682B4', 'median': '#000080', 'background': '#F5F5F5'
}
_DEFAULT_FONTS = {
    'title_size': 14, 'label_size': 12, 'tick_size': 10, 'legend_size': 10,
    'title_weight': 'bold', 'label_weight': 'normal'
}
_DEFAULT_GRID = {'alpha': 0.3, 'style': '-', 'width': 0.5, 'x_grid': True, 'y_grid': False}
_DEFAULT_SPINES = {'top': False, 'right': False, 'left': True, 'bottom': True, 'color': '#333333', 'width': 1.0}
_DEFAULT_BUSINESS = {
    'subtitle_suffix': ' - Bootstrap Analysis', 'axis_label_style': 'Lower is Better',
    'stability_legend': True, 'confidence_indicators': True
}


def get_default_style_config() -> Dict[str, Any]:
    """Get default styling configuration for consistent visualization appearance."""
    return {
        'colors': _DEFAULT_COLORS.copy(),
        'fonts': _DEFAULT_FONTS.copy(),
        'grid': _DEFAULT_GRID.copy(),
        'spines': _DEFAULT_SPINES.copy(),
        'business_elements': _DEFAULT_BUSINESS.copy()
    }


def _get_layout_parameters(models_count: int) -> tuple:
    """Get layout type, rotation, and spacing based on model count."""
    if models_count <= 5:
        return 'compact', 0, 1.0
    elif models_count <= 10:
        return 'standard', 15, 1.2
    elif models_count <= 15:
        return 'expanded', 30, 1.5
    return 'large', 45, 2.0


def create_adaptive_layout_config(models_count: int) -> Dict[str, Any]:
    """Create adaptive layout configuration based on number of models."""
    if models_count <= 0:
        raise ValueError(f"models_count must be positive, got {models_count}")

    layout_type, rotation, spacing = _get_layout_parameters(models_count)
    return {
        'layout_type': layout_type,
        'rotation_angle': rotation,
        'spacing_factor': spacing,
        'tick_spacing': max(1, models_count // 10),
        'label_padding': 0.1 * spacing,
        'margin_adjustment': {'left': 0.1, 'right': 0.9, 'bottom': 0.15 * spacing, 'top': 0.9},
        'font_scaling': {
            'title': 1.0,
            'labels': max(0.8, 1.0 - (models_count - 10) * 0.05),
            'ticks': max(0.7, 1.0 - (models_count - 10) * 0.03)
        }
    }


def _validate_config_values(config: Dict[str, Any]) -> List[str]:
    """Validate individual config values for type and range constraints."""
    errors = []
    if 'models_to_analyze' in config:
        m = config['models_to_analyze']
        if not isinstance(m, int) or m <= 0 or m > 50:
            errors.append("models_to_analyze must be positive integer <= 50")
    if 'fig_width' in config:
        w = config['fig_width']
        if not isinstance(w, (int, float)) or w <= 0 or w > 30:
            errors.append("fig_width must be positive number <= 30")
    if 'fig_height' in config:
        h = config['fig_height']
        if not isinstance(h, (int, float)) or h <= 0 or h > 20:
            errors.append("fig_height must be positive number <= 20")
    if 'fig_width' in config and 'fig_height' in config:
        ratio = config['fig_width'] / config['fig_height']
        if ratio < 0.5 or ratio > 4.0:
            errors.append("Figure aspect ratio should be between 0.5 and 4.0")
    return errors


def validate_visualization_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate visualization configuration for completeness and correctness."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be dictionary")
    errors = []
    required = {'models_to_analyze': 'scope', 'fig_width': 'width', 'fig_height': 'height'}
    for key, desc in required.items():
        if key not in config:
            errors.append(f"Missing required key '{key}' ({desc})")
    errors.extend(_validate_config_values(config))
    return len(errors) == 0, errors