"""
Tests for Visualization Configuration Module.

Tests cover:
- build_visualization_configs(): Central configuration builder with validation
- get_default_style_config(): Default styling configuration
- create_adaptive_layout_config(): Adaptive layout based on model count
- validate_visualization_config(): Configuration validation

Design Principles:
- Property-based tests for configuration invariants
- Edge case tests for boundary conditions
- Pure function tests (no mocking needed)

Mathematical Properties Validated:
- models_to_analyze must be 1-50
- Figure dimensions must be positive
- Aspect ratio must be reasonable (0.5-4.0)
- Layout type determined by model count thresholds

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import numpy as np

from src.features.selection.support.visualization_config import (
    # Internal builders
    _build_bootstrap_viz_config,
    _build_violin_config,
    _build_boxplot_config,
    _build_analysis_configs,
    # Main functions
    build_visualization_configs,
    get_default_style_config,
    _get_layout_parameters,
    create_adaptive_layout_config,
    _validate_config_values,
    validate_visualization_config,
    # Module-level constants
    _DEFAULT_COLORS,
    _DEFAULT_FONTS,
    _DEFAULT_GRID,
    _DEFAULT_SPINES,
    _DEFAULT_BUSINESS,
)


# =============================================================================
# Tests for _build_bootstrap_viz_config
# =============================================================================


class TestBuildBootstrapVizConfig:
    """Tests for bootstrap visualization config builder."""

    def test_returns_dict_with_expected_keys(self):
        """Config contains all required keys."""
        config = _build_bootstrap_viz_config(10, 16, 10)

        expected_keys = {
            'models_to_analyze',
            'fig_width',
            'fig_height',
            'display_results',
            'create_visualizations',
            'return_detailed',
        }
        assert set(config.keys()) == expected_keys

    def test_preserves_input_parameters(self):
        """Config preserves provided parameters."""
        config = _build_bootstrap_viz_config(15, 20, 12)

        assert config['models_to_analyze'] == 15
        assert config['fig_width'] == 20
        assert config['fig_height'] == 12

    def test_boolean_flags_default_true(self):
        """Display/visualization flags default to True."""
        config = _build_bootstrap_viz_config(5, 10, 8)

        assert config['display_results'] is True
        assert config['create_visualizations'] is True
        assert config['return_detailed'] is True


# =============================================================================
# Tests for _build_violin_config
# =============================================================================


class TestBuildViolinConfig:
    """Tests for violin plot config builder."""

    def test_small_set_dimensions(self):
        """Small model set (<=10) uses minimal width reduction."""
        config = _build_violin_config(5, 16, 10, is_large_set=False)

        assert config['fig_width'] == 14  # 16 - 2
        assert config['fig_height'] == 10  # No increase

    def test_large_set_dimensions(self):
        """Large model set (>10) increases height and reduces width more."""
        config = _build_violin_config(15, 16, 10, is_large_set=True)

        assert config['fig_width'] == 12  # 16 - 4
        assert config['fig_height'] == 12  # 10 + 2

    def test_contains_plotting_parameters(self):
        """Config includes styling parameters for violin plots."""
        config = _build_violin_config(10, 16, 10, is_large_set=False)

        assert config['kde_alpha'] == 0.7
        assert config['scatter_size'] == 60
        assert config['line_width'] == 2.5


# =============================================================================
# Tests for _build_boxplot_config
# =============================================================================


class TestBuildBoxplotConfig:
    """Tests for boxplot config builder."""

    def test_small_set_layout(self):
        """Small model set uses compact rotation."""
        config = _build_boxplot_config(5, 16, 10, is_large_set=False)

        assert config['fig_width'] == 16  # No width increase
        assert config['rotation_angle'] == 30

    def test_large_set_layout(self):
        """Large model set increases width and rotation for readability."""
        config = _build_boxplot_config(15, 16, 10, is_large_set=True)

        assert config['fig_width'] == 20  # 16 + 4
        assert config['rotation_angle'] == 45

    def test_contains_styling_parameters(self):
        """Config includes boxplot-specific styling."""
        config = _build_boxplot_config(10, 16, 10, is_large_set=False)

        assert config['box_alpha'] == 0.7
        assert config['label_max_chars'] == 35


# =============================================================================
# Tests for _build_analysis_configs
# =============================================================================


class TestBuildAnalysisConfigs:
    """Tests for win rate and information ratio analysis configs."""

    def test_returns_two_configs(self):
        """Returns both win_rate_analysis and information_ratio configs."""
        configs = _build_analysis_configs(10, 16, 10)

        assert 'win_rate_analysis' in configs
        assert 'information_ratio' in configs
        assert len(configs) == 2

    def test_win_rate_height_scales_with_models(self):
        """Win rate figure height scales with model count."""
        configs_small = _build_analysis_configs(9, 16, 10)
        configs_large = _build_analysis_configs(30, 16, 10)

        assert configs_small['win_rate_analysis']['fig_height'] == 6  # max(6, 9//3=3) = 6
        assert configs_large['win_rate_analysis']['fig_height'] == 10  # max(6, 30//3=10) = 10

    def test_information_ratio_config(self):
        """Information ratio config has appropriate settings."""
        configs = _build_analysis_configs(10, 16, 10)
        ir_config = configs['information_ratio']

        assert ir_config['fig_height'] == 12  # 10 + 2
        assert ir_config['scatter_size'] == 80
        assert ir_config['alpha'] == 0.7


# =============================================================================
# Tests for build_visualization_configs (Main Function)
# =============================================================================


class TestBuildVisualizationConfigs:
    """Tests for the main visualization config builder."""

    def test_returns_all_config_sections(self):
        """Returns configs for all visualization types."""
        configs = build_visualization_configs()

        expected_sections = {
            'bootstrap_visualization',
            'violin_plot',
            'boxplot',
            'win_rate_analysis',
            'information_ratio',
        }
        assert set(configs.keys()) == expected_sections

    def test_default_parameters(self):
        """Uses sensible defaults when no parameters provided."""
        configs = build_visualization_configs()

        # Default is 15 models
        assert configs['bootstrap_visualization']['models_to_analyze'] == 15
        # Default width is 16
        assert configs['bootstrap_visualization']['fig_width'] == 16

    def test_custom_parameters_propagate(self):
        """Custom parameters propagate to all configs."""
        configs = build_visualization_configs(
            models_to_analyze=25, base_fig_width=20, base_fig_height=12
        )

        # All configs should use provided models_to_analyze
        assert configs['bootstrap_visualization']['models_to_analyze'] == 25
        assert configs['violin_plot']['models_to_analyze'] == 25
        assert configs['boxplot']['models_to_analyze'] == 25

    def test_rejects_zero_models(self):
        """Raises ValueError for zero models."""
        with pytest.raises(ValueError) as exc_info:
            build_visualization_configs(models_to_analyze=0)

        assert "must be 1-50" in str(exc_info.value)
        assert "CRITICAL" in str(exc_info.value)

    def test_rejects_negative_models(self):
        """Raises ValueError for negative models."""
        with pytest.raises(ValueError) as exc_info:
            build_visualization_configs(models_to_analyze=-5)

        assert "must be 1-50" in str(exc_info.value)

    def test_rejects_excessive_models(self):
        """Raises ValueError for >50 models."""
        with pytest.raises(ValueError) as exc_info:
            build_visualization_configs(models_to_analyze=51)

        assert "must be 1-50" in str(exc_info.value)

    @pytest.mark.parametrize("n_models", [1, 10, 25, 50])
    def test_valid_model_counts_accepted(self, n_models):
        """All valid model counts (1-50) produce configs."""
        configs = build_visualization_configs(models_to_analyze=n_models)

        assert configs['bootstrap_visualization']['models_to_analyze'] == n_models

    def test_large_set_detection(self):
        """Correctly detects large model sets (>10)."""
        configs_small = build_visualization_configs(models_to_analyze=10)
        configs_large = build_visualization_configs(models_to_analyze=11)

        # Small set: no height increase for violin
        assert configs_small['violin_plot']['fig_height'] == 10
        # Large set: height increase for violin
        assert configs_large['violin_plot']['fig_height'] == 12


# =============================================================================
# Tests for get_default_style_config
# =============================================================================


class TestGetDefaultStyleConfig:
    """Tests for default style configuration getter."""

    def test_returns_all_style_sections(self):
        """Returns all required style configuration sections."""
        style = get_default_style_config()

        expected_sections = {'colors', 'fonts', 'grid', 'spines', 'business_elements'}
        assert set(style.keys()) == expected_sections

    def test_colors_section(self):
        """Colors section contains expected keys."""
        style = get_default_style_config()
        colors = style['colors']

        assert 'stable' in colors
        assert 'moderate' in colors
        assert 'unstable' in colors
        assert 'default' in colors
        assert 'median' in colors
        assert 'background' in colors

    def test_fonts_section(self):
        """Fonts section contains sizing and weight parameters."""
        style = get_default_style_config()
        fonts = style['fonts']

        assert fonts['title_size'] == 14
        assert fonts['label_size'] == 12
        assert fonts['title_weight'] == 'bold'

    def test_grid_section(self):
        """Grid section has visibility and styling parameters."""
        style = get_default_style_config()
        grid = style['grid']

        assert grid['alpha'] == 0.3
        assert grid['x_grid'] is True
        assert grid['y_grid'] is False

    def test_spines_section(self):
        """Spines section controls border visibility."""
        style = get_default_style_config()
        spines = style['spines']

        assert spines['top'] is False
        assert spines['right'] is False
        assert spines['left'] is True
        assert spines['bottom'] is True

    def test_returns_copy_not_reference(self):
        """Each call returns independent copy (no shared state)."""
        style1 = get_default_style_config()
        style2 = get_default_style_config()

        # Modify one
        style1['colors']['stable'] = '#000000'

        # Other should be unchanged
        assert style2['colors']['stable'] == '#2E8B57'


# =============================================================================
# Tests for _get_layout_parameters
# =============================================================================


class TestGetLayoutParameters:
    """Tests for layout parameter selection."""

    @pytest.mark.parametrize(
        "models_count,expected_layout,expected_rotation,expected_spacing",
        [
            (1, 'compact', 0, 1.0),
            (5, 'compact', 0, 1.0),
            (6, 'standard', 15, 1.2),
            (10, 'standard', 15, 1.2),
            (11, 'expanded', 30, 1.5),
            (15, 'expanded', 30, 1.5),
            (16, 'large', 45, 2.0),
            (50, 'large', 45, 2.0),
        ],
    )
    def test_layout_thresholds(
        self, models_count, expected_layout, expected_rotation, expected_spacing
    ):
        """Layout type, rotation, and spacing follow defined thresholds."""
        layout_type, rotation, spacing = _get_layout_parameters(models_count)

        assert layout_type == expected_layout
        assert rotation == expected_rotation
        assert spacing == expected_spacing


# =============================================================================
# Tests for create_adaptive_layout_config
# =============================================================================


class TestCreateAdaptiveLayoutConfig:
    """Tests for adaptive layout configuration."""

    def test_returns_complete_config(self):
        """Returns config with all required fields."""
        config = create_adaptive_layout_config(10)

        expected_keys = {
            'layout_type',
            'rotation_angle',
            'spacing_factor',
            'tick_spacing',
            'label_padding',
            'margin_adjustment',
            'font_scaling',
        }
        assert set(config.keys()) == expected_keys

    def test_compact_layout_for_small_counts(self):
        """Small model counts use compact layout."""
        config = create_adaptive_layout_config(5)

        assert config['layout_type'] == 'compact'
        assert config['rotation_angle'] == 0
        assert config['spacing_factor'] == 1.0

    def test_large_layout_for_high_counts(self):
        """High model counts use large layout with more rotation."""
        config = create_adaptive_layout_config(20)

        assert config['layout_type'] == 'large'
        assert config['rotation_angle'] == 45
        assert config['spacing_factor'] == 2.0

    def test_tick_spacing_scales(self):
        """Tick spacing increases with model count."""
        config_small = create_adaptive_layout_config(5)
        config_large = create_adaptive_layout_config(30)

        assert config_small['tick_spacing'] == 1  # max(1, 5//10=0) = 1
        assert config_large['tick_spacing'] == 3  # max(1, 30//10=3) = 3

    def test_font_scaling_decreases_for_large_counts(self):
        """Font scaling decreases as model count increases."""
        config = create_adaptive_layout_config(30)
        font_scaling = config['font_scaling']

        # labels: max(0.8, 1.0 - (30-10)*0.05) = max(0.8, 0.0) = 0.8
        assert font_scaling['labels'] == 0.8
        # ticks: max(0.7, 1.0 - (30-10)*0.03) = max(0.7, 0.4) = 0.7
        assert font_scaling['ticks'] == 0.7

    def test_margin_adjustment_structure(self):
        """Margin adjustment has all four sides."""
        config = create_adaptive_layout_config(10)
        margins = config['margin_adjustment']

        assert 'left' in margins
        assert 'right' in margins
        assert 'top' in margins
        assert 'bottom' in margins

    def test_rejects_zero_models(self):
        """Raises ValueError for zero models."""
        with pytest.raises(ValueError) as exc_info:
            create_adaptive_layout_config(0)

        assert "must be positive" in str(exc_info.value)

    def test_rejects_negative_models(self):
        """Raises ValueError for negative models."""
        with pytest.raises(ValueError) as exc_info:
            create_adaptive_layout_config(-5)

        assert "must be positive" in str(exc_info.value)


# =============================================================================
# Tests for _validate_config_values
# =============================================================================


class TestValidateConfigValues:
    """Tests for configuration value validation."""

    def test_valid_config_returns_empty_errors(self):
        """Valid config produces no errors."""
        config = {'models_to_analyze': 15, 'fig_width': 16, 'fig_height': 10}
        errors = _validate_config_values(config)

        assert errors == []

    def test_invalid_models_to_analyze(self):
        """Catches invalid models_to_analyze values."""
        # Zero
        errors = _validate_config_values({'models_to_analyze': 0})
        assert any("models_to_analyze" in e for e in errors)

        # Negative
        errors = _validate_config_values({'models_to_analyze': -5})
        assert any("models_to_analyze" in e for e in errors)

        # Too large
        errors = _validate_config_values({'models_to_analyze': 51})
        assert any("models_to_analyze" in e for e in errors)

        # Wrong type
        errors = _validate_config_values({'models_to_analyze': 15.5})
        assert any("models_to_analyze" in e for e in errors)

    def test_invalid_fig_width(self):
        """Catches invalid fig_width values."""
        # Zero
        errors = _validate_config_values({'fig_width': 0})
        assert any("fig_width" in e for e in errors)

        # Too large
        errors = _validate_config_values({'fig_width': 35})
        assert any("fig_width" in e for e in errors)

    def test_invalid_fig_height(self):
        """Catches invalid fig_height values."""
        # Zero
        errors = _validate_config_values({'fig_height': 0})
        assert any("fig_height" in e for e in errors)

        # Too large
        errors = _validate_config_values({'fig_height': 25})
        assert any("fig_height" in e for e in errors)

    def test_invalid_aspect_ratio(self):
        """Catches unreasonable aspect ratios."""
        # Too narrow (ratio < 0.5)
        config = {'fig_width': 4, 'fig_height': 10}  # ratio = 0.4
        errors = _validate_config_values(config)
        assert any("aspect ratio" in e for e in errors)

        # Too wide (ratio > 4.0)
        config = {'fig_width': 20, 'fig_height': 4}  # ratio = 5.0
        errors = _validate_config_values(config)
        assert any("aspect ratio" in e for e in errors)

    def test_accumulates_multiple_errors(self):
        """Returns all errors, not just first."""
        config = {
            'models_to_analyze': -5,
            'fig_width': 0,
            'fig_height': 100,
        }
        errors = _validate_config_values(config)

        # Should have at least 3 errors
        assert len(errors) >= 3


# =============================================================================
# Tests for validate_visualization_config
# =============================================================================


class TestValidateVisualizationConfig:
    """Tests for comprehensive config validation."""

    def test_valid_config_passes(self):
        """Complete valid config passes validation."""
        config = {'models_to_analyze': 15, 'fig_width': 16, 'fig_height': 10}
        is_valid, errors = validate_visualization_config(config)

        assert is_valid is True
        assert errors == []

    def test_missing_required_key_fails(self):
        """Missing required keys produce errors."""
        config = {'models_to_analyze': 15, 'fig_width': 16}  # Missing fig_height
        is_valid, errors = validate_visualization_config(config)

        assert is_valid is False
        assert any("fig_height" in e for e in errors)

    def test_invalid_values_fail(self):
        """Invalid values produce errors."""
        config = {'models_to_analyze': -5, 'fig_width': 16, 'fig_height': 10}
        is_valid, errors = validate_visualization_config(config)

        assert is_valid is False
        assert any("models_to_analyze" in e for e in errors)

    def test_non_dict_raises_error(self):
        """Non-dictionary input raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_visualization_config("not a dict")

        assert "must be dictionary" in str(exc_info.value)

    def test_empty_config_fails(self):
        """Empty config fails with missing key errors."""
        is_valid, errors = validate_visualization_config({})

        assert is_valid is False
        assert len(errors) == 3  # All 3 required keys missing

    def test_partial_config_reports_all_missing(self):
        """Partial config reports all missing keys."""
        config = {'fig_width': 16}  # Missing models_to_analyze and fig_height
        is_valid, errors = validate_visualization_config(config)

        assert is_valid is False
        missing_keys = [e for e in errors if "Missing" in e]
        assert len(missing_keys) == 2


# =============================================================================
# Tests for Module Constants
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constant definitions."""

    def test_default_colors_complete(self):
        """Default colors has all required entries."""
        expected_keys = {'stable', 'moderate', 'unstable', 'default', 'median', 'background'}
        assert set(_DEFAULT_COLORS.keys()) == expected_keys

    def test_color_values_are_valid_hex(self):
        """All color values are valid hex color codes."""
        for key, value in _DEFAULT_COLORS.items():
            assert value.startswith('#'), f"{key} should start with #"
            assert len(value) == 7, f"{key} should be 7 chars (#RRGGBB)"

    def test_default_fonts_structure(self):
        """Default fonts has size and weight parameters."""
        assert 'title_size' in _DEFAULT_FONTS
        assert 'label_size' in _DEFAULT_FONTS
        assert 'title_weight' in _DEFAULT_FONTS
        assert all(isinstance(v, int) for k, v in _DEFAULT_FONTS.items() if 'size' in k)

    def test_default_grid_structure(self):
        """Default grid has alpha and visibility flags."""
        assert isinstance(_DEFAULT_GRID['alpha'], float)
        assert isinstance(_DEFAULT_GRID['x_grid'], bool)
        assert isinstance(_DEFAULT_GRID['y_grid'], bool)

    def test_default_spines_all_sides(self):
        """Default spines covers all four sides."""
        required_sides = {'top', 'right', 'left', 'bottom'}
        assert required_sides.issubset(set(_DEFAULT_SPINES.keys()))

    def test_default_business_elements(self):
        """Default business elements has required text settings."""
        assert 'subtitle_suffix' in _DEFAULT_BUSINESS
        assert 'axis_label_style' in _DEFAULT_BUSINESS
        assert 'stability_legend' in _DEFAULT_BUSINESS
