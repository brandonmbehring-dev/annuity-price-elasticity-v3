"""
Unit tests for src/config/visualization_builders.py

Tests violin plot params, line colors, annotation offsets, and
the main visualization config builder.
"""

import pytest


class TestBuildViolinParams:
    """Tests for _build_violin_params function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with all required violin plot keys."""
        from src.config.visualization_builders import _build_violin_params

        params = _build_violin_params()

        assert isinstance(params, dict)
        assert 'hue' in params
        assert 'hue_order' in params
        assert 'split' in params
        assert 'dodge' in params
        assert 'density_norm' in params
        assert 'inner' in params
        assert 'legend' in params

    def test_hue_order_correct(self):
        """Hue order is [True, False] for boolean coloring."""
        from src.config.visualization_builders import _build_violin_params

        params = _build_violin_params()
        assert params['hue_order'] == [True, False]

    def test_density_norm_is_width(self):
        """Density normalization is 'width' for consistent sizing."""
        from src.config.visualization_builders import _build_violin_params

        params = _build_violin_params()
        assert params['density_norm'] == 'width'

    def test_inner_is_quartile(self):
        """Inner display shows quartile markers."""
        from src.config.visualization_builders import _build_violin_params

        params = _build_violin_params()
        assert params['inner'] == 'quartile'


class TestBuildLineColors:
    """Tests for _build_line_colors function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with CI bounds, median, and scatter colors."""
        from src.config.visualization_builders import _build_line_colors

        colors = _build_line_colors()

        assert isinstance(colors, dict)
        assert 'ci_bounds' in colors
        assert 'median' in colors
        assert 'scatter' in colors

    def test_ci_bounds_is_red(self):
        """CI bounds color is red for visibility."""
        from src.config.visualization_builders import _build_line_colors

        colors = _build_line_colors()
        assert colors['ci_bounds'] == 'tab:red'

    def test_median_is_black(self):
        """Median line is black for clarity."""
        from src.config.visualization_builders import _build_line_colors

        colors = _build_line_colors()
        assert colors['median'] == 'k'


class TestBuildAnnotationOffsets:
    """Tests for _build_annotation_offsets function."""

    def test_returns_dict_with_percentage_offsets(self):
        """Returns dict with percentage plot annotation offsets."""
        from src.config.visualization_builders import _build_annotation_offsets

        offsets = _build_annotation_offsets()

        assert 'pct_bottom' in offsets
        assert 'pct_median' in offsets
        assert 'pct_top' in offsets

    def test_returns_dict_with_dollar_offsets(self):
        """Returns dict with dollar plot annotation offsets."""
        from src.config.visualization_builders import _build_annotation_offsets

        offsets = _build_annotation_offsets()

        assert 'dollar_bottom' in offsets
        assert 'dollar_median' in offsets
        assert 'dollar_top' in offsets

    def test_offsets_are_tuples(self):
        """All offsets are tuples (x, y)."""
        from src.config.visualization_builders import _build_annotation_offsets

        offsets = _build_annotation_offsets()

        for key, value in offsets.items():
            assert isinstance(value, tuple), f"{key} should be tuple"
            assert len(value) == 2, f"{key} should have 2 elements"


class TestBuildVisualizationConfig:
    """Tests for build_visualization_config function."""

    def test_returns_visualization_config_type(self):
        """Returns VisualizationConfig TypedDict (dict at runtime)."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()

        assert isinstance(config, dict)

    def test_default_figure_size(self):
        """Default figure size is (10, 15)."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()
        assert config['figure_size'] == (10, 15)

    def test_custom_figure_size(self):
        """Custom figure size is respected."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config(figure_size=(8, 12))
        assert config['figure_size'] == (8, 12)

    def test_default_seaborn_style(self):
        """Default seaborn style is 'whitegrid'."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()
        assert config['seaborn_style'] == 'whitegrid'

    def test_default_seaborn_palette(self):
        """Default seaborn palette is 'deep'."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()
        assert config['seaborn_palette'] == 'deep'

    def test_default_output_directory(self):
        """Default output directory is 'BI_TEAM'."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()
        assert config['output_directory'] == 'BI_TEAM'

    def test_default_dpi(self):
        """Default DPI is 300 for high resolution."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()
        assert config['dpi'] == 300

    def test_custom_dpi(self):
        """Custom DPI is respected."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config(dpi=150)
        assert config['dpi'] == 150

    def test_includes_violin_params(self):
        """Config includes violin plot parameters."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()
        assert 'violin_params' in config
        assert config['violin_params']['inner'] == 'quartile'

    def test_includes_line_colors(self):
        """Config includes line color configuration."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()
        assert 'line_colors' in config
        assert config['line_colors']['ci_bounds'] == 'tab:red'

    def test_includes_annotation_offsets(self):
        """Config includes annotation offset configuration."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()
        assert 'annotation_offsets' in config
        assert 'pct_bottom' in config['annotation_offsets']

    def test_custom_file_prefix(self):
        """Custom file prefix is respected."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config(file_prefix='custom_prefix')
        assert config['file_prefix'] == 'custom_prefix'

    def test_default_file_prefix(self):
        """Default file prefix is 'price_elasticity_FlexGuard'."""
        from src.config.visualization_builders import build_visualization_config

        config = build_visualization_config()
        assert config['file_prefix'] == 'price_elasticity_FlexGuard'
