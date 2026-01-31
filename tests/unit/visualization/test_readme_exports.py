"""
Tests for src.visualization.readme_exports module.

Tests README plot export utilities:
- export_for_readme
- export_business_intelligence_plots
- export_model_performance_plots
- export_data_pipeline_plots
- create_readme_plot_catalog
- validate_plot_exports
- apply_readme_styling

Target coverage: 80%+
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call
import datetime

import matplotlib.pyplot as plt
import pytest

from src.visualization.readme_exports import (
    README_COLORS,
    apply_readme_styling,
    create_readme_plot_catalog,
    export_business_intelligence_plots,
    export_data_pipeline_plots,
    export_for_readme,
    export_model_performance_plots,
    validate_plot_exports,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_figure():
    """Create a mock matplotlib figure."""
    fig = MagicMock(spec=plt.Figure)
    fig.axes = []
    return fig


@pytest.fixture
def mock_figure_with_axes():
    """Create a mock figure with axes that have labels."""
    fig = MagicMock(spec=plt.Figure)
    ax = MagicMock()
    ax.get_xlabel.return_value = 'X Label'
    ax.get_ylabel.return_value = 'Y Label'
    ax.spines = {
        'top': MagicMock(),
        'bottom': MagicMock(),
        'left': MagicMock(),
        'right': MagicMock()
    }
    fig.axes = [ax]
    return fig


@pytest.fixture
def mock_path_exists(tmp_path):
    """Create a temporary directory structure for testing."""
    # Create the expected directory structure
    images_dir = tmp_path / "docs" / "images"
    images_dir.mkdir(parents=True)
    return images_dir


# =============================================================================
# EXPORT_FOR_README TESTS
# =============================================================================


class TestExportForReadme:
    """Tests for export_for_readme function."""

    @patch('src.visualization.readme_exports.Path')
    @patch('src.visualization.readme_exports.datetime')
    def test_basic_export(self, mock_datetime, mock_path_class, mock_figure):
        """Test basic plot export."""
        # Setup mocks
        mock_datetime.datetime.now.return_value.strftime.return_value = '20260130'
        mock_path_instance = MagicMock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.__truediv__ = lambda self, x: MagicMock()

        result = export_for_readme(
            mock_figure,
            category='business_intelligence',
            plot_name='test_plot',
            description='Test description'
        )

        # Verify savefig was called twice (versioned and latest)
        assert mock_figure.savefig.call_count == 2

    @patch('src.visualization.readme_exports.Path')
    @patch('src.visualization.readme_exports.datetime')
    def test_export_without_description(self, mock_datetime, mock_path_class, mock_figure):
        """Test export without description."""
        mock_datetime.datetime.now.return_value.strftime.return_value = '20260130'
        mock_path_instance = MagicMock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.__truediv__ = lambda self, x: MagicMock()

        result = export_for_readme(
            mock_figure,
            category='model_performance',
            plot_name='test_plot'
        )

        # Should not crash without description
        assert mock_figure.savefig.call_count == 2

    @patch('src.visualization.readme_exports.Path')
    @patch('src.visualization.readme_exports.datetime')
    def test_export_custom_version(self, mock_datetime, mock_path_class, mock_figure):
        """Test export with custom version."""
        mock_datetime.datetime.now.return_value.strftime.return_value = '20260130'
        mock_path_instance = MagicMock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.__truediv__ = lambda self, x: MagicMock()

        result = export_for_readme(
            mock_figure,
            category='data_pipeline',
            plot_name='test_plot',
            version='v7'
        )

        # Verify savefig was called
        assert mock_figure.savefig.called

    @patch('src.visualization.readme_exports.Path')
    def test_export_creates_directory(self, mock_path_class, mock_figure):
        """Test that directory is created if it doesn't exist."""
        mock_path_instance = MagicMock()
        mock_category_dir = MagicMock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.__truediv__ = lambda self, x: mock_category_dir
        mock_category_dir.__truediv__ = lambda self, x: MagicMock()

        export_for_readme(mock_figure, 'test_category', 'test_plot')

        # Verify mkdir was called
        mock_category_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


# =============================================================================
# EXPORT_BUSINESS_INTELLIGENCE_PLOTS TESTS
# =============================================================================


class TestExportBusinessIntelligencePlots:
    """Tests for export_business_intelligence_plots function."""

    @patch('src.visualization.readme_exports.export_for_readme')
    def test_exports_both_figures(self, mock_export, mock_figure):
        """Test that both percentage and dollar figures are exported."""
        fig_pct = MagicMock(spec=plt.Figure)
        fig_dollars = MagicMock(spec=plt.Figure)

        export_business_intelligence_plots(fig_pct, fig_dollars)

        # Should call export_for_readme twice
        assert mock_export.call_count == 2

    @patch('src.visualization.readme_exports.export_for_readme')
    def test_exports_correct_categories(self, mock_export, mock_figure):
        """Test exports use correct category names."""
        fig_pct = MagicMock(spec=plt.Figure)
        fig_dollars = MagicMock(spec=plt.Figure)

        export_business_intelligence_plots(fig_pct, fig_dollars)

        # Both should be business_intelligence category
        calls = mock_export.call_args_list
        assert calls[0][0][1] == 'business_intelligence'
        assert calls[1][0][1] == 'business_intelligence'

    @patch('src.visualization.readme_exports.export_for_readme')
    def test_exports_correct_plot_names(self, mock_export, mock_figure):
        """Test exports use correct plot names."""
        fig_pct = MagicMock(spec=plt.Figure)
        fig_dollars = MagicMock(spec=plt.Figure)

        export_business_intelligence_plots(fig_pct, fig_dollars)

        calls = mock_export.call_args_list
        assert 'pct' in calls[0][0][2]
        assert 'dollars' in calls[1][0][2]


# =============================================================================
# EXPORT_MODEL_PERFORMANCE_PLOTS TESTS
# =============================================================================


class TestExportModelPerformancePlots:
    """Tests for export_model_performance_plots function."""

    @patch('src.visualization.readme_exports.export_for_readme')
    def test_exports_comprehensive_only(self, mock_export, mock_figure):
        """Test export with only comprehensive figure."""
        comprehensive_fig = MagicMock(spec=plt.Figure)

        export_model_performance_plots(comprehensive_fig)

        assert mock_export.call_count == 1

    @patch('src.visualization.readme_exports.export_for_readme')
    def test_exports_both_figures(self, mock_export, mock_figure):
        """Test export with both figures."""
        comprehensive_fig = MagicMock(spec=plt.Figure)
        bootstrap_fig = MagicMock(spec=plt.Figure)

        export_model_performance_plots(comprehensive_fig, bootstrap_fig)

        assert mock_export.call_count == 2

    @patch('src.visualization.readme_exports.export_for_readme')
    def test_exports_correct_category(self, mock_export, mock_figure):
        """Test exports use model_performance category."""
        comprehensive_fig = MagicMock(spec=plt.Figure)

        export_model_performance_plots(comprehensive_fig)

        calls = mock_export.call_args_list
        assert calls[0][0][1] == 'model_performance'


# =============================================================================
# EXPORT_DATA_PIPELINE_PLOTS TESTS
# =============================================================================


class TestExportDataPipelinePlots:
    """Tests for export_data_pipeline_plots function."""

    @patch('src.visualization.readme_exports.export_for_readme')
    def test_exports_sales_only(self, mock_export, mock_figure):
        """Test export with only sales figure."""
        sales_fig = MagicMock(spec=plt.Figure)

        export_data_pipeline_plots(sales_fig)

        assert mock_export.call_count == 1

    @patch('src.visualization.readme_exports.export_for_readme')
    def test_exports_both_figures(self, mock_export, mock_figure):
        """Test export with both figures."""
        sales_fig = MagicMock(spec=plt.Figure)
        quality_fig = MagicMock(spec=plt.Figure)

        export_data_pipeline_plots(sales_fig, quality_fig)

        assert mock_export.call_count == 2

    @patch('src.visualization.readme_exports.export_for_readme')
    def test_exports_correct_category(self, mock_export, mock_figure):
        """Test exports use data_pipeline category."""
        sales_fig = MagicMock(spec=plt.Figure)

        export_data_pipeline_plots(sales_fig)

        calls = mock_export.call_args_list
        assert calls[0][0][1] == 'data_pipeline'


# =============================================================================
# CREATE_README_PLOT_CATALOG TESTS
# =============================================================================


class TestCreateReadmePlotCatalog:
    """Tests for create_readme_plot_catalog function."""

    @patch('src.visualization.readme_exports.Path')
    def test_catalog_no_directory(self, mock_path_class):
        """Test catalog when directory doesn't exist."""
        mock_path_instance = MagicMock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.exists.return_value = False

        result = create_readme_plot_catalog()

        assert result == "No README plots available yet."

    @patch('src.visualization.readme_exports.Path')
    def test_catalog_with_plots(self, mock_path_class):
        """Test catalog with existing plots."""
        mock_base_dir = MagicMock()
        mock_path_class.return_value = mock_base_dir
        mock_base_dir.exists.return_value = True

        # Create mock category directory with plots
        mock_category_dir = MagicMock()
        mock_base_dir.__truediv__ = lambda self, x: mock_category_dir
        mock_category_dir.exists.return_value = True

        # Create mock plot paths
        mock_plot = MagicMock()
        mock_plot.stem = 'business_intelligence_test_plot_latest'
        mock_plot.name = 'business_intelligence_test_plot_latest.png'
        mock_category_dir.glob.return_value = [mock_plot]

        result = create_readme_plot_catalog()

        assert '# README Plot Catalog' in result
        assert '## Usage in README' in result

    @patch('src.visualization.readme_exports.Path')
    def test_catalog_empty_categories(self, mock_path_class):
        """Test catalog when categories exist but have no plots."""
        mock_base_dir = MagicMock()
        mock_path_class.return_value = mock_base_dir
        mock_base_dir.exists.return_value = True

        mock_category_dir = MagicMock()
        mock_base_dir.__truediv__ = lambda self, x: mock_category_dir
        mock_category_dir.exists.return_value = True
        mock_category_dir.glob.return_value = []  # No plots

        result = create_readme_plot_catalog()

        assert '# README Plot Catalog' in result


# =============================================================================
# VALIDATE_PLOT_EXPORTS TESTS
# =============================================================================


class TestValidatePlotExports:
    """Tests for validate_plot_exports function."""

    @patch('src.visualization.readme_exports.Path')
    def test_all_plots_exist(self, mock_path_class):
        """Test validation when all critical plots exist."""
        mock_base_dir = MagicMock()
        mock_path_class.return_value = mock_base_dir

        mock_full_path = MagicMock()
        mock_base_dir.__truediv__ = lambda self, x: mock_full_path
        mock_full_path.exists.return_value = True

        result = validate_plot_exports()

        assert result is True

    @patch('src.visualization.readme_exports.Path')
    def test_missing_plots(self, mock_path_class):
        """Test validation when plots are missing."""
        mock_base_dir = MagicMock()
        mock_path_class.return_value = mock_base_dir

        mock_full_path = MagicMock()
        mock_base_dir.__truediv__ = lambda self, x: mock_full_path
        mock_full_path.exists.return_value = False

        result = validate_plot_exports()

        assert result is False


# =============================================================================
# APPLY_README_STYLING TESTS
# =============================================================================


class TestApplyReadmeStyling:
    """Tests for apply_readme_styling function."""

    @patch('src.visualization.readme_exports.plt')
    def test_styling_without_title(self, mock_plt, mock_figure_with_axes):
        """Test styling without title."""
        result = apply_readme_styling(mock_figure_with_axes)

        # Should not call suptitle without title
        mock_figure_with_axes.suptitle.assert_not_called()
        # Should return the figure
        assert result == mock_figure_with_axes

    @patch('src.visualization.readme_exports.plt')
    def test_styling_with_title(self, mock_plt, mock_figure_with_axes):
        """Test styling with title."""
        result = apply_readme_styling(mock_figure_with_axes, title='Test Title')

        # Should call suptitle
        mock_figure_with_axes.suptitle.assert_called_once()
        call_args = mock_figure_with_axes.suptitle.call_args
        assert call_args[0][0] == 'Test Title'

    @patch('src.visualization.readme_exports.plt')
    def test_styling_applies_grid(self, mock_plt, mock_figure_with_axes):
        """Test that grid is applied to axes."""
        apply_readme_styling(mock_figure_with_axes)

        ax = mock_figure_with_axes.axes[0]
        ax.grid.assert_called_once_with(True, alpha=0.3, linewidth=0.5)

    @patch('src.visualization.readme_exports.plt')
    def test_styling_axes_below(self, mock_plt, mock_figure_with_axes):
        """Test that axes are set below."""
        apply_readme_styling(mock_figure_with_axes)

        ax = mock_figure_with_axes.axes[0]
        ax.set_axisbelow.assert_called_once_with(True)

    @patch('src.visualization.readme_exports.plt')
    def test_styling_spines(self, mock_plt, mock_figure_with_axes):
        """Test that spines are styled."""
        apply_readme_styling(mock_figure_with_axes)

        ax = mock_figure_with_axes.axes[0]
        # Each spine should have set_linewidth and set_color called
        for spine in ax.spines.values():
            spine.set_linewidth.assert_called_once_with(0.5)
            spine.set_color.assert_called_once_with('#333333')

    @patch('src.visualization.readme_exports.plt')
    def test_styling_tick_params(self, mock_plt, mock_figure_with_axes):
        """Test that tick params are set."""
        apply_readme_styling(mock_figure_with_axes)

        ax = mock_figure_with_axes.axes[0]
        ax.tick_params.assert_called_once_with(labelsize=10, colors='#333333')

    @patch('src.visualization.readme_exports.plt')
    def test_styling_labels(self, mock_plt, mock_figure_with_axes):
        """Test that labels are styled."""
        apply_readme_styling(mock_figure_with_axes)

        ax = mock_figure_with_axes.axes[0]
        # Should update labels with bold font
        ax.set_xlabel.assert_called_once()
        ax.set_ylabel.assert_called_once()

    @patch('src.visualization.readme_exports.plt')
    def test_styling_no_labels(self, mock_plt):
        """Test styling when axes have no labels."""
        fig = MagicMock(spec=plt.Figure)
        ax = MagicMock()
        ax.get_xlabel.return_value = ''  # Empty label
        ax.get_ylabel.return_value = ''
        ax.spines = {
            'top': MagicMock(),
            'bottom': MagicMock(),
            'left': MagicMock(),
            'right': MagicMock()
        }
        fig.axes = [ax]

        apply_readme_styling(fig)

        # Should not set labels when they're empty
        ax.set_xlabel.assert_not_called()
        ax.set_ylabel.assert_not_called()

    @patch('src.visualization.readme_exports.plt')
    def test_styling_tight_layout(self, mock_plt, mock_figure_with_axes):
        """Test that tight_layout is called."""
        apply_readme_styling(mock_figure_with_axes)

        mock_plt.tight_layout.assert_called_once()


# =============================================================================
# README_COLORS TESTS
# =============================================================================


class TestReadmeColors:
    """Tests for README_COLORS constant."""

    def test_colors_exist(self):
        """Test that all expected colors exist."""
        expected_colors = [
            'primary', 'secondary', 'success', 'warning',
            'info', 'neutral', 'light', 'accent'
        ]
        for color_name in expected_colors:
            assert color_name in README_COLORS

    def test_colors_are_hex(self):
        """Test that all colors are valid hex codes."""
        for color_name, color_value in README_COLORS.items():
            assert color_value.startswith('#')
            assert len(color_value) == 7  # #RRGGBB format

    def test_primary_color(self):
        """Test primary color is professional blue."""
        assert README_COLORS['primary'] == '#1f77b4'
