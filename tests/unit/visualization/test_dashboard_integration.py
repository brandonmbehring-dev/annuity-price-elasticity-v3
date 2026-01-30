"""
Unit tests for src/visualization/dashboard_integration.py.

Tests validate the DashboardIntegration class and related functions
for generating integrated visualization dashboards and reports.

Target: 60%+ coverage for dashboard_integration.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_aic_results():
    """Sample AIC results DataFrame for testing."""
    np.random.seed(42)
    n_models = 50
    return pd.DataFrame({
        'model_id': range(n_models),
        'aic': np.random.uniform(500, 700, n_models),
        'r_squared': np.random.uniform(0.3, 0.9, n_models),
        'n_features': np.random.randint(2, 8, n_models),
        'economically_valid': np.random.choice([True, False], n_models, p=[0.6, 0.4])
    })


@pytest.fixture
def sample_final_model():
    """Sample final model selection."""
    return {
        'model_id': 10,
        'aic': 525.5,
        'r_squared': 0.82,
        'n_features': 4,
        'features': ['prudential_rate_t0', 'competitor_weighted_t2'],
        'selected_model': {
            'features': 'prudential_rate_t0, competitor_weighted_t2',
            'r_squared': 0.82
        }
    }


@pytest.fixture
def sample_analysis_results(sample_aic_results, sample_final_model):
    """Complete analysis results dictionary."""
    return {
        'aic_results': sample_aic_results,
        'final_model': sample_final_model,
        'metadata': {
            'analysis_id': 'TEST_001',
            'dataset_info': {
                'total_observations': 1000
            }
        }
    }


@pytest.fixture
def empty_analysis_results():
    """Empty analysis results for edge case testing."""
    return {
        'aic_results': pd.DataFrame(),
        'final_model': None,
        'metadata': {}
    }


# =============================================================================
# CLASS INITIALIZATION TESTS
# =============================================================================


class TestDashboardIntegrationInit:
    """Tests for DashboardIntegration class initialization."""

    def test_init_default_branding(self):
        """Should initialize with default branding."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()

        assert 'primary_color' in dashboard.branding
        assert 'secondary_color' in dashboard.branding
        assert 'accent_color' in dashboard.branding
        assert dashboard.branding['logo_text'] == 'Feature Selection V2'

    def test_init_custom_branding(self):
        """Should accept custom branding configuration."""
        from src.visualization.dashboard_integration import DashboardIntegration

        custom_branding = {
            'primary_color': '#FF0000',
            'secondary_color': '#00FF00',
            'accent_color': '#0000FF',
            'logo_text': 'Custom Logo',
            'subtitle': 'Custom Subtitle',
            'footer_text': 'Custom Footer'
        }

        dashboard = DashboardIntegration(company_branding=custom_branding)

        assert dashboard.branding['primary_color'] == '#FF0000'
        assert dashboard.branding['logo_text'] == 'Custom Logo'

    def test_init_sets_dashboard_config(self):
        """Should initialize dashboard configuration."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()

        assert 'master_figure_size' in dashboard.dashboard_config
        assert 'section_figure_size' in dashboard.dashboard_config
        assert 'export_dpi' in dashboard.dashboard_config
        assert dashboard.dashboard_config['export_dpi'] == 300

    @patch('src.visualization.dashboard_integration.StatisticalValidationPlots')
    @patch('src.visualization.dashboard_integration.BusinessCommunicationPlots')
    @patch('src.visualization.dashboard_integration.ModelComparisonPlots')
    def test_init_creates_plot_modules(self, mock_model, mock_business, mock_stat):
        """Should initialize visualization module instances."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()

        mock_stat.assert_called_once()
        mock_business.assert_called_once()
        mock_model.assert_called_once()


# =============================================================================
# HELPER METHOD TESTS
# =============================================================================


class TestCreateExecutiveSummaryText:
    """Tests for _create_executive_summary_text helper method."""

    def test_creates_summary_with_final_model(self, sample_analysis_results):
        """Should create summary when final model exists."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        summary = dashboard._create_executive_summary_text(sample_analysis_results)

        assert 'EXECUTIVE SUMMARY' in summary
        assert 'Selected Model' in summary or 'Status' in summary

    def test_creates_summary_without_final_model(self, empty_analysis_results):
        """Should create summary when no final model."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        summary = dashboard._create_executive_summary_text(empty_analysis_results)

        assert 'EXECUTIVE SUMMARY' in summary
        assert 'No suitable model' in summary or 'Status' in summary


class TestCreateKpiSection:
    """Tests for _create_kpi_section helper method."""

    def test_creates_kpi_with_results(self, sample_analysis_results):
        """Should populate KPI section with results."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        mock_ax = MagicMock()

        dashboard._create_kpi_section(mock_ax, sample_analysis_results)

        # Should turn off axis and add text
        mock_ax.axis.assert_called_with('off')
        mock_ax.text.assert_called()

    def test_creates_kpi_with_empty_results(self, empty_analysis_results):
        """Should handle empty results gracefully."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        mock_ax = MagicMock()

        dashboard._create_kpi_section(mock_ax, empty_analysis_results)

        mock_ax.axis.assert_called_with('off')
        mock_ax.text.assert_called()


class TestCreateModelSelectionOverview:
    """Tests for _create_model_selection_overview helper method."""

    def test_creates_bar_chart_with_data(self, sample_aic_results, sample_final_model):
        """Should create bar chart when data exists."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        mock_ax = MagicMock()
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 50
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar, mock_bar]

        dashboard._create_model_selection_overview(mock_ax, sample_aic_results, sample_final_model)

        mock_ax.set_title.assert_called_once()
        mock_ax.bar.assert_called_once()

    def test_shows_placeholder_without_data(self):
        """Should show placeholder when no data."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        mock_ax = MagicMock()

        dashboard._create_model_selection_overview(mock_ax, pd.DataFrame(), None)

        mock_ax.text.assert_called()


class TestCreateRiskAssessmentOverview:
    """Tests for _create_risk_assessment_overview helper method."""

    def test_creates_risk_bars(self, sample_analysis_results):
        """Should create risk assessment bar chart."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        mock_ax = MagicMock()
        mock_ax.bar.return_value = [MagicMock(), MagicMock(), MagicMock()]

        dashboard._create_risk_assessment_overview(mock_ax, sample_analysis_results)

        mock_ax.set_title.assert_called_once_with('Risk Assessment', fontweight='bold')
        mock_ax.bar.assert_called_once()


class TestGetAppendixText:
    """Tests for _get_appendix_text helper method."""

    def test_returns_methodology_text(self):
        """Should return technical appendix text."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        text = dashboard._get_appendix_text()

        assert 'TECHNICAL APPENDIX' in text
        assert 'Bootstrap' in text
        assert 'Information Criteria' in text


# =============================================================================
# DASHBOARD CREATION TESTS
# =============================================================================


class TestSetupDashboardFigure:
    """Tests for _setup_dashboard_figure method."""

    @patch('src.visualization.dashboard_integration.plt')
    def test_returns_figure_and_gridspec(self, mock_plt):
        """Should return figure and GridSpec tuple."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig

        dashboard = DashboardIntegration()
        fig, gs = dashboard._setup_dashboard_figure()

        assert fig == mock_fig
        assert gs is not None


class TestAddDashboardTitles:
    """Tests for _add_dashboard_titles method."""

    def test_adds_title_and_subtitle(self):
        """Should add suptitle and subtitle text."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        mock_fig = MagicMock()

        dashboard._add_dashboard_titles(mock_fig)

        mock_fig.suptitle.assert_called_once()
        mock_fig.text.assert_called_once()


class TestCreateMasterDashboard:
    """Tests for create_master_dashboard method."""

    @patch('src.visualization.dashboard_integration.plt')
    def test_returns_figure(self, mock_plt, sample_analysis_results):
        """Should return matplotlib Figure."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 50
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar, mock_bar]
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig

        dashboard = DashboardIntegration()
        result = dashboard.create_master_dashboard(sample_analysis_results)

        assert result == mock_fig

    @patch('src.visualization.dashboard_integration.plt')
    def test_saves_to_path_when_provided(self, mock_plt, sample_analysis_results, tmp_path):
        """Should save figure when save_path provided."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 50
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar, mock_bar]
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig

        dashboard = DashboardIntegration()
        save_path = tmp_path / "test_dashboard.png"
        dashboard.create_master_dashboard(sample_analysis_results, save_path=save_path)

        mock_fig.savefig.assert_called_once()


class TestCreateAnalystSummaryDashboard:
    """Tests for create_analyst_summary_dashboard method."""

    @patch('src.visualization.dashboard_integration.plt')
    def test_returns_figure(self, mock_plt, sample_analysis_results):
        """Should return matplotlib Figure."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_fig = MagicMock()
        mock_axes = [[MagicMock() for _ in range(3)] for _ in range(2)]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        dashboard = DashboardIntegration()
        result = dashboard.create_analyst_summary_dashboard(sample_analysis_results)

        assert result == mock_fig


class TestCreateTechnicalDeepDive:
    """Tests for create_technical_deep_dive method."""

    @patch('src.visualization.dashboard_integration.plt')
    def test_returns_figure(self, mock_plt, sample_analysis_results):
        """Should return matplotlib Figure."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig

        dashboard = DashboardIntegration()
        result = dashboard.create_technical_deep_dive(sample_analysis_results)

        assert result == mock_fig


class TestCreateCoverPage:
    """Tests for create_cover_page method."""

    @patch('src.visualization.dashboard_integration.plt')
    def test_returns_figure(self, mock_plt, sample_analysis_results):
        """Should return matplotlib Figure for cover page."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        dashboard = DashboardIntegration()
        result = dashboard.create_cover_page(sample_analysis_results)

        assert result == mock_fig
        mock_ax.axis.assert_called_with('off')


class TestCreateTechnicalAppendix:
    """Tests for create_technical_appendix method."""

    @patch('src.visualization.dashboard_integration.plt')
    def test_returns_figure(self, mock_plt, sample_analysis_results):
        """Should return matplotlib Figure for appendix."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        dashboard = DashboardIntegration()
        result = dashboard.create_technical_appendix(sample_analysis_results)

        assert result == mock_fig


# =============================================================================
# STAKEHOLDER REPORT TESTS
# =============================================================================


class TestGenerateStakeholderReports:
    """Tests for generate_stakeholder_reports method."""

    @patch('src.visualization.dashboard_integration.create_business_communication_report')
    @patch('src.visualization.dashboard_integration.plt')
    def test_returns_dict_of_reports(self, mock_plt, mock_business_report, sample_analysis_results, tmp_path):
        """Should return dictionary mapping stakeholder types to paths."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_business_report.return_value = {'report1': tmp_path / 'report1.png'}
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 50
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar, mock_bar]
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        dashboard = DashboardIntegration()
        result = dashboard.generate_stakeholder_reports(
            sample_analysis_results,
            tmp_path,
            stakeholder_types=['executive']
        )

        assert isinstance(result, dict)
        assert 'executive' in result

    @patch('src.visualization.dashboard_integration.plt')
    def test_creates_output_directories(self, mock_plt, sample_analysis_results, tmp_path):
        """Should create output directories for each stakeholder."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        dashboard = DashboardIntegration()

        # Even if generation fails, directories should be created
        dashboard.generate_stakeholder_reports(
            sample_analysis_results,
            tmp_path,
            stakeholder_types=['executive']
        )

        assert (tmp_path / 'executive_report').exists()


# =============================================================================
# PDF EXPORT TESTS
# =============================================================================


class TestExportComprehensivePdfReport:
    """Tests for export_comprehensive_pdf_report method."""

    @patch('src.visualization.dashboard_integration.PdfPages')
    @patch('src.visualization.dashboard_integration.plt')
    def test_creates_pdf_file(self, mock_plt, mock_pdf_pages, sample_analysis_results, tmp_path):
        """Should create PDF at specified path."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 50
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar, mock_bar]
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        mock_pdf = MagicMock()
        mock_pdf.infodict.return_value = {}
        mock_pdf_pages.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf_pages.return_value.__exit__ = MagicMock(return_value=False)

        dashboard = DashboardIntegration()
        pdf_path = tmp_path / "test_report.pdf"

        result = dashboard.export_comprehensive_pdf_report(
            sample_analysis_results,
            pdf_path,
            include_sections=['master']
        )

        assert result == pdf_path
        mock_pdf_pages.assert_called_once_with(pdf_path)


# =============================================================================
# STANDALONE FUNCTION TESTS
# =============================================================================


class TestCreateIntegratedDashboardReport:
    """Tests for create_integrated_dashboard_report standalone function."""

    @patch('src.visualization.dashboard_integration.DashboardIntegration')
    def test_returns_dict_of_paths(self, mock_dashboard_class, sample_analysis_results, tmp_path):
        """Should return dictionary of report file paths."""
        from src.visualization.dashboard_integration import create_integrated_dashboard_report

        mock_dashboard = MagicMock()
        mock_fig = MagicMock()
        mock_dashboard.create_master_dashboard.return_value = mock_fig
        mock_dashboard_class.return_value = mock_dashboard

        result = create_integrated_dashboard_report(
            sample_analysis_results,
            tmp_path,
            report_types=['master_dashboard']
        )

        assert isinstance(result, dict)

    @patch('src.visualization.dashboard_integration.DashboardIntegration')
    def test_creates_output_directory(self, mock_dashboard_class, sample_analysis_results, tmp_path):
        """Should create output directory if not exists."""
        from src.visualization.dashboard_integration import create_integrated_dashboard_report

        mock_dashboard = MagicMock()
        mock_dashboard_class.return_value = mock_dashboard

        output_dir = tmp_path / "new_output_dir"
        create_integrated_dashboard_report(
            sample_analysis_results,
            output_dir,
            report_types=[]
        )

        assert output_dir.exists()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_aic_results_handled(self, empty_analysis_results):
        """Should handle empty AIC results gracefully."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        summary = dashboard._create_executive_summary_text(empty_analysis_results)

        # Should not raise and should produce some output
        assert summary is not None
        assert len(summary) > 0

    def test_missing_metadata_handled(self):
        """Should handle missing metadata."""
        from src.visualization.dashboard_integration import DashboardIntegration

        dashboard = DashboardIntegration()
        analysis_results = {'aic_results': pd.DataFrame(), 'final_model': None}

        summary = dashboard._create_executive_summary_text(analysis_results)

        assert 'EXECUTIVE SUMMARY' in summary

    @patch('src.visualization.dashboard_integration.plt')
    def test_generate_reports_handles_errors(self, mock_plt, sample_analysis_results, tmp_path):
        """Should handle errors in report generation gracefully."""
        from src.visualization.dashboard_integration import DashboardIntegration

        mock_plt.figure.side_effect = Exception("Plot error")

        dashboard = DashboardIntegration()
        result = dashboard.generate_stakeholder_reports(
            sample_analysis_results,
            tmp_path,
            stakeholder_types=['executive']
        )

        # Should return dict even on error
        assert isinstance(result, dict)
        assert 'executive' in result
        assert result['executive'] == {}  # Empty due to error
