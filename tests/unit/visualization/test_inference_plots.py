"""
Unit tests for src/visualization/inference_plots.py.

Tests validate data preparation functions and visualization generation
using hybrid approach: test data transforms directly, mock plot creation.

Target: 60%+ coverage for inference_plots.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_bootstrap_data():
    """Sample bootstrap percentage change data (1000 samples x 19 rate scenarios)."""
    np.random.seed(42)
    rate_options = np.linspace(0, 0.045, 19)  # 0 to 4.5%
    data = np.random.normal(0, 5, (1000, 19))  # % changes
    return pd.DataFrame(data, columns=rate_options)


@pytest.fixture
def sample_bootstrap_dollars():
    """Sample bootstrap dollar impact data."""
    np.random.seed(42)
    rate_options = np.linspace(0, 0.045, 19)
    # Dollar impacts ranging from -50M to +50M
    data = np.random.normal(0, 10000000, (1000, 19))
    return pd.DataFrame(data, columns=rate_options)


@pytest.fixture
def sample_output_pct():
    """Sample confidence interval output for percentage changes."""
    return pd.DataFrame({
        'rate_change_in_basis_points': [0, 25, 50, 100, 200],
        'bottom': [-5.2, -4.1, -3.0, -1.5, 1.2],
        'median': [-2.1, -1.5, -0.8, 0.5, 3.5],
        'top': [1.0, 1.1, 1.4, 2.5, 5.8]
    })


@pytest.fixture
def sample_output_dollar():
    """Sample confidence interval output for dollar impacts."""
    return pd.DataFrame({
        'rate_change_in_basis_points': [0, 25, 50, 100, 200],
        'bottom': [-5200000, -4100000, -3000000, -1500000, 1200000],
        'median': [-2100000, -1500000, -800000, 500000, 3500000],
        'top': [1000000, 1100000, 1400000, 2500000, 5800000]
    })


@pytest.fixture
def sample_rate_options():
    """Sample rate options array."""
    return np.linspace(0, 4.5, 19)


@pytest.fixture
def sample_rate_context():
    """Sample rate context for title generation."""
    return {
        'prudential_current': 4.75,
        'prudential_lag': 4.65,
        'competitor_current': 4.25,
        'competitor_lag': 4.15
    }


@pytest.fixture
def sample_viz_config():
    """Sample visualization configuration."""
    return {
        'seaborn_style': 'whitegrid',
        'seaborn_palette': 'deep',
        'figure_size': (12, 8),
        'violin_params': {
            'hue': 'output_type',
            'hue_order': ['pct_change'],
            'split': False,
            'dodge': False,
            'density_norm': 'area',
            'inner': None,
            'legend': False
        },
        'line_colors': {
            'scatter': 'black',
            'ci_bounds': 'red',
            'median': 'blue'
        },
        'annotation_offsets': {
            'pct_bottom': (-10, 0),
            'pct_median': (0, 10),
            'pct_top': (10, 0),
            'dollar_bottom': (-10, 0),
            'dollar_median': (0, 10),
            'dollar_top': (10, 0)
        }
    }


# =============================================================================
# DATA MELTING TESTS
# =============================================================================


class TestMeltBootstrapData:
    """Tests for _melt_bootstrap_data helper function."""

    def test_melt_bootstrap_data_basic(self, sample_bootstrap_data):
        """Melts bootstrap data with output_type and date."""
        from src.visualization.inference_plots import _melt_bootstrap_data

        result = _melt_bootstrap_data(
            sample_bootstrap_data,
            output_type='pct_change',
            current_date='2023-12-15'
        )

        # Should have expected columns
        assert 'simulation_run' in result.columns
        assert 'output_type' in result.columns
        assert 'rate_change_in_basis_points' in result.columns
        assert 'value' in result.columns
        assert 'prediction_date' in result.columns

    def test_melt_bootstrap_data_row_count(self, sample_bootstrap_data):
        """Melted data should have correct row count."""
        from src.visualization.inference_plots import _melt_bootstrap_data

        result = _melt_bootstrap_data(
            sample_bootstrap_data,
            output_type='pct_change',
            current_date='2023-12-15'
        )

        # 1000 samples × 19 rate scenarios = 19000 rows
        expected_rows = 1000 * 19
        assert len(result) == expected_rows

    def test_melt_bootstrap_data_rate_conversion(self, sample_bootstrap_data):
        """Rate values should be converted to basis points."""
        from src.visualization.inference_plots import _melt_bootstrap_data

        result = _melt_bootstrap_data(
            sample_bootstrap_data,
            output_type='pct_change',
            current_date='2023-12-15'
        )

        # Original columns were 0 to 0.045 (decimal)
        # Should be 0 to 4.5 (multiplied by 100)
        rates = result['rate_change_in_basis_points'].unique()
        assert rates.max() == pytest.approx(4.5, rel=0.01)
        assert rates.min() == pytest.approx(0, abs=0.01)

    def test_melt_bootstrap_data_preserves_output_type(self, sample_bootstrap_data):
        """Output type should be preserved in melted data."""
        from src.visualization.inference_plots import _melt_bootstrap_data

        result = _melt_bootstrap_data(
            sample_bootstrap_data,
            output_type='dollars',
            current_date='2023-12-15'
        )

        assert (result['output_type'] == 'dollars').all()


class TestCreateCategoricalLabels:
    """Tests for _create_categorical_labels helper function."""

    def test_create_categorical_labels_format(self):
        """Labels should be formatted as '###bps'."""
        from src.visualization.inference_plots import _create_categorical_labels

        df = pd.DataFrame({
            'rate_change_in_basis_points': [0, 25, 50, 100]
        })

        labels = _create_categorical_labels(df)

        assert labels == ['0bps', '25bps', '50bps', '100bps']

    def test_create_categorical_labels_sorted(self):
        """Labels should be in sorted order."""
        from src.visualization.inference_plots import _create_categorical_labels

        df = pd.DataFrame({
            'rate_change_in_basis_points': [100, 0, 50, 25]  # Unsorted
        })

        labels = _create_categorical_labels(df)

        assert labels == ['0bps', '25bps', '50bps', '100bps']

    def test_create_categorical_labels_negative_rates(self):
        """Should handle negative rate changes."""
        from src.visualization.inference_plots import _create_categorical_labels

        df = pd.DataFrame({
            'rate_change_in_basis_points': [-50, -25, 0, 25, 50]
        })

        labels = _create_categorical_labels(df)

        assert '-50bps' in labels
        assert '-25bps' in labels
        assert labels == ['-50bps', '-25bps', '0bps', '25bps', '50bps']


class TestApplyCategoricalOrdering:
    """Tests for _apply_categorical_ordering helper function."""

    def test_apply_categorical_ordering_converts_to_string(self):
        """Should convert rate values to string with bps suffix."""
        from src.visualization.inference_plots import _apply_categorical_ordering

        df = pd.DataFrame({
            'rate_change_in_basis_points': [0, 25, 50],
            'value': [1, 2, 3]
        })
        x_labels = ['0bps', '25bps', '50bps']

        result = _apply_categorical_ordering(df, x_labels)

        assert result['rate_change_in_basis_points'].dtype.name == 'category'

    def test_apply_categorical_ordering_preserves_other_columns(self):
        """Should not modify other columns."""
        from src.visualization.inference_plots import _apply_categorical_ordering

        df = pd.DataFrame({
            'rate_change_in_basis_points': [0, 25],
            'value': [1.5, 2.5],
            'label': ['a', 'b']
        })
        x_labels = ['0bps', '25bps']

        result = _apply_categorical_ordering(df, x_labels)

        assert result['value'].tolist() == [1.5, 2.5]
        assert result['label'].tolist() == ['a', 'b']

    def test_apply_categorical_ordering_immutable(self):
        """Should not modify original DataFrame."""
        from src.visualization.inference_plots import _apply_categorical_ordering

        df = pd.DataFrame({
            'rate_change_in_basis_points': [0, 25],
            'value': [1, 2]
        })
        df_original = df.copy()
        x_labels = ['0bps', '25bps']

        _ = _apply_categorical_ordering(df, x_labels)

        # Original should be unchanged (int type, not category)
        assert df['rate_change_in_basis_points'].dtype == df_original['rate_change_in_basis_points'].dtype


# =============================================================================
# CONFIDENCE INTERVAL PREPARATION TESTS
# =============================================================================


class TestPrepareConfidenceIntervals:
    """Tests for _prepare_confidence_intervals helper function."""

    def test_prepare_ci_returns_two_dataframes(self, sample_output_pct):
        """Should return tuple of two DataFrames."""
        from src.visualization.inference_plots import _prepare_confidence_intervals

        x_labels = ['0bps', '25bps', '50bps', '100bps', '200bps']
        df_ci, df_ci_columns = _prepare_confidence_intervals(
            sample_output_pct, x_labels, 'pct_change'
        )

        assert isinstance(df_ci, pd.DataFrame)
        assert isinstance(df_ci_columns, pd.DataFrame)

    def test_prepare_ci_df_ci_has_output_type(self, sample_output_pct):
        """df_ci should have output_type column."""
        from src.visualization.inference_plots import _prepare_confidence_intervals

        x_labels = ['0bps', '25bps', '50bps', '100bps', '200bps']
        df_ci, _ = _prepare_confidence_intervals(
            sample_output_pct, x_labels, 'pct_change'
        )

        assert 'output_type' in df_ci.columns
        assert (df_ci['output_type'] == 'pct_change').all()

    def test_prepare_ci_df_ci_columns_has_quantiles(self, sample_output_pct):
        """df_ci_columns should have bottom, median, top columns."""
        from src.visualization.inference_plots import _prepare_confidence_intervals

        x_labels = ['0bps', '25bps', '50bps', '100bps', '200bps']
        _, df_ci_columns = _prepare_confidence_intervals(
            sample_output_pct, x_labels, 'pct_change'
        )

        assert 'bottom' in df_ci_columns.columns
        assert 'median' in df_ci_columns.columns
        assert 'top' in df_ci_columns.columns


# =============================================================================
# VISUALIZATION DATA PREPARATION TESTS
# =============================================================================


class TestPrepareVisualizationDataPct:
    """Tests for prepare_visualization_data_pct function."""

    def test_returns_four_items(self, sample_bootstrap_data, sample_output_pct, sample_rate_options):
        """Should return tuple of 4 items."""
        from src.visualization.inference_plots import prepare_visualization_data_pct

        result = prepare_visualization_data_pct(
            sample_bootstrap_data,
            sample_output_pct,
            sample_rate_options,
            '2023-12-15'
        )

        assert len(result) == 4
        df_dist, df_ci, df_ci_columns, x_labels = result
        assert isinstance(df_dist, pd.DataFrame)
        assert isinstance(df_ci, pd.DataFrame)
        assert isinstance(df_ci_columns, pd.DataFrame)
        assert isinstance(x_labels, list)

    def test_df_dist_has_expected_columns(self, sample_bootstrap_data, sample_output_pct, sample_rate_options):
        """df_dist should have required columns for plotting."""
        from src.visualization.inference_plots import prepare_visualization_data_pct

        df_dist, _, _, _ = prepare_visualization_data_pct(
            sample_bootstrap_data,
            sample_output_pct,
            sample_rate_options,
            '2023-12-15'
        )

        assert 'rate_change_in_basis_points' in df_dist.columns
        assert 'value' in df_dist.columns
        assert 'output_type' in df_dist.columns


class TestPrepareVisualizationDataDollars:
    """Tests for prepare_visualization_data_dollars function."""

    def test_returns_four_items(self, sample_bootstrap_dollars, sample_output_dollar, sample_rate_options):
        """Should return tuple of 4 items."""
        from src.visualization.inference_plots import prepare_visualization_data_dollars

        result = prepare_visualization_data_dollars(
            sample_bootstrap_dollars,
            sample_output_dollar,
            sample_rate_options,
            '2023-12-15'
        )

        assert len(result) == 4

    def test_df_dist_has_dollar_output_type(self, sample_bootstrap_dollars, sample_output_dollar, sample_rate_options):
        """df_dist should have 'dollars' output type."""
        from src.visualization.inference_plots import prepare_visualization_data_dollars

        df_dist, _, _, _ = prepare_visualization_data_dollars(
            sample_bootstrap_dollars,
            sample_output_dollar,
            sample_rate_options,
            '2023-12-15'
        )

        assert (df_dist['output_type'] == 'dollars').all()


# =============================================================================
# RATE CONTEXT EXTRACTION TESTS
# =============================================================================


class TestExtractRateContext:
    """Tests for _extract_rate_context helper function."""

    def test_extracts_all_values(self, sample_rate_context):
        """Should extract all four rate values."""
        from src.visualization.inference_plots import _extract_rate_context

        P, P_lag, C, C_lag = _extract_rate_context(sample_rate_context)

        assert P == pytest.approx(4.75, rel=0.01)
        assert P_lag == pytest.approx(4.65, rel=0.01)
        assert C == pytest.approx(4.25, rel=0.01)
        assert C_lag == pytest.approx(4.15, rel=0.01)

    def test_rounds_to_two_decimals(self):
        """Should round values to 2 decimal places."""
        from src.visualization.inference_plots import _extract_rate_context

        context = {
            'prudential_current': 4.756789,
            'prudential_lag': 4.651234,
            'competitor_current': 4.254567,
            'competitor_lag': 4.148901
        }

        P, P_lag, C, C_lag = _extract_rate_context(context)

        assert P == 4.76  # Rounded from 4.756789
        assert P_lag == 4.65  # Rounded from 4.651234


# =============================================================================
# VISUALIZATION GENERATION TESTS (MOCKED)
# =============================================================================


class TestGeneratePriceElasticityVisualizationPct:
    """Tests for generate_price_elasticity_visualization_pct with mocked matplotlib."""

    @patch('src.visualization.inference_plots.plt')
    @patch('src.visualization.inference_plots.sns')
    def test_returns_figure(self, mock_sns, mock_plt, sample_output_pct, sample_rate_context, sample_viz_config):
        """Should return a matplotlib Figure."""
        from src.visualization.inference_plots import generate_price_elasticity_visualization_pct

        # Setup mocks
        mock_figure = MagicMock()
        mock_axes = MagicMock()
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        df_dist = pd.DataFrame({
            'rate_change_in_basis_points': pd.Categorical(['0bps', '25bps'], ['0bps', '25bps']),
            'value': [1.0, 2.0],
            'output_type': ['pct_change', 'pct_change']
        })
        df_ci = pd.DataFrame({
            'rate_change_in_basis_points': pd.Categorical(['0bps', '25bps'], ['0bps', '25bps']),
            'value': [1.5, 2.5]
        })
        df_ci_columns = pd.DataFrame({
            'rate_change_in_basis_points': pd.Categorical(['0bps', '25bps'], ['0bps', '25bps']),
            'bottom': [-1.0, -0.5],
            'median': [1.5, 2.5],
            'top': [4.0, 5.5]
        })
        x_labels = ['0bps', '25bps']

        result = generate_price_elasticity_visualization_pct(
            df_dist, df_ci, df_ci_columns, x_labels,
            sample_rate_context, '2023-12-15', sample_viz_config
        )

        assert result == mock_figure


class TestGeneratePriceElasticityVisualizationDollars:
    """Tests for generate_price_elasticity_visualization_dollars with mocked matplotlib."""

    @patch('src.visualization.inference_plots.plt')
    @patch('src.visualization.inference_plots.sns')
    def test_returns_figure(self, mock_sns, mock_plt, sample_output_dollar, sample_rate_context, sample_viz_config):
        """Should return a matplotlib Figure."""
        from src.visualization.inference_plots import generate_price_elasticity_visualization_dollars

        # Setup mocks
        mock_figure = MagicMock()
        mock_axes = MagicMock()
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        df_dist = pd.DataFrame({
            'rate_change_in_basis_points': pd.Categorical(['0bps', '25bps'], ['0bps', '25bps']),
            'value': [1000000, 2000000],
            'output_type': ['dollars', 'dollars']
        })
        df_ci = pd.DataFrame({
            'rate_change_in_basis_points': pd.Categorical(['0bps', '25bps'], ['0bps', '25bps']),
            'value': [1500000, 2500000]
        })
        df_ci_columns = pd.DataFrame({
            'rate_change_in_basis_points': pd.Categorical(['0bps', '25bps'], ['0bps', '25bps']),
            'bottom': [-1000000, -500000],
            'median': [1500000, 2500000],
            'top': [4000000, 5500000]
        })
        x_labels = ['0bps', '25bps']

        result = generate_price_elasticity_visualization_dollars(
            df_dist, df_ci, df_ci_columns, x_labels,
            sample_rate_context, '2023-12-15', sample_viz_config
        )

        assert result == mock_figure


# =============================================================================
# FILE SAVE TESTS
# =============================================================================


class TestSaveVisualizationFiles:
    """Tests for save_visualization_files function."""

    def test_returns_path_dict(self, tmp_path):
        """Should return dict with 'percentage' and 'dollar' paths."""
        from src.visualization.inference_plots import save_visualization_files

        mock_figure_pct = MagicMock()
        mock_figure_dollar = MagicMock()

        result = save_visualization_files(
            mock_figure_pct,
            mock_figure_dollar,
            tmp_path,
            '2023-12-15'
        )

        assert 'percentage' in result
        assert 'dollar' in result
        assert isinstance(result['percentage'], Path)
        assert isinstance(result['dollar'], Path)

    def test_creates_output_directory(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        from src.visualization.inference_plots import save_visualization_files

        mock_figure_pct = MagicMock()
        mock_figure_dollar = MagicMock()
        output_dir = tmp_path / 'new_subdir'

        save_visualization_files(
            mock_figure_pct,
            mock_figure_dollar,
            output_dir,
            '2023-12-15'
        )

        assert output_dir.exists()

    def test_calls_savefig_on_figures(self, tmp_path):
        """Should call savefig on both figures."""
        from src.visualization.inference_plots import save_visualization_files

        mock_figure_pct = MagicMock()
        mock_figure_dollar = MagicMock()

        save_visualization_files(
            mock_figure_pct,
            mock_figure_dollar,
            tmp_path,
            '2023-12-15'
        )

        mock_figure_pct.savefig.assert_called_once()
        mock_figure_dollar.savefig.assert_called_once()


# =============================================================================
# EXPORT TESTS
# =============================================================================


class TestPrepareBootstrapExports:
    """Tests for _prepare_bootstrap_exports helper function."""

    def test_adds_output_type_column(self, sample_bootstrap_data, sample_bootstrap_dollars):
        """Should add output_type column to both DataFrames."""
        from src.visualization.inference_plots import _prepare_bootstrap_exports

        df_pct_export, df_dollars_export = _prepare_bootstrap_exports(
            sample_bootstrap_data,
            sample_bootstrap_dollars,
            datetime(2023, 12, 15)
        )

        assert 'output_type' in df_pct_export.columns
        assert 'output_type' in df_dollars_export.columns
        assert (df_pct_export['output_type'] == 'pct_change').all()
        assert (df_dollars_export['output_type'] == 'dollars').all()

    def test_adds_simulation_run_column(self, sample_bootstrap_data, sample_bootstrap_dollars):
        """Should add simulation_run column."""
        from src.visualization.inference_plots import _prepare_bootstrap_exports

        df_pct_export, df_dollars_export = _prepare_bootstrap_exports(
            sample_bootstrap_data,
            sample_bootstrap_dollars,
            datetime(2023, 12, 15)
        )

        assert 'simulation_run' in df_pct_export.columns
        assert 'simulation_run' in df_dollars_export.columns


class TestMeltBootstrapForExport:
    """Tests for _melt_bootstrap_for_export helper function."""

    def test_adds_prediction_date(self):
        """Should add prediction_date column."""
        from src.visualization.inference_plots import _melt_bootstrap_for_export

        df = pd.DataFrame({
            'simulation_run': [0, 1],
            'output_type': ['pct_change', 'pct_change'],
            0.0: [1.0, 2.0],
            0.25: [1.5, 2.5]
        })

        result = _melt_bootstrap_for_export(df, '2023-12-15')

        assert 'prediction_date' in result.columns
        assert (result['prediction_date'] == '2023-12-15').all()

    def test_converts_rates_to_basis_points(self):
        """Should multiply rate values by 100."""
        from src.visualization.inference_plots import _melt_bootstrap_for_export

        df = pd.DataFrame({
            'simulation_run': [0],
            'output_type': ['pct_change'],
            0.01: [5.0],  # 1% rate
            0.02: [10.0]  # 2% rate
        })

        result = _melt_bootstrap_for_export(df, '2023-12-15')

        # Rates should be 1 and 2 (multiplied by 100)
        rates = result['rate_change_in_basis_points'].tolist()
        assert 1.0 in rates
        assert 2.0 in rates


class TestExportCsvFiles:
    """Tests for export_csv_files function."""

    def test_returns_dict_of_saved_files(self, tmp_path, sample_bootstrap_data,
                                         sample_bootstrap_dollars, sample_output_pct,
                                         sample_output_dollar):
        """Should return dict with all saved file paths."""
        from src.visualization.inference_plots import export_csv_files

        # Create mock melted BI data
        df_to_bi_melt = pd.DataFrame({
            'rate_change_in_basis_points': [0, 25],
            'range': ['bottom', 'bottom'],
            'dollar': [100, 200],
            'pct_change': [1.0, 2.0]
        })

        result = export_csv_files(
            sample_bootstrap_data,
            sample_bootstrap_dollars,
            sample_output_pct,
            sample_output_dollar,
            df_to_bi_melt,
            tmp_path,
            '2023-12-15',
            datetime(2023, 12, 15)
        )

        assert isinstance(result, dict)
        assert len(result) > 0
        # All values should be Path objects
        for key, path in result.items():
            assert isinstance(path, Path)

    def test_creates_output_directory(self, tmp_path, sample_bootstrap_data,
                                       sample_bootstrap_dollars, sample_output_pct,
                                       sample_output_dollar):
        """Should create output directory if needed."""
        from src.visualization.inference_plots import export_csv_files

        output_dir = tmp_path / 'csv_exports'
        df_to_bi_melt = pd.DataFrame({
            'rate_change_in_basis_points': [0],
            'range': ['bottom'],
            'dollar': [100],
            'pct_change': [1.0]
        })

        export_csv_files(
            sample_bootstrap_data,
            sample_bootstrap_dollars,
            sample_output_pct,
            sample_output_dollar,
            df_to_bi_melt,
            output_dir,
            '2023-12-15',
            datetime(2023, 12, 15)
        )

        assert output_dir.exists()
