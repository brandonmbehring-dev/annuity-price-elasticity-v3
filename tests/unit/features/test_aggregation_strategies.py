"""
Unit Tests for Aggregation Strategies Module
============================================

Tests for src/features/aggregation/ covering:
- AggregationStrategyBase (base class with validation)
- WeightedAggregation (market-share weighted, RILA default)
- TopNAggregation (top N competitors, FIA default)
- FirmLevelAggregation (firm-specific, MYGA default)
- MedianAggregation (simple median baseline)

Target: 85% coverage for both base.py and strategies.py

Test Pattern:
- Test basic functionality with simple data
- Test edge cases (missing columns, NaN, insufficient data)
- Test validation and error messages
- Test aggregation mathematical correctness
- Verify immutability (original DataFrames unchanged)

Author: Claude Code
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
import pytest

from src.features.aggregation.base import AggregationStrategyBase
from src.features.aggregation.strategies import (
    WeightedAggregation,
    TopNAggregation,
    FirmLevelAggregation,
    MedianAggregation,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_rates_data():
    """Simple DataFrame with company rate columns."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=3),
        'company_a': [4.5, 4.6, 4.7],
        'company_b': [4.2, 4.3, 4.4],
        'company_c': [4.0, 4.1, 4.2],
        'company_d': [3.8, 3.9, 4.0],
        'company_e': [3.5, 3.6, 3.7]
    })


@pytest.fixture
def weights_long_format():
    """Weights in long format (company | market_share)."""
    return pd.DataFrame({
        'company': ['company_a', 'company_b', 'company_c', 'company_d', 'company_e'],
        'market_share': [0.3, 0.25, 0.2, 0.15, 0.1]
    })


@pytest.fixture
def weights_wide_format():
    """Weights in wide format (columns are companies)."""
    return pd.DataFrame({
        'company_a': [0.3],
        'company_b': [0.25],
        'company_c': [0.2],
        'company_d': [0.15],
        'company_e': [0.1]
    })


@pytest.fixture
def rates_with_nan():
    """Rates DataFrame with some NaN values."""
    return pd.DataFrame({
        'company_a': [4.5, np.nan, 4.7],
        'company_b': [4.2, 4.3, np.nan],
        'company_c': [np.nan, 4.1, 4.2]
    })


@pytest.fixture
def company_columns():
    """Standard list of company columns."""
    return ['company_a', 'company_b', 'company_c', 'company_d', 'company_e']


# =============================================================================
# BASE CLASS TESTS
# =============================================================================


def test_base_class_min_companies_validation():
    """Test that base class validates min_companies parameter."""
    # Should reject zero or negative
    with pytest.raises(ValueError, match="min_companies must be positive"):
        class TestStrategy(AggregationStrategyBase):
            @property
            def requires_weights(self):
                return False
            @property
            def strategy_name(self):
                return "test"
            def aggregate(self, rates_df, company_columns, weights_df=None):
                return pd.Series()

        TestStrategy(min_companies=0)


def test_base_class_min_companies_property():
    """Test min_companies property is accessible."""
    strategy = WeightedAggregation(min_companies=5)
    assert strategy.min_companies == 5


def test_base_validate_inputs_empty_dataframe():
    """Test validation rejects empty DataFrame."""
    strategy = WeightedAggregation()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="rates_df cannot be empty"):
        strategy._validate_inputs(empty_df, ['company_a'])


def test_base_validate_inputs_empty_company_list():
    """Test validation rejects empty company_columns list."""
    strategy = WeightedAggregation()
    df = pd.DataFrame({'value': [1, 2, 3]})

    with pytest.raises(ValueError, match="company_columns cannot be empty"):
        strategy._validate_inputs(df, [])


def test_base_validate_inputs_insufficient_companies(simple_rates_data):
    """Test validation rejects insufficient company data."""
    strategy = WeightedAggregation(min_companies=3)

    # Only provide 2 companies when 3 required
    with pytest.raises(ValueError, match="Insufficient company data"):
        strategy._validate_inputs(simple_rates_data, ['company_a', 'company_b'])


def test_base_validate_inputs_returns_available_columns(simple_rates_data, company_columns):
    """Test validation returns list of available columns."""
    strategy = MedianAggregation(min_companies=3)
    weights = pd.DataFrame({'market_share': [0.5, 0.5]})

    # Request 5 columns, all exist
    available = strategy._validate_inputs(simple_rates_data, company_columns, weights)

    assert len(available) == 5
    assert set(available) == set(company_columns)


def test_base_validate_inputs_missing_columns_still_valid(simple_rates_data):
    """Test validation succeeds if enough columns exist (some missing OK)."""
    strategy = MedianAggregation(min_companies=3)

    # Request 6 columns, only 5 exist, but 5 >= 3 so should pass
    requested = ['company_a', 'company_b', 'company_c', 'company_x', 'company_y', 'company_z']
    available = strategy._validate_inputs(simple_rates_data, requested)

    assert len(available) == 3
    assert 'company_a' in available
    assert 'company_x' not in available


def test_base_validate_weights_required_but_missing(simple_rates_data, company_columns):
    """Test validation fails if weights required but not provided."""
    strategy = WeightedAggregation()  # requires_weights = True

    with pytest.raises(ValueError, match="requires weights_df"):
        strategy._validate_inputs(simple_rates_data, company_columns, weights_df=None)


def test_base_validate_weights_required_but_empty(simple_rates_data, company_columns):
    """Test validation fails if weights DataFrame is empty."""
    strategy = WeightedAggregation()
    empty_weights = pd.DataFrame()

    with pytest.raises(ValueError, match="weights_df cannot be empty"):
        strategy._validate_inputs(simple_rates_data, company_columns, weights_df=empty_weights)


def test_base_handle_missing_values(rates_with_nan):
    """Test _handle_missing_values fills NaN correctly."""
    strategy = MedianAggregation()
    columns = ['company_a', 'company_b', 'company_c']

    result = strategy._handle_missing_values(rates_with_nan, columns, fill_value=0.0)

    # NaN should be filled with 0.0
    assert result['company_a'].iloc[1] == 0.0  # Was NaN
    assert result['company_b'].iloc[2] == 0.0  # Was NaN
    assert result['company_c'].iloc[0] == 0.0  # Was NaN

    # Original values should be preserved
    assert result['company_a'].iloc[0] == 4.5


def test_base_handle_missing_values_custom_fill():
    """Test _handle_missing_values with custom fill value."""
    df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
    strategy = MedianAggregation()

    result = strategy._handle_missing_values(df, ['a'], fill_value=999.0)

    assert result['a'].iloc[1] == 999.0


def test_base_handle_missing_values_immutable(rates_with_nan):
    """Test that _handle_missing_values doesn't modify original."""
    strategy = MedianAggregation()
    original_nan_count = rates_with_nan.isna().sum().sum()

    _ = strategy._handle_missing_values(rates_with_nan, ['company_a'])

    # Original should still have NaN
    assert rates_with_nan.isna().sum().sum() == original_nan_count


# =============================================================================
# WEIGHTED AGGREGATION TESTS
# =============================================================================


def test_weighted_aggregation_basic(simple_rates_data, company_columns, weights_long_format):
    """Test basic weighted aggregation calculation."""
    strategy = WeightedAggregation()

    result = strategy.aggregate(simple_rates_data, company_columns, weights_long_format)

    # Verify it returns a Series
    assert isinstance(result, pd.Series)
    assert len(result) == len(simple_rates_data)

    # First row: [4.5, 4.2, 4.0, 3.8, 3.5] with weights [0.3, 0.25, 0.2, 0.15, 0.1]
    # Weighted mean = 4.5*0.3 + 4.2*0.25 + 4.0*0.2 + 3.8*0.15 + 3.5*0.1
    #               = 1.35 + 1.05 + 0.8 + 0.57 + 0.35 = 4.12
    expected = 4.12
    assert np.isclose(result.iloc[0], expected, atol=0.01)


def test_weighted_aggregation_wide_format_weights(simple_rates_data, company_columns, weights_wide_format):
    """Test weighted aggregation with wide format weights."""
    strategy = WeightedAggregation()

    result = strategy.aggregate(simple_rates_data, company_columns, weights_wide_format)

    # Should produce same result as long format
    expected = 4.12
    assert np.isclose(result.iloc[0], expected, atol=0.01)


def test_weighted_aggregation_properties():
    """Test WeightedAggregation properties."""
    strategy = WeightedAggregation()

    assert strategy.requires_weights == True
    assert strategy.strategy_name == "weighted"


def test_weighted_aggregation_custom_weight_column():
    """Test weighted aggregation with custom weight column name."""
    df = pd.DataFrame({'company_a': [5.0], 'company_b': [3.0]})
    weights = pd.DataFrame({
        'company': ['company_a', 'company_b'],
        'custom_weight': [0.7, 0.3]
    })

    strategy = WeightedAggregation(min_companies=2, weight_column='custom_weight')
    result = strategy.aggregate(df, ['company_a', 'company_b'], weights)

    # 5.0 * 0.7 + 3.0 * 0.3 = 3.5 + 0.9 = 4.4
    assert np.isclose(result.iloc[0], 4.4)


def test_weighted_aggregation_weight_normalization():
    """Test that weights are normalized to sum to 1."""
    df = pd.DataFrame({'company_a': [4.0], 'company_b': [2.0]})
    weights = pd.DataFrame({
        'company': ['company_a', 'company_b'],
        'market_share': [3.0, 1.0]  # Sum to 4, not 1
    })

    strategy = WeightedAggregation(min_companies=2)
    result = strategy.aggregate(df, ['company_a', 'company_b'], weights)

    # Should normalize: 3/(3+1)=0.75, 1/(3+1)=0.25
    # 4.0 * 0.75 + 2.0 * 0.25 = 3.0 + 0.5 = 3.5
    assert np.isclose(result.iloc[0], 3.5)


def test_weighted_aggregation_zero_weights_fallback():
    """Test fallback to equal weights when all weights are zero."""
    df = pd.DataFrame({'company_a': [4.0], 'company_b': [2.0]})
    weights = pd.DataFrame({
        'company': ['company_a', 'company_b'],
        'market_share': [0.0, 0.0]
    })

    strategy = WeightedAggregation(min_companies=2)
    result = strategy.aggregate(df, ['company_a', 'company_b'], weights)

    # Equal weights: (4.0 + 2.0) / 2 = 3.0
    assert np.isclose(result.iloc[0], 3.0)


def test_weighted_aggregation_handles_missing_weights():
    """Test handling of missing companies in weights."""
    df = pd.DataFrame({
        'company_a': [4.0],
        'company_b': [3.0],
        'company_c': [2.0]
    })
    weights = pd.DataFrame({
        'company': ['company_a', 'company_b'],  # Missing company_c
        'market_share': [0.6, 0.4]
    })

    strategy = WeightedAggregation()
    result = strategy.aggregate(df, ['company_a', 'company_b', 'company_c'], weights)

    # company_c gets weight 0, then normalized
    # After normalization: a=0.6, b=0.4, c=0
    # Result = 4.0*0.6 + 3.0*0.4 + 2.0*0 = 2.4 + 1.2 = 3.6
    assert np.isclose(result.iloc[0], 3.6)


# =============================================================================
# TOP-N AGGREGATION TESTS
# =============================================================================


def test_topn_aggregation_basic(simple_rates_data, company_columns):
    """Test basic top-N aggregation (mean of top N rates)."""
    strategy = TopNAggregation(n_competitors=3)

    result = strategy.aggregate(simple_rates_data, company_columns)

    # First row: [4.5, 4.2, 4.0, 3.8, 3.5]
    # Top 3: [4.5, 4.2, 4.0]
    # Mean = (4.5 + 4.2 + 4.0) / 3 = 12.7 / 3 = 4.233...
    expected = (4.5 + 4.2 + 4.0) / 3
    assert np.isclose(result.iloc[0], expected, atol=0.01)


def test_topn_aggregation_properties():
    """Test TopNAggregation properties."""
    strategy = TopNAggregation(n_competitors=5)

    assert strategy.requires_weights == False
    assert strategy.strategy_name == "top_n"
    assert strategy.n_competitors == 5


def test_topn_aggregation_n_larger_than_available():
    """Test top-N when N is larger than available companies."""
    df = pd.DataFrame({
        'company_a': [5.0],
        'company_b': [4.0],
        'company_c': [3.0]
    })

    # Request top 10, but only 3 available
    strategy = TopNAggregation(n_competitors=10)
    result = strategy.aggregate(df, ['company_a', 'company_b', 'company_c'])

    # Should use all 3: (5 + 4 + 3) / 3 = 4.0
    assert np.isclose(result.iloc[0], 4.0)


def test_topn_aggregation_with_nan():
    """Test top-N aggregation handles NaN correctly."""
    df = pd.DataFrame({
        'company_a': [5.0],
        'company_b': [np.nan],
        'company_c': [3.0],
        'company_d': [4.0]
    })

    strategy = TopNAggregation(n_competitors=2)
    result = strategy.aggregate(df, ['company_a', 'company_b', 'company_c', 'company_d'])

    # Available: [5.0, 3.0, 4.0] (NaN excluded)
    # Top 2: [5.0, 4.0]
    # Mean = (5 + 4) / 2 = 4.5
    assert np.isclose(result.iloc[0], 4.5)


def test_topn_aggregation_all_nan_row():
    """Test top-N with all NaN values (filled with 0 by _handle_missing_values)."""
    df = pd.DataFrame({
        'company_a': [np.nan],
        'company_b': [np.nan]
    })

    strategy = TopNAggregation(n_competitors=2, min_companies=2)
    result = strategy.aggregate(df, ['company_a', 'company_b'])

    # _handle_missing_values fills NaN with 0, so result is 0.0, not NaN
    assert result.iloc[0] == 0.0


def test_topn_aggregation_single_competitor():
    """Test top-N with n=1 selects highest rate."""
    df = pd.DataFrame({
        'company_a': [5.0, 3.0],
        'company_b': [4.0, 7.0],
        'company_c': [3.0, 2.0]
    })

    strategy = TopNAggregation(n_competitors=1)
    result = strategy.aggregate(df, ['company_a', 'company_b', 'company_c'])

    # First row: max([5.0, 4.0, 3.0]) = 5.0
    assert result.iloc[0] == 5.0
    # Second row: max([3.0, 7.0, 2.0]) = 7.0
    assert result.iloc[1] == 7.0


# =============================================================================
# FIRM-LEVEL AGGREGATION TESTS
# =============================================================================


def test_firmlevel_aggregation_mean(simple_rates_data, company_columns):
    """Test firm-level aggregation with mean method."""
    strategy = FirmLevelAggregation(aggregation_method='mean')

    result = strategy.aggregate(simple_rates_data, company_columns)

    # First row: mean([4.5, 4.2, 4.0, 3.8, 3.5]) = 20.0 / 5 = 4.0
    expected = (4.5 + 4.2 + 4.0 + 3.8 + 3.5) / 5
    assert np.isclose(result.iloc[0], expected)


def test_firmlevel_aggregation_median():
    """Test firm-level aggregation with median method."""
    df = pd.DataFrame({
        'company_a': [5.0],
        'company_b': [4.0],
        'company_c': [3.0],
        'company_d': [2.0],
        'company_e': [1.0]
    })

    strategy = FirmLevelAggregation(aggregation_method='median')
    result = strategy.aggregate(df, ['company_a', 'company_b', 'company_c', 'company_d', 'company_e'])

    # Median of [5, 4, 3, 2, 1] = 3
    assert result.iloc[0] == 3.0


def test_firmlevel_aggregation_max():
    """Test firm-level aggregation with max method."""
    df = pd.DataFrame({
        'company_a': [5.0, 2.0],
        'company_b': [4.0, 8.0],
        'company_c': [3.0, 1.0]
    })

    strategy = FirmLevelAggregation(aggregation_method='max')
    result = strategy.aggregate(df, ['company_a', 'company_b', 'company_c'])

    assert result.iloc[0] == 5.0  # Max of [5, 4, 3]
    assert result.iloc[1] == 8.0  # Max of [2, 8, 1]


def test_firmlevel_aggregation_min():
    """Test firm-level aggregation with min method."""
    df = pd.DataFrame({
        'company_a': [5.0, 2.0],
        'company_b': [4.0, 8.0],
        'company_c': [3.0, 1.0]
    })

    strategy = FirmLevelAggregation(aggregation_method='min')
    result = strategy.aggregate(df, ['company_a', 'company_b', 'company_c'])

    assert result.iloc[0] == 3.0  # Min of [5, 4, 3]
    assert result.iloc[1] == 1.0  # Min of [2, 8, 1]


def test_firmlevel_aggregation_properties():
    """Test FirmLevelAggregation properties."""
    strategy = FirmLevelAggregation(aggregation_method='median')

    assert strategy.requires_weights == False
    assert strategy.strategy_name == "firm_level"
    assert strategy.aggregation_method == "median"


def test_firmlevel_aggregation_invalid_method():
    """Test that invalid aggregation method raises error."""
    with pytest.raises(ValueError, match="aggregation_method must be one of"):
        FirmLevelAggregation(aggregation_method='invalid')


# =============================================================================
# MEDIAN AGGREGATION TESTS
# =============================================================================


def test_median_aggregation_basic(simple_rates_data, company_columns):
    """Test basic median aggregation."""
    strategy = MedianAggregation()

    result = strategy.aggregate(simple_rates_data, company_columns)

    # First row: median([4.5, 4.2, 4.0, 3.8, 3.5]) = 4.0
    assert result.iloc[0] == 4.0


def test_median_aggregation_properties():
    """Test MedianAggregation properties."""
    strategy = MedianAggregation()

    assert strategy.requires_weights == False
    assert strategy.strategy_name == "median"


def test_median_aggregation_even_count():
    """Test median with even number of companies."""
    df = pd.DataFrame({
        'company_a': [4.0],
        'company_b': [5.0],
        'company_c': [2.0],
        'company_d': [3.0]
    })

    strategy = MedianAggregation()
    result = strategy.aggregate(df, ['company_a', 'company_b', 'company_c', 'company_d'])

    # Median of [2, 3, 4, 5] = (3 + 4) / 2 = 3.5
    assert result.iloc[0] == 3.5


def test_median_aggregation_with_nan():
    """Test median aggregation handles NaN correctly."""
    df = pd.DataFrame({
        'company_a': [5.0],
        'company_b': [np.nan],
        'company_c': [3.0]
    })

    strategy = MedianAggregation()
    # NaN handling happens in base class _handle_missing_values (fills with 0)
    result = strategy.aggregate(df, ['company_a', 'company_b', 'company_c'])

    # After filling NaN with 0: [5.0, 0.0, 3.0]
    # Median = 3.0
    assert result.iloc[0] == 3.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_all_strategies_return_consistent_shape(simple_rates_data, company_columns, weights_long_format):
    """Test that all strategies return same-shaped output."""
    strategies = [
        WeightedAggregation(),
        TopNAggregation(n_competitors=3),
        FirmLevelAggregation(aggregation_method='mean'),
        MedianAggregation()
    ]

    results = []
    for strategy in strategies:
        if strategy.requires_weights:
            result = strategy.aggregate(simple_rates_data, company_columns, weights_long_format)
        else:
            result = strategy.aggregate(simple_rates_data, company_columns)
        results.append(result)

    # All should return same length
    for result in results:
        assert len(result) == len(simple_rates_data)
        assert isinstance(result, pd.Series)


def test_strategies_are_immutable(simple_rates_data, company_columns, weights_long_format):
    """Test that all strategies don't modify input DataFrames."""
    original_df = simple_rates_data.copy()
    original_weights = weights_long_format.copy()

    strategies = [
        WeightedAggregation(),
        TopNAggregation(),
        FirmLevelAggregation(),
        MedianAggregation()
    ]

    for strategy in strategies:
        if strategy.requires_weights:
            _ = strategy.aggregate(simple_rates_data, company_columns, weights_long_format)
        else:
            _ = strategy.aggregate(simple_rates_data, company_columns)

    # Verify no modifications
    pd.testing.assert_frame_equal(simple_rates_data, original_df)
    pd.testing.assert_frame_equal(weights_long_format, original_weights)


def test_strategy_comparison_different_results():
    """Test that different strategies produce different results."""
    df = pd.DataFrame({
        'company_a': [5.0],
        'company_b': [4.0],
        'company_c': [3.0],
        'company_d': [2.0]
    })
    weights = pd.DataFrame({
        'company': ['company_a', 'company_b', 'company_c', 'company_d'],
        'market_share': [0.4, 0.3, 0.2, 0.1]
    })
    companies = ['company_a', 'company_b', 'company_c', 'company_d']

    weighted = WeightedAggregation().aggregate(df, companies, weights).iloc[0]
    topn = TopNAggregation(n_competitors=2).aggregate(df, companies).iloc[0]
    median = MedianAggregation().aggregate(df, companies).iloc[0]
    firm_mean = FirmLevelAggregation(aggregation_method='mean').aggregate(df, companies).iloc[0]

    # Weighted: 5*0.4 + 4*0.3 + 3*0.2 + 2*0.1 = 2 + 1.2 + 0.6 + 0.2 = 4.0
    assert np.isclose(weighted, 4.0)

    # Top-2: (5 + 4) / 2 = 4.5
    assert np.isclose(topn, 4.5)

    # Median: median([5, 4, 3, 2]) = 3.5
    assert np.isclose(median, 3.5)

    # Mean: (5 + 4 + 3 + 2) / 4 = 3.5
    assert np.isclose(firm_mean, 3.5)

    # Verify they're different (weighted != top-n)
    assert not np.isclose(weighted, topn)


def test_min_companies_enforcement_across_strategies():
    """Test min_companies parameter works for all strategies."""
    df = pd.DataFrame({
        'company_a': [4.0],
        'company_b': [3.0]
    })
    weights = pd.DataFrame({
        'company': ['company_a', 'company_b'],
        'market_share': [0.6, 0.4]
    })

    strategies = [
        WeightedAggregation(min_companies=3),
        TopNAggregation(min_companies=3),
        FirmLevelAggregation(min_companies=3),
        MedianAggregation(min_companies=3)
    ]

    for strategy in strategies:
        with pytest.raises(ValueError, match="Insufficient company data"):
            if strategy.requires_weights:
                strategy.aggregate(df, ['company_a', 'company_b'], weights)
            else:
                strategy.aggregate(df, ['company_a', 'company_b'])
