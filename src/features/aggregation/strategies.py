"""
Concrete Aggregation Strategy Implementations.

Provides product-specific aggregation methods for competitor rates:
- WeightedAggregation: Market-share weighted (RILA default)
- TopNAggregation: Top N competitors by rate (FIA default)
- FirmLevelAggregation: Firm-specific calculations (MYGA)

Usage:
    from src.features.aggregation.strategies import (
        WeightedAggregation,
        TopNAggregation,
        FirmLevelAggregation,
    )

    strategy = WeightedAggregation()
    competitor_rate = strategy.aggregate(rates_df, company_columns, weights_df)
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from src.features.aggregation.base import AggregationStrategyBase


class WeightedAggregation(AggregationStrategyBase):
    """Market-share weighted aggregation strategy.

    Default for RILA products. Uses market share weights to compute
    weighted mean of competitor rates.

    Parameters
    ----------
    min_companies : int
        Minimum companies required (default: 3)
    weight_column : str
        Column name in weights_df containing weights
    """

    def __init__(
        self,
        min_companies: int = 3,
        weight_column: str = "market_share",
    ):
        super().__init__(min_companies=min_companies)
        self._weight_column = weight_column

    @property
    def requires_weights(self) -> bool:
        return True

    @property
    def strategy_name(self) -> str:
        return "weighted"

    def aggregate(
        self,
        rates_df: pd.DataFrame,
        company_columns: List[str],
        weights_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute market-share weighted competitor rate.

        Parameters
        ----------
        rates_df : pd.DataFrame
            DataFrame with company rate columns
        company_columns : List[str]
            Company columns to aggregate
        weights_df : pd.DataFrame
            Market weights by company

        Returns
        -------
        pd.Series
            Weighted mean competitor rate series
        """
        available = self._validate_inputs(rates_df, company_columns, weights_df)

        # Get weights for available companies
        weights = self._get_normalized_weights(weights_df, available)

        # Compute weighted mean
        result = rates_df.copy()
        company_data = self._handle_missing_values(result, available)[available]

        weighted_sum = (company_data * weights).sum(axis=1)

        return weighted_sum

    def _get_normalized_weights(
        self, weights_df: pd.DataFrame, companies: List[str]
    ) -> pd.Series:
        """Extract and normalize weights for given companies.

        Parameters
        ----------
        weights_df : pd.DataFrame
            Full weights DataFrame
        companies : List[str]
            Companies to get weights for

        Returns
        -------
        pd.Series
            Normalized weights summing to 1
        """
        # Handle different weight DataFrame formats
        if "company" in weights_df.columns:
            # Long format: company | weight
            weights = weights_df.set_index("company")[self._weight_column]
            weights = weights.reindex(companies, fill_value=0)
        else:
            # Wide format: columns are companies
            weights = weights_df[companies].iloc[0]

        # Normalize to sum to 1
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            # Equal weights fallback
            weights = pd.Series(1.0 / len(companies), index=companies)

        return weights


class TopNAggregation(AggregationStrategyBase):
    """Top-N competitors aggregation strategy.

    Default for FIA products. Computes mean of top N competitor rates.

    Parameters
    ----------
    n_competitors : int
        Number of top competitors to include (default: 5)
    min_companies : int
        Minimum companies required (default: 3)
    """

    def __init__(
        self,
        n_competitors: int = 5,
        min_companies: int = 3,
    ):
        super().__init__(min_companies=min_companies)
        self._n_competitors = n_competitors

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def strategy_name(self) -> str:
        return "top_n"

    @property
    def n_competitors(self) -> int:
        """Number of competitors to include."""
        return self._n_competitors

    def aggregate(
        self,
        rates_df: pd.DataFrame,
        company_columns: List[str],
        weights_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute mean of top N competitor rates.

        Parameters
        ----------
        rates_df : pd.DataFrame
            DataFrame with company rate columns
        company_columns : List[str]
            Company columns to consider
        weights_df : Optional[pd.DataFrame]
            Not used (ignored)

        Returns
        -------
        pd.Series
            Mean of top N competitor rates
        """
        available = self._validate_inputs(rates_df, company_columns)

        result = rates_df.copy()
        company_data = self._handle_missing_values(result, available)[available]

        # For each row, get top N values and compute mean
        def top_n_mean(row):
            valid_values = row.dropna()
            if len(valid_values) == 0:
                return np.nan
            n = min(self._n_competitors, len(valid_values))
            return valid_values.nlargest(n).mean()

        return company_data.apply(top_n_mean, axis=1)


class FirmLevelAggregation(AggregationStrategyBase):
    """Firm-level aggregation strategy.

    Default for MYGA products. Computes aggregates at individual firm level,
    enabling firm-specific analysis.

    Parameters
    ----------
    min_companies : int
        Minimum companies required (default: 3)
    aggregation_method : str
        How to combine firms: "mean", "median", "max", "min"
    """

    def __init__(
        self,
        min_companies: int = 3,
        aggregation_method: str = "mean",
    ):
        super().__init__(min_companies=min_companies)
        valid_methods = {"mean", "median", "max", "min"}
        if aggregation_method not in valid_methods:
            raise ValueError(
                f"aggregation_method must be one of {valid_methods}: "
                f"{aggregation_method}"
            )
        self._aggregation_method = aggregation_method

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def strategy_name(self) -> str:
        return "firm_level"

    @property
    def aggregation_method(self) -> str:
        """Method used to combine firms."""
        return self._aggregation_method

    def aggregate(
        self,
        rates_df: pd.DataFrame,
        company_columns: List[str],
        weights_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute firm-level aggregate of competitor rates.

        Parameters
        ----------
        rates_df : pd.DataFrame
            DataFrame with company rate columns
        company_columns : List[str]
            Company columns to aggregate
        weights_df : Optional[pd.DataFrame]
            Not used (ignored)

        Returns
        -------
        pd.Series
            Firm-level aggregated competitor rates
        """
        available = self._validate_inputs(rates_df, company_columns)

        result = rates_df.copy()
        company_data = self._handle_missing_values(result, available)[available]

        # Apply aggregation method
        if self._aggregation_method == "mean":
            return company_data.mean(axis=1)
        elif self._aggregation_method == "median":
            return company_data.median(axis=1)
        elif self._aggregation_method == "max":
            return company_data.max(axis=1)
        elif self._aggregation_method == "min":
            return company_data.min(axis=1)
        else:
            # Should not reach here due to validation
            raise ValueError(f"Unknown method: {self._aggregation_method}")


class MedianAggregation(AggregationStrategyBase):
    """Median aggregation strategy.

    Simple median of all competitor rates. Useful as baseline.

    Parameters
    ----------
    min_companies : int
        Minimum companies required (default: 3)
    """

    def __init__(self, min_companies: int = 3):
        super().__init__(min_companies=min_companies)

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def strategy_name(self) -> str:
        return "median"

    def aggregate(
        self,
        rates_df: pd.DataFrame,
        company_columns: List[str],
        weights_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute median of competitor rates.

        Parameters
        ----------
        rates_df : pd.DataFrame
            DataFrame with company rate columns
        company_columns : List[str]
            Company columns to aggregate
        weights_df : Optional[pd.DataFrame]
            Not used (ignored)

        Returns
        -------
        pd.Series
            Median competitor rate series
        """
        available = self._validate_inputs(rates_df, company_columns)

        result = rates_df.copy()
        company_data = self._handle_missing_values(result, available)[available]

        return company_data.median(axis=1)


__all__ = [
    "WeightedAggregation",
    "TopNAggregation",
    "FirmLevelAggregation",
    "MedianAggregation",
]
