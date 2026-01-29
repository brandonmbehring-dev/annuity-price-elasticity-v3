"""
Base Classes for Competitor Aggregation Strategies.

Defines abstract base class and common utilities for aggregation strategies.

Usage:
    from src.features.aggregation.base import AggregationStrategyBase
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import numpy as np


class AggregationStrategyBase(ABC):
    """Abstract base class for competitor rate aggregation.

    Provides common validation and utility methods used by all strategies.

    Subclasses must implement:
        - aggregate(): Core aggregation logic
        - requires_weights: Property indicating if weights are needed
        - strategy_name: Property returning strategy identifier
    """

    def __init__(self, min_companies: int = 3):
        """Initialize strategy with configuration.

        Parameters
        ----------
        min_companies : int
            Minimum number of companies required for valid calculation
        """
        if min_companies <= 0:
            raise ValueError(
                f"min_companies must be positive, got {min_companies}"
            )
        self._min_companies = min_companies

    @property
    def min_companies(self) -> int:
        """Minimum companies required for valid calculation."""
        return self._min_companies

    @abstractmethod
    def aggregate(
        self,
        rates_df: pd.DataFrame,
        company_columns: List[str],
        weights_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Aggregate competitor rates using the strategy.

        Parameters
        ----------
        rates_df : pd.DataFrame
            DataFrame containing company rate columns
        company_columns : List[str]
            List of company column names to aggregate
        weights_df : Optional[pd.DataFrame]
            Market weights (if required by strategy)

        Returns
        -------
        pd.Series
            Aggregated competitor rate series
        """
        pass

    @property
    @abstractmethod
    def requires_weights(self) -> bool:
        """Whether this strategy requires market weights."""
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the strategy identifier."""
        pass

    def _validate_inputs(
        self,
        rates_df: pd.DataFrame,
        company_columns: List[str],
        weights_df: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """Validate inputs and return available company columns.

        Parameters
        ----------
        rates_df : pd.DataFrame
            DataFrame to validate
        company_columns : List[str]
            Expected company columns
        weights_df : Optional[pd.DataFrame]
            Weights (validated if requires_weights)

        Returns
        -------
        List[str]
            Available company columns that exist in DataFrame

        Raises
        ------
        ValueError
            If validation fails
        """
        if rates_df.empty:
            raise ValueError(
                "CRITICAL: rates_df cannot be empty. "
                "Business impact: No data available for competitor analysis. "
                "Required action: Verify data loading completed successfully."
            )

        if not company_columns:
            raise ValueError(
                "CRITICAL: company_columns cannot be empty. "
                "Business impact: Cannot calculate competitor rates. "
                "Required action: Verify company_columns configuration."
            )

        # Find available columns
        available = [c for c in company_columns if c in rates_df.columns]

        if len(available) < self._min_companies:
            missing = set(company_columns) - set(rates_df.columns)
            raise ValueError(
                f"Insufficient company data for {self.strategy_name}. "
                f"Found {len(available)}, need minimum {self._min_companies}. "
                f"Available: {available}. "
                f"Missing: {missing}. "
                f"Check data loading and filtering."
            )

        # Validate weights if required
        if self.requires_weights:
            if weights_df is None:
                raise ValueError(
                    f"Strategy '{self.strategy_name}' requires weights_df. "
                    "Business impact: Cannot compute weighted aggregation. "
                    "Required action: Provide market weights DataFrame."
                )
            if weights_df.empty:
                raise ValueError(
                    "weights_df cannot be empty for weighted aggregation."
                )

        return available

    def _handle_missing_values(
        self, df: pd.DataFrame, columns: List[str], fill_value: float = 0.0
    ) -> pd.DataFrame:
        """Handle missing values in company columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with company columns
        columns : List[str]
            Columns to process
        fill_value : float
            Value to fill NaN (default: 0.0)

        Returns
        -------
        pd.DataFrame
            DataFrame with filled values
        """
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[col] = result[col].fillna(fill_value)
        return result


__all__ = ["AggregationStrategyBase"]
