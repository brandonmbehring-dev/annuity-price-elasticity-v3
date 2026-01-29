"""
Base Class for Data Source Adapters.

Provides common functionality for all adapter implementations.

Usage:
    from src.data.adapters.base import DataAdapterBase
"""

from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
import pandas as pd


class DataAdapterBase(ABC):
    """Abstract base class for data source adapters.

    Provides common utilities and interface definition for all adapters.

    Subclasses must implement:
        - load_sales_data()
        - load_competitive_rates()
        - load_market_weights()
        - load_macro_data()
        - save_output()
        - source_type property
    """

    @abstractmethod
    def load_sales_data(
        self, product_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """Load sales data from the data source."""
        pass

    @abstractmethod
    def load_competitive_rates(self, start_date: str) -> pd.DataFrame:
        """Load competitive rate data."""
        pass

    @abstractmethod
    def load_market_weights(self) -> pd.DataFrame:
        """Load market share weights."""
        pass

    @abstractmethod
    def load_macro_data(self) -> pd.DataFrame:
        """Load macroeconomic indicator data."""
        pass

    @abstractmethod
    def save_output(
        self, df: pd.DataFrame, name: str, format: str = "parquet"
    ) -> str:
        """Save output to the data destination."""
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the adapter type identifier."""
        pass

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: list,
        data_name: str,
    ) -> None:
        """Validate DataFrame has required columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        required_columns : list
            Columns that must exist
        data_name : str
            Name for error messages

        Raises
        ------
        ValueError
            If required columns are missing
        """
        if df.empty:
            raise ValueError(
                f"CRITICAL: {data_name} is empty. "
                f"Business impact: No data available for analysis. "
                f"Required action: Verify data source and extraction."
            )

        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"CRITICAL: {data_name} missing required columns: {missing}. "
                f"Available: {list(df.columns)[:10]}... "
                f"Required action: Verify data schema matches expectations."
            )

    def _save_parquet(self, df: pd.DataFrame, path: Path) -> str:
        """Save DataFrame as parquet.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        path : Path
            Destination path

        Returns
        -------
        str
            Absolute path where saved
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False)
        return str(path.absolute())

    def _save_csv(self, df: pd.DataFrame, path: Path) -> str:
        """Save DataFrame as CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        path : Path
            Destination path

        Returns
        -------
        str
            Absolute path where saved
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return str(path.absolute())

    def _save_excel(self, df: pd.DataFrame, path: Path) -> str:
        """Save DataFrame as Excel.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        path : Path
            Destination path

        Returns
        -------
        str
            Absolute path where saved
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, index=False, engine="openpyxl")
        return str(path.absolute())


__all__ = ["DataAdapterBase"]
