"""
Local Filesystem Data Source Adapter.

Development adapter for loading data from local filesystem.

Usage:
    from src.data.adapters.local_adapter import LocalAdapter

    adapter = LocalAdapter(Path("./data"))
    df = adapter.load_sales_data()
"""

from typing import Optional
from pathlib import Path
import pandas as pd

from src.data.adapters.base import DataAdapterBase


class LocalAdapter(DataAdapterBase):
    """Local filesystem data source adapter.

    Loads data from local directory structure. Useful for development
    and offline testing with local data copies.

    Parameters
    ----------
    data_dir : Path
        Base directory containing data subdirectories
    output_dir : Optional[Path]
        Directory for outputs (defaults to data_dir/outputs)

    Expected Directory Structure:
        data_dir/
        ├── sales/          # TDE sales parquet files
        ├── rates/          # WINK rates parquet files
        ├── weights/        # Market weights
        └── macro/          # Macro indicators

    Examples
    --------
    >>> adapter = LocalAdapter(Path("./data"))
    >>> sales_df = adapter.load_sales_data()
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Optional[Path] = None,
    ):
        self._data_dir = Path(data_dir)
        self._output_dir = output_dir or (self._data_dir / "outputs")

        if not self._data_dir.exists():
            raise ValueError(
                f"CRITICAL: Data directory does not exist: {self._data_dir}. "
                f"Business impact: Cannot load data for analysis. "
                f"Required action: Create directory or specify correct path."
            )

    @property
    def source_type(self) -> str:
        """Return 'local' as the data source identifier."""
        return "local"

    @property
    def data_dir(self) -> Path:
        """Base data directory."""
        return self._data_dir

    def _load_parquet_from_dir(
        self,
        subdir: str,
        product_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and concatenate parquet files from subdirectory.

        Parameters
        ----------
        subdir : str
            Subdirectory name under data_dir
        product_filter : Optional[str]
            Filter by product_name if present

        Returns
        -------
        pd.DataFrame
            Concatenated data
        """
        dir_path = self._data_dir / subdir

        if not dir_path.exists():
            raise ValueError(
                f"CRITICAL: Subdirectory not found: {dir_path}. "
                f"Business impact: Cannot load {subdir} data. "
                f"Required action: Create directory with parquet files."
            )

        parquet_files = list(dir_path.glob("*.parquet"))

        if not parquet_files:
            raise ValueError(
                f"CRITICAL: No parquet files in {dir_path}. "
                f"Business impact: No {subdir} data available. "
                f"Required action: Add parquet files to directory."
            )

        dfs = []
        for f in parquet_files:
            df_temp = pd.read_parquet(f, engine="pyarrow")

            if product_filter and "product_name" in df_temp.columns:
                df_temp = df_temp[
                    df_temp["product_name"] == product_filter
                ].reset_index(drop=True)

            dfs.append(df_temp)

        return pd.concat(dfs, ignore_index=True)

    def load_sales_data(
        self, product_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """Load TDE sales data from local filesystem.

        Parameters
        ----------
        product_filter : Optional[str]
            Filter by product name

        Returns
        -------
        pd.DataFrame
            Sales data
        """
        df = self._load_parquet_from_dir("sales", product_filter)

        required = ["application_signed_date", "premium_amount"]
        self._validate_dataframe(df, required, "sales_data")

        return df

    def load_competitive_rates(self, start_date: str) -> pd.DataFrame:
        """Load WINK competitive rates from local filesystem.

        Parameters
        ----------
        start_date : str
            Start date filter (YYYY-MM-DD)

        Returns
        -------
        pd.DataFrame
            Competitive rates
        """
        df = self._load_parquet_from_dir("rates")

        if "effective_date" in df.columns:
            df = df[df["effective_date"] >= start_date].reset_index(drop=True)

        return df

    def load_market_weights(self) -> pd.DataFrame:
        """Load market share weights from local filesystem.

        Returns
        -------
        pd.DataFrame
            Market weights
        """
        return self._load_parquet_from_dir("weights")

    def load_macro_data(self) -> pd.DataFrame:
        """Load macro indicators from local filesystem.

        Returns
        -------
        pd.DataFrame
            Macro data
        """
        return self._load_parquet_from_dir("macro")

    def save_output(
        self, df: pd.DataFrame, name: str, format: str = "parquet"
    ) -> str:
        """Save output to local filesystem.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        name : str
            Output filename (without extension)
        format : str
            Output format: "parquet", "csv", "excel"

        Returns
        -------
        str
            Absolute path where saved
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            path = self._output_dir / f"{name}.parquet"
            return self._save_parquet(df, path)
        elif format == "csv":
            path = self._output_dir / f"{name}.csv"
            return self._save_csv(df, path)
        elif format == "excel":
            path = self._output_dir / f"{name}.xlsx"
            return self._save_excel(df, path)
        else:
            raise ValueError(f"Unsupported format: {format}")


__all__ = ["LocalAdapter"]
