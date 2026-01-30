"""
Test Fixture Data Source Adapter.

Adapter for loading pre-captured test fixtures. Enables offline testing
and validation against known baselines.

Usage:
    from src.data.adapters.fixture_adapter import FixtureAdapter

    adapter = FixtureAdapter(Path("tests/fixtures/rila"))
    df = adapter.load_sales_data()
"""

from typing import Optional
from pathlib import Path
import pandas as pd

from src.data.adapters.base import DataAdapterBase
from src.core.product_registry import (
    get_fixture_filter_name,
    is_product_code,
)


class FixtureAdapter(DataAdapterBase):
    """Test fixture data source adapter.

    Loads pre-captured data fixtures for testing and validation.
    Supports both single-file and multi-file fixture structures.

    Parameters
    ----------
    fixtures_dir : Path
        Directory containing fixture files
    output_dir : Optional[Path]
        Directory for test outputs (defaults to fixtures_dir/outputs)

    Expected Structure:
        fixtures_dir/
        ├── sales_fixture.parquet        # Single file
        ├── rates_fixture.parquet
        ├── weights_fixture.parquet
        ├── macro_fixture.parquet
        └── outputs/                     # Test outputs

    Or multi-file:
        fixtures_dir/
        ├── sales/
        │   ├── part1.parquet
        │   └── part2.parquet
        └── ...

    Examples
    --------
    >>> adapter = FixtureAdapter(Path("tests/fixtures/rila"))
    >>> df = adapter.load_sales_data()
    """

    # Fixture file naming conventions
    FIXTURE_NAMES = {
        "sales": [
            "sales_fixture.parquet",
            "raw_sales_data.parquet",
            "filtered_flexguard_6y20.parquet",
            "sales.parquet"
        ],
        "rates": [
            "rates_fixture.parquet",
            "raw_wink_data.parquet",
            "wink_competitive_rates_pivoted.parquet",
            "wink_rates.parquet",
            "rates.parquet"
        ],
        "weights": [
            "weights_fixture.parquet",
            "market_share_weights.parquet",
            "market_weights.parquet",
            "weights.parquet"
        ],
        "macro": [
            "macro_fixture.parquet",
            "macro_data.parquet",
            "macro.parquet"
        ],
    }

    def __init__(
        self,
        fixtures_dir: Path,
        output_dir: Optional[Path] = None,
    ):
        self._fixtures_dir = Path(fixtures_dir)
        self._output_dir = output_dir or (self._fixtures_dir / "outputs")

        if not self._fixtures_dir.exists():
            raise ValueError(
                f"CRITICAL: Fixtures directory not found: {self._fixtures_dir}. "
                f"Business impact: Cannot run tests without fixtures. "
                f"Required action: Create fixtures or specify correct path."
            )

    @property
    def source_type(self) -> str:
        return "fixture"

    @property
    def fixtures_dir(self) -> Path:
        """Fixtures directory path."""
        return self._fixtures_dir

    def _find_fixture_file(self, data_type: str) -> Path:
        """Find the fixture file for a data type.

        Tries multiple naming conventions and directory structures.

        Parameters
        ----------
        data_type : str
            Data type: "sales", "rates", "weights", "macro"

        Returns
        -------
        Path
            Path to fixture file

        Raises
        ------
        FileNotFoundError
            If no fixture file found
        """
        # Try single-file naming conventions
        for name in self.FIXTURE_NAMES.get(data_type, []):
            path = self._fixtures_dir / name
            if path.exists():
                return path

        # Try subdirectory structure
        subdir = self._fixtures_dir / data_type
        if subdir.exists():
            parquet_files = list(subdir.glob("*.parquet"))
            if parquet_files:
                return subdir  # Return directory for multi-file loading

        # Special case: Economic indicators in subdirectory
        if data_type == "macro":
            econ_dir = self._fixtures_dir / "economic_indicators"
            if econ_dir.exists() and econ_dir.is_dir():
                parquet_files = list(econ_dir.glob("*.parquet"))
                if parquet_files:
                    return econ_dir

        raise FileNotFoundError(
            f"CRITICAL: No fixture found for '{data_type}'. "
            f"Searched in: {self._fixtures_dir}. "
            f"Expected names: {self.FIXTURE_NAMES.get(data_type, [])}. "
            f"Required action: Create fixture file or verify path."
        )

    def _load_fixture(
        self,
        data_type: str,
        product_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load fixture data.

        Handles both single-file and multi-file fixtures.

        Parameters
        ----------
        data_type : str
            Data type to load
        product_filter : Optional[str]
            Filter by product_name

        Returns
        -------
        pd.DataFrame
            Fixture data
        """
        path = self._find_fixture_file(data_type)

        if path.is_file():
            # Single file
            df = pd.read_parquet(path, engine="pyarrow")
        else:
            # Directory with multiple files
            parquet_files = list(path.glob("*.parquet"))
            dfs = [pd.read_parquet(f, engine="pyarrow") for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)

        # Apply product filter if specified
        if product_filter and "product_name" in df.columns:
            # Use Product Registry to map product code to fixture name
            if is_product_code(product_filter):
                actual_filter = get_fixture_filter_name(product_filter)
            else:
                actual_filter = product_filter
            df = df[df["product_name"] == actual_filter].reset_index(drop=True)

        return df

    def load_sales_data(
        self, product_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """Load sales fixture.

        Parameters
        ----------
        product_filter : Optional[str]
            Filter by product name

        Returns
        -------
        pd.DataFrame
            Sales fixture data
        """
        df = self._load_fixture("sales", product_filter)

        # Relaxed validation for fixtures (may have subset of columns)
        if df.empty:
            raise ValueError(
                "CRITICAL: Sales fixture is empty. "
                "Business impact: Cannot run tests. "
                "Required action: Verify fixture data capture."
            )

        return df

    def load_competitive_rates(self, start_date: str) -> pd.DataFrame:
        """Load rates fixture.

        Parameters
        ----------
        start_date : str
            Start date filter (may be ignored in fixtures)

        Returns
        -------
        pd.DataFrame
            Rates fixture data
        """
        df = self._load_fixture("rates")

        # Apply date filter if column exists
        if "effective_date" in df.columns:
            df = df[df["effective_date"] >= start_date].reset_index(drop=True)

        return df

    def load_market_weights(self) -> pd.DataFrame:
        """Load weights fixture.

        Returns
        -------
        pd.DataFrame
            Weights fixture data
        """
        return self._load_fixture("weights")

    def load_macro_data(self) -> pd.DataFrame:
        """Load macro fixture.

        Returns
        -------
        pd.DataFrame
            Macro fixture data
        """
        return self._load_fixture("macro")

    def save_output(
        self, df: pd.DataFrame, name: str, format: str = "parquet"
    ) -> str:
        """Save test output.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        name : str
            Output name
        format : str
            Output format

        Returns
        -------
        str
            Path where saved
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


__all__ = ["FixtureAdapter"]
