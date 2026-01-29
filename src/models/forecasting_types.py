"""
Forecasting Results Type Definitions.

Provides dataclass containers for forecasting pipeline outputs.

Usage:
    from src.models.forecasting_types import ForecastingResults

    results = run_forecasting_pipeline(...)
    fr = ForecastingResults.from_pipeline_output(results)
    print(fr.mape_improvement)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd


@dataclass(frozen=True)
class ForecastMetrics:
    """Metrics from a single forecasting run.

    Attributes
    ----------
    mape : float
        Mean Absolute Percentage Error
    r2_score : float
        R-squared score
    coverage : Optional[float]
        Prediction interval coverage (if applicable)
    """
    mape: float
    r2_score: float
    coverage: Optional[float] = None


@dataclass
class ForecastingResults:
    """Container for forecasting pipeline outputs.

    Wraps the dictionary output of `run_forecasting_pipeline()` with
    structured attribute access.

    Attributes
    ----------
    benchmark_results : Dict[str, Any]
        Raw benchmark forecasting results
    model_results : Dict[str, Any]
        Raw model forecasting results
    comparison : Dict[str, Any]
        Comparison metrics between benchmark and model
    config_used : Dict[str, Any]
        Configuration parameters used for the run

    Properties
    ----------
    mape_improvement : float
        Percentage improvement in MAPE over benchmark
    r2_improvement : float
        Percentage improvement in R2 over benchmark
    model_outperforms : bool
        Whether the model beats the benchmark

    Examples
    --------
    >>> results = ForecastingResults.from_pipeline_output(pipeline_output)
    >>> results.mape_improvement
    15.3
    >>> results.model_outperforms
    True
    """
    benchmark_results: Dict[str, Any]
    model_results: Dict[str, Any]
    comparison: Dict[str, Any]
    config_used: Dict[str, Any]

    @classmethod
    def from_pipeline_output(cls, output: Dict[str, Any]) -> "ForecastingResults":
        """Create ForecastingResults from run_forecasting_pipeline() output.

        Parameters
        ----------
        output : Dict[str, Any]
            Raw output from `run_forecasting_pipeline()`

        Returns
        -------
        ForecastingResults
            Structured results container
        """
        return cls(
            benchmark_results=output["benchmark_results"],
            model_results=output["model_results"],
            comparison=output["comparison"],
            config_used=output["config_used"],
        )

    @property
    def mape_improvement(self) -> float:
        """Percentage improvement in MAPE over benchmark."""
        return self.comparison.get("mape_improvement_pct", 0.0)

    @property
    def r2_improvement(self) -> float:
        """Percentage improvement in R2 score over benchmark."""
        return self.comparison.get("r2_improvement_pct", 0.0)

    @property
    def model_outperforms(self) -> bool:
        """Whether the model outperforms the benchmark."""
        return self.comparison.get("model_outperforms", False)

    @property
    def model_mape(self) -> float:
        """Model MAPE score."""
        return self.model_results.get("metrics", {}).get("mape", float("nan"))

    @property
    def benchmark_mape(self) -> float:
        """Benchmark MAPE score."""
        return self.benchmark_results.get("metrics", {}).get("mape", float("nan"))

    @property
    def model_r2(self) -> float:
        """Model R2 score."""
        return self.model_results.get("metrics", {}).get("r2_score", float("nan"))

    @property
    def benchmark_r2(self) -> float:
        """Benchmark R2 score."""
        return self.benchmark_results.get("metrics", {}).get("r2_score", float("nan"))

    @property
    def model_features(self) -> List[str]:
        """Features used in the model."""
        return self.config_used.get("model_features", [])

    @property
    def n_forecasts(self) -> int:
        """Number of forecasting periods."""
        start = self.config_used.get("start_cutoff", 0)
        end = self.config_used.get("end_cutoff", 0)
        return max(0, end - start)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation with all fields
        """
        return {
            "benchmark_results": self.benchmark_results,
            "model_results": self.model_results,
            "comparison": self.comparison,
            "config_used": self.config_used,
            # Convenience fields
            "mape_improvement": self.mape_improvement,
            "r2_improvement": self.r2_improvement,
            "model_outperforms": self.model_outperforms,
        }

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns
        -------
        str
            Formatted summary of results
        """
        return (
            f"ForecastingResults Summary:\n"
            f"  Forecasts: {self.n_forecasts}\n"
            f"  Model MAPE: {self.model_mape:.4f}\n"
            f"  Benchmark MAPE: {self.benchmark_mape:.4f}\n"
            f"  MAPE Improvement: {self.mape_improvement:.1f}%\n"
            f"  Model R2: {self.model_r2:.4f}\n"
            f"  Benchmark R2: {self.benchmark_r2:.4f}\n"
            f"  R2 Improvement: {self.r2_improvement:.1f}%\n"
            f"  Model Outperforms: {self.model_outperforms}\n"
            f"  Features: {', '.join(self.model_features)}"
        )


__all__ = ["ForecastingResults", "ForecastMetrics"]
