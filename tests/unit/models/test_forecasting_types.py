"""
Unit tests for src/models/forecasting_types.py.

Tests ForecastMetrics and ForecastingResults dataclasses.
"""

import math
import pytest

from src.models.forecasting_types import ForecastMetrics, ForecastingResults


class TestForecastMetrics:
    """Tests for ForecastMetrics dataclass."""

    def test_creates_with_all_fields(self):
        """Creates ForecastMetrics with all fields including coverage."""
        metrics = ForecastMetrics(mape=0.15, r2_score=0.85, coverage=0.95)

        assert metrics.mape == 0.15
        assert metrics.r2_score == 0.85
        assert metrics.coverage == 0.95

    def test_creates_without_coverage(self):
        """Creates ForecastMetrics without optional coverage."""
        metrics = ForecastMetrics(mape=0.10, r2_score=0.90)

        assert metrics.mape == 0.10
        assert metrics.r2_score == 0.90
        assert metrics.coverage is None

    def test_frozen_dataclass_is_immutable(self):
        """ForecastMetrics is frozen/immutable."""
        metrics = ForecastMetrics(mape=0.15, r2_score=0.85)

        with pytest.raises(AttributeError):
            metrics.mape = 0.20

    def test_equality_comparison(self):
        """ForecastMetrics supports equality comparison."""
        metrics1 = ForecastMetrics(mape=0.15, r2_score=0.85, coverage=0.95)
        metrics2 = ForecastMetrics(mape=0.15, r2_score=0.85, coverage=0.95)
        metrics3 = ForecastMetrics(mape=0.20, r2_score=0.85, coverage=0.95)

        assert metrics1 == metrics2
        assert metrics1 != metrics3


class TestForecastingResultsCreation:
    """Tests for ForecastingResults creation and basic access."""

    @pytest.fixture
    def sample_pipeline_output(self):
        """Sample pipeline output for testing."""
        return {
            "benchmark_results": {
                "metrics": {"mape": 0.20, "r2_score": 0.75},
                "forecasts": [1.0, 2.0, 3.0],
            },
            "model_results": {
                "metrics": {"mape": 0.15, "r2_score": 0.85},
                "forecasts": [1.1, 2.1, 3.1],
            },
            "comparison": {
                "mape_improvement_pct": 25.0,
                "r2_improvement_pct": 13.3,
                "model_outperforms": True,
            },
            "config_used": {
                "model_features": ["feature1", "feature2", "feature3"],
                "start_cutoff": 10,
                "end_cutoff": 20,
            },
        }

    def test_creates_directly(self, sample_pipeline_output):
        """Creates ForecastingResults with direct constructor."""
        results = ForecastingResults(
            benchmark_results=sample_pipeline_output["benchmark_results"],
            model_results=sample_pipeline_output["model_results"],
            comparison=sample_pipeline_output["comparison"],
            config_used=sample_pipeline_output["config_used"],
        )

        assert results.benchmark_results == sample_pipeline_output["benchmark_results"]
        assert results.model_results == sample_pipeline_output["model_results"]

    def test_from_pipeline_output(self, sample_pipeline_output):
        """Creates ForecastingResults from pipeline output dict."""
        results = ForecastingResults.from_pipeline_output(sample_pipeline_output)

        assert results.benchmark_results == sample_pipeline_output["benchmark_results"]
        assert results.model_results == sample_pipeline_output["model_results"]
        assert results.comparison == sample_pipeline_output["comparison"]
        assert results.config_used == sample_pipeline_output["config_used"]

    def test_from_pipeline_output_missing_key_raises(self):
        """Raises KeyError when required key missing from pipeline output."""
        incomplete_output = {
            "benchmark_results": {},
            "model_results": {},
            # Missing: comparison, config_used
        }

        with pytest.raises(KeyError):
            ForecastingResults.from_pipeline_output(incomplete_output)


class TestForecastingResultsProperties:
    """Tests for ForecastingResults computed properties."""

    @pytest.fixture
    def results(self):
        """ForecastingResults with complete data."""
        return ForecastingResults(
            benchmark_results={
                "metrics": {"mape": 0.20, "r2_score": 0.75},
            },
            model_results={
                "metrics": {"mape": 0.15, "r2_score": 0.85},
            },
            comparison={
                "mape_improvement_pct": 25.0,
                "r2_improvement_pct": 13.3,
                "model_outperforms": True,
            },
            config_used={
                "model_features": ["feature1", "feature2"],
                "start_cutoff": 10,
                "end_cutoff": 20,
            },
        )

    def test_mape_improvement(self, results):
        """mape_improvement returns percentage improvement."""
        assert results.mape_improvement == 25.0

    def test_r2_improvement(self, results):
        """r2_improvement returns percentage improvement."""
        assert results.r2_improvement == 13.3

    def test_model_outperforms(self, results):
        """model_outperforms returns boolean."""
        assert results.model_outperforms is True

    def test_model_mape(self, results):
        """model_mape returns model MAPE score."""
        assert results.model_mape == 0.15

    def test_benchmark_mape(self, results):
        """benchmark_mape returns benchmark MAPE score."""
        assert results.benchmark_mape == 0.20

    def test_model_r2(self, results):
        """model_r2 returns model R2 score."""
        assert results.model_r2 == 0.85

    def test_benchmark_r2(self, results):
        """benchmark_r2 returns benchmark R2 score."""
        assert results.benchmark_r2 == 0.75

    def test_model_features(self, results):
        """model_features returns list of features."""
        assert results.model_features == ["feature1", "feature2"]

    def test_n_forecasts(self, results):
        """n_forecasts returns count of forecast periods."""
        assert results.n_forecasts == 10  # 20 - 10


class TestForecastingResultsPropertiesEdgeCases:
    """Tests for edge cases in ForecastingResults properties."""

    def test_mape_improvement_missing_returns_default(self):
        """Returns 0.0 when mape_improvement_pct missing."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={},  # Missing mape_improvement_pct
            config_used={},
        )

        assert results.mape_improvement == 0.0

    def test_r2_improvement_missing_returns_default(self):
        """Returns 0.0 when r2_improvement_pct missing."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={},  # Missing r2_improvement_pct
            config_used={},
        )

        assert results.r2_improvement == 0.0

    def test_model_outperforms_missing_returns_false(self):
        """Returns False when model_outperforms missing."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={},  # Missing model_outperforms
            config_used={},
        )

        assert results.model_outperforms is False

    def test_model_mape_missing_returns_nan(self):
        """Returns NaN when model metrics missing."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},  # Missing metrics
            comparison={},
            config_used={},
        )

        assert math.isnan(results.model_mape)

    def test_benchmark_mape_missing_returns_nan(self):
        """Returns NaN when benchmark metrics missing."""
        results = ForecastingResults(
            benchmark_results={},  # Missing metrics
            model_results={},
            comparison={},
            config_used={},
        )

        assert math.isnan(results.benchmark_mape)

    def test_model_r2_missing_returns_nan(self):
        """Returns NaN when model r2_score missing."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={"metrics": {}},  # Missing r2_score
            comparison={},
            config_used={},
        )

        assert math.isnan(results.model_r2)

    def test_benchmark_r2_missing_returns_nan(self):
        """Returns NaN when benchmark r2_score missing."""
        results = ForecastingResults(
            benchmark_results={"metrics": {}},  # Missing r2_score
            model_results={},
            comparison={},
            config_used={},
        )

        assert math.isnan(results.benchmark_r2)

    def test_model_features_missing_returns_empty_list(self):
        """Returns empty list when model_features missing."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={},
            config_used={},  # Missing model_features
        )

        assert results.model_features == []

    def test_n_forecasts_missing_cutoffs_returns_zero(self):
        """Returns 0 when cutoffs missing."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={},
            config_used={},  # Missing cutoffs
        )

        assert results.n_forecasts == 0

    def test_n_forecasts_negative_returns_zero(self):
        """Returns 0 when end < start (negative range)."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={},
            config_used={
                "start_cutoff": 20,
                "end_cutoff": 10,  # End before start
            },
        )

        assert results.n_forecasts == 0


class TestForecastingResultsToDict:
    """Tests for ForecastingResults.to_dict() method."""

    def test_to_dict_includes_all_fields(self):
        """to_dict includes all base fields."""
        results = ForecastingResults(
            benchmark_results={"key": "benchmark"},
            model_results={"key": "model"},
            comparison={"mape_improvement_pct": 25.0},
            config_used={"setting": "value"},
        )

        d = results.to_dict()

        assert d["benchmark_results"] == {"key": "benchmark"}
        assert d["model_results"] == {"key": "model"}
        assert d["comparison"] == {"mape_improvement_pct": 25.0}
        assert d["config_used"] == {"setting": "value"}

    def test_to_dict_includes_convenience_fields(self):
        """to_dict includes computed convenience fields."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={
                "mape_improvement_pct": 25.0,
                "r2_improvement_pct": 13.3,
                "model_outperforms": True,
            },
            config_used={},
        )

        d = results.to_dict()

        assert d["mape_improvement"] == 25.0
        assert d["r2_improvement"] == 13.3
        assert d["model_outperforms"] is True

    def test_to_dict_with_defaults(self):
        """to_dict handles missing fields with defaults."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={},
            config_used={},
        )

        d = results.to_dict()

        assert d["mape_improvement"] == 0.0
        assert d["r2_improvement"] == 0.0
        assert d["model_outperforms"] is False


class TestForecastingResultsSummary:
    """Tests for ForecastingResults.summary() method."""

    def test_summary_includes_all_metrics(self):
        """summary() includes all key metrics."""
        results = ForecastingResults(
            benchmark_results={
                "metrics": {"mape": 0.20, "r2_score": 0.75},
            },
            model_results={
                "metrics": {"mape": 0.15, "r2_score": 0.85},
            },
            comparison={
                "mape_improvement_pct": 25.0,
                "r2_improvement_pct": 13.3,
                "model_outperforms": True,
            },
            config_used={
                "model_features": ["feature1", "feature2"],
                "start_cutoff": 10,
                "end_cutoff": 20,
            },
        )

        summary = results.summary()

        assert "ForecastingResults Summary:" in summary
        assert "Forecasts: 10" in summary
        assert "Model MAPE: 0.1500" in summary
        assert "Benchmark MAPE: 0.2000" in summary
        assert "MAPE Improvement: 25.0%" in summary
        assert "Model R2: 0.8500" in summary
        assert "Benchmark R2: 0.7500" in summary
        assert "R2 Improvement: 13.3%" in summary
        assert "Model Outperforms: True" in summary
        assert "Features: feature1, feature2" in summary

    def test_summary_handles_missing_data(self):
        """summary() handles missing data gracefully."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={},
            config_used={},
        )

        summary = results.summary()

        # Should contain nan for missing metrics
        assert "ForecastingResults Summary:" in summary
        assert "Forecasts: 0" in summary
        assert "nan" in summary  # MAPE and R2 will be nan
        assert "Model Outperforms: False" in summary
        assert "Features:" in summary  # Empty features list

    def test_summary_is_string(self):
        """summary() returns a string."""
        results = ForecastingResults(
            benchmark_results={},
            model_results={},
            comparison={},
            config_used={},
        )

        assert isinstance(results.summary(), str)


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_exports_forecasting_results(self):
        """Module exports ForecastingResults."""
        from src.models.forecasting_types import __all__

        assert "ForecastingResults" in __all__

    def test_exports_forecast_metrics(self):
        """Module exports ForecastMetrics."""
        from src.models.forecasting_types import __all__

        assert "ForecastMetrics" in __all__

    def test_all_exports_importable(self):
        """All exports are importable."""
        from src.models.forecasting_types import __all__, ForecastingResults, ForecastMetrics

        assert ForecastingResults is not None
        assert ForecastMetrics is not None
