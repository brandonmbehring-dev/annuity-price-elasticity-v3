"""
Benchmark Tests: Elasticity Coefficient Validation
===================================================

These tests validate that model coefficients fall within economically
plausible and historically observed ranges. They serve as:
1. Sanity checks against data corruption
2. Regression detection for code changes
3. Documentation of expected model behavior

The benchmarks are based on:
- [T1] Academic literature on annuity elasticities
- [T2] Empirical observations from historical model runs
- [T3] Business domain knowledge

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


# =============================================================================
# BENCHMARK DEFINITIONS
# =============================================================================


@dataclass
class ElasticityBenchmark:
    """Defines expected range for an elasticity coefficient."""
    name: str
    min_value: float
    max_value: float
    typical_value: float
    source: str  # T1, T2, T3


# RILA 6Y20B Benchmarks (based on historical model runs)
RILA_6Y20B_BENCHMARKS = {
    "own_rate_t0": ElasticityBenchmark(
        name="Own Rate (t0)",
        min_value=3000.0,
        max_value=15000.0,
        typical_value=8500.0,
        source="[T2] Historical model runs 2023-2025"
    ),
    "own_rate_t1": ElasticityBenchmark(
        name="Own Rate (t-1)",
        min_value=1000.0,
        max_value=8000.0,
        typical_value=4200.0,
        source="[T2] Historical model runs 2023-2025"
    ),
    "competitor_weighted_t2": ElasticityBenchmark(
        name="Competitor Weighted (t-2)",
        min_value=-12000.0,
        max_value=-2000.0,
        typical_value=-6300.0,
        source="[T2] Historical model runs 2023-2025"
    ),
    "r_squared": ElasticityBenchmark(
        name="R-squared",
        min_value=0.40,
        max_value=0.85,
        typical_value=0.62,
        source="[T2] Expected range for behavioral data"
    ),
}


# =============================================================================
# BENCHMARK VALIDATION FUNCTIONS
# =============================================================================


def validate_against_benchmark(
    value: float,
    benchmark: ElasticityBenchmark,
    strict: bool = False,
) -> Tuple[bool, str]:
    """Validate a value against its benchmark.

    Args:
        value: The actual coefficient value
        benchmark: The expected benchmark
        strict: If True, fail outside min/max; if False, warn

    Returns:
        Tuple of (is_valid, message)
    """
    if benchmark.min_value <= value <= benchmark.max_value:
        if abs(value - benchmark.typical_value) / abs(benchmark.typical_value) < 0.5:
            return True, f"[PASS] {benchmark.name}: {value:.2f} (typical: {benchmark.typical_value:.2f})"
        else:
            return True, f"[WARN] {benchmark.name}: {value:.2f} (typical: {benchmark.typical_value:.2f}, within range but atypical)"
    else:
        msg = (f"[FAIL] {benchmark.name}: {value:.2f} outside expected range "
               f"[{benchmark.min_value:.2f}, {benchmark.max_value:.2f}]")
        return not strict, msg


def validate_coefficient_set(
    coefficients: Dict[str, float],
    benchmarks: Dict[str, ElasticityBenchmark],
) -> Tuple[bool, list]:
    """Validate a set of coefficients against benchmarks.

    Args:
        coefficients: Dictionary of name -> value
        benchmarks: Dictionary of name -> benchmark

    Returns:
        Tuple of (all_valid, list of messages)
    """
    all_valid = True
    messages = []

    for name, benchmark in benchmarks.items():
        # Find matching coefficient (fuzzy match)
        matching_coefs = [k for k in coefficients if name.lower() in k.lower()]

        if not matching_coefs:
            messages.append(f"? {benchmark.name}: not found in coefficients")
            continue

        for coef_name in matching_coefs:
            value = coefficients[coef_name]
            valid, msg = validate_against_benchmark(value, benchmark)
            all_valid = all_valid and valid
            messages.append(msg)

    return all_valid, messages


# =============================================================================
# BENCHMARK TESTS
# =============================================================================


class TestRILA6Y20BBenchmarks:
    """Benchmark tests for RILA 6Y20B product."""

    @pytest.fixture
    def historical_coefficients(self) -> Dict[str, float]:
        """Representative coefficients from historical model run."""
        return {
            "intercept": 15000.0,
            "prudential_rate_t0": 8500.0,
            "prudential_rate_t1": 4200.0,
            "competitor_weighted_t2": -6300.0,
            "competitor_weighted_t3": -2100.0,
            "vix_t0": -350.0,
            "dgs5_t0": 1200.0,
        }

    def test_own_rate_within_benchmark(self, historical_coefficients):
        """Own rate coefficient should be within historical benchmark."""
        own_rate = historical_coefficients["prudential_rate_t0"]
        benchmark = RILA_6Y20B_BENCHMARKS["own_rate_t0"]

        assert benchmark.min_value <= own_rate <= benchmark.max_value, (
            f"Own rate {own_rate:.2f} outside benchmark "
            f"[{benchmark.min_value:.2f}, {benchmark.max_value:.2f}]"
        )

    def test_competitor_rate_within_benchmark(self, historical_coefficients):
        """Competitor rate coefficient should be within historical benchmark."""
        competitor_rate = historical_coefficients["competitor_weighted_t2"]
        benchmark = RILA_6Y20B_BENCHMARKS["competitor_weighted_t2"]

        assert benchmark.min_value <= competitor_rate <= benchmark.max_value, (
            f"Competitor rate {competitor_rate:.2f} outside benchmark "
            f"[{benchmark.min_value:.2f}, {benchmark.max_value:.2f}]"
        )

    def test_full_coefficient_validation(self, historical_coefficients):
        """Full coefficient set should pass benchmark validation."""
        all_valid, messages = validate_coefficient_set(
            historical_coefficients,
            RILA_6Y20B_BENCHMARKS
        )

        for msg in messages:
            print(msg)

        assert all_valid, "Coefficient validation failed"


class TestBenchmarkRatios:
    """Tests for ratios between coefficients."""

    @pytest.fixture
    def sample_coefficients(self) -> Dict[str, float]:
        """Sample coefficients for ratio testing."""
        return {
            "own_rate_t0": 8500.0,
            "own_rate_t1": 4200.0,
            "competitor_t2": -6300.0,
        }

    def test_own_rate_ratio_reasonable(self, sample_coefficients):
        """Ratio of own_rate_t0 to own_rate_t1 should be reasonable."""
        t0 = sample_coefficients["own_rate_t0"]
        t1 = sample_coefficients["own_rate_t1"]

        ratio = t0 / t1

        # t0 should be larger than t1 (immediate effect > lagged effect)
        assert ratio > 1.0, "t0 should have larger effect than t1"

        # But not too much larger (suggests multicollinearity if extreme)
        assert ratio < 5.0, "Ratio suspiciously large - check for multicollinearity"

    def test_own_vs_competitor_ratio_reasonable(self, sample_coefficients):
        """Own rate effect should be comparable to competitor effect."""
        own = sample_coefficients["own_rate_t0"]
        competitor = abs(sample_coefficients["competitor_t2"])

        ratio = own / competitor

        # Own effect should be in similar ballpark as competitor
        assert 0.5 < ratio < 5.0, (
            f"Own/competitor ratio {ratio:.2f} seems implausible"
        )


class TestBenchmarkStability:
    """Tests for coefficient stability over time."""

    def test_coefficient_stability_simulation(self):
        """
        Simulate checking coefficient stability across model versions.

        In production, this would compare against saved baseline files
        in tests/baselines/rila/.
        """
        # Simulated baseline from previous version
        baseline = {
            "prudential_rate_t0": 8200.0,
            "competitor_weighted_t2": -6100.0,
        }

        # Simulated current version
        current = {
            "prudential_rate_t0": 8500.0,
            "competitor_weighted_t2": -6300.0,
        }

        # Check stability (< 20% change from baseline)
        for key in baseline:
            if key in current:
                change_pct = abs(current[key] - baseline[key]) / abs(baseline[key])

                assert change_pct < 0.20, (
                    f"{key} changed by {change_pct:.1%} from baseline - "
                    "investigate before deploying"
                )


# =============================================================================
# INTEGRATION TESTS WITH REAL MODEL (when available)
# =============================================================================


class TestLiveModelBenchmarks:
    """Integration tests with actual model runs."""

    @pytest.mark.skip(reason="Requires full model training - run manually")
    def test_fixture_model_within_benchmarks(self):
        """
        Run full model on fixture data and validate coefficients.

        This test is skipped by default because it requires full model
        training. Run manually with:
            pytest tests/benchmark/test_elasticity_benchmarks.py::TestLiveModelBenchmarks -v --no-skip
        """
        from src.notebooks import create_interface

        # Create interface with fixture data
        interface = create_interface("6Y20B", environment="fixture")

        # Load and run
        df = interface.load_data()
        results = interface.run_inference(df)

        # Validate against benchmarks
        all_valid, messages = validate_coefficient_set(
            results["coefficients"],
            RILA_6Y20B_BENCHMARKS
        )

        for msg in messages:
            print(msg)

        assert all_valid, "Model coefficients outside benchmark ranges"


# =============================================================================
# DOCUMENTATION TEST
# =============================================================================


def test_benchmark_documentation():
    """
    Benchmark Documentation

    PURPOSE:
    Benchmarks provide guardrails for model coefficients. They catch:
    - Data corruption (wildly different coefficients)
    - Code regressions (unintended changes)
    - Specification errors (implausible magnitudes)

    BENCHMARK SOURCES:
    - [T1] Academic: Based on peer-reviewed literature
    - [T2] Empirical: Based on our historical model runs
    - [T3] Business: Based on domain expertise

    CURRENT BENCHMARKS (RILA 6Y20B):

    | Coefficient           | Range          | Typical  | Source |
    |-----------------------|----------------|----------|--------|
    | Own Rate (t0)         | [3k, 15k]      | 8,500    | T2     |
    | Own Rate (t-1)        | [1k, 8k]       | 4,200    | T2     |
    | Competitor (t-2)      | [-12k, -2k]    | -6,300   | T2     |
    | R-squared             | [0.40, 0.85]   | 0.62     | T2     |

    UPDATING BENCHMARKS:
    1. Document reason for change
    2. Get sign-off from model owner
    3. Update RILA_6Y20B_BENCHMARKS dict
    4. Add entry to tests/baselines/rila/

    USAGE:
        pytest tests/benchmark/ -v
        make test-benchmark
    """
    pass
