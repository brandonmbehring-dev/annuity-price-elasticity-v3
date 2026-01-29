"""
Property-Based Testing Suite for Annuity Price Elasticity.

Uses Hypothesis to generate test cases that verify invariants and properties
that should hold across the entire input space, not just specific examples.

Test Categories:
- Rate transforms: Bounds, monotonicity, invertibility
- DataFrame invariants: Shape preservation, type stability, no NaN introduction
- Statistical constraints: Coefficient signs, R2 bounds, variance positivity
- Pipeline idempotency: Same input produces same output
- Temporal freshness: Different days produce different data

Usage:
    pytest tests/property_based/ -v -m property
    make test-property
"""

__all__ = [
    "test_rate_transforms",
    "test_dataframe_invariants",
    "test_statistical_constraints",
    "test_pipeline_idempotency",
]
