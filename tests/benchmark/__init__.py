"""
Benchmark Tests for RILA Elasticity Models
==========================================

This module contains benchmark tests that validate model outputs against
known baselines and economic expectations.

Benchmark Categories:
1. Elasticity Bounds: Verify coefficients fall within expected ranges
2. Constraint Consistency: Verify economic constraints are satisfied
3. Performance Baselines: Track model fit metrics over time
4. Regression Detection: Catch unintended changes

Usage:
    pytest tests/benchmark/ -v
    make test-benchmark

Author: Claude Code
Date: 2026-01-30
"""
