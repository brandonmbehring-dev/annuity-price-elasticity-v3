"""
Calibration Tests for Annuity Price Elasticity Models.

This package contains Monte Carlo calibration tests that validate:
1. Confidence interval coverage (~95% for 95% CI)
2. Bootstrap stability across seeds
3. Statistical properties of estimators

These tests are computationally intensive and run on a scheduled basis
(not on every CI build).

Run all calibration tests:
    pytest tests/calibration/ -v

Run only coverage tests:
    pytest tests/calibration/ -v -k "coverage"

Decision Reference:
    DL-001: Bootstrap Sample Size (1000 samples) - docs/practices/DECISIONS.md
"""
