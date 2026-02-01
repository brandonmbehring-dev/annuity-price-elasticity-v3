"""
Known-Answer Tests for Annuity Price Elasticity
================================================

This package validates model outputs against:
1. **Literature bounds** - Published research on annuity price sensitivity [T1]
2. **Golden reference values** - Frozen outputs for regression detection [T2]
3. **Coefficient sign constraints** - Economic theory expectations [T1]
4. **R-squared calibration** - Expected model performance ranges [T2]

Knowledge Tier Tags:
    [T1] = Academically validated (with citation)
    [T2] = Empirical finding from prior work
    [T3] = Assumption needing domain justification

References:
    - LIMRA (2023) "Price Sensitivity in Annuity Markets"
    - SEC Release No. 34-72685 (2014) - RILA Regulatory Framework
    - Production baseline: 02_time_series_forecasting_refactored.ipynb

Usage:
    pytest tests/known_answer/ -v -m known_answer
"""
