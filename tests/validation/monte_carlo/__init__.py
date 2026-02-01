"""
Monte Carlo Validation Tests
=============================

Statistical simulation tests validating model properties:
1. Bootstrap confidence interval coverage
2. Coefficient sign stability across samples
3. Performance metric distribution

Usage:
    pytest tests/validation/monte_carlo/ -v -m monte_carlo

References:
    - src/features/selection/stability/bootstrap_stability_analysis.py
    - src/models/inference_validation.py
"""
