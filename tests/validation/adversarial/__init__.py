"""
Adversarial Validation Tests
=============================

Edge case stress tests for robustness:
1. Extreme rate values (0%, 100%)
2. Missing data patterns
3. Outlier handling
4. Boundary conditions

Usage:
    pytest tests/validation/adversarial/ -v -m adversarial

References:
    - knowledge/practices/LEAKAGE_CHECKLIST.md
    - src/validation/production_validators.py
"""
