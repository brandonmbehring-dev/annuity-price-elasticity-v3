"""
Anti-Pattern Tests for RILA Elasticity Models
==============================================

This module contains tests specifically designed to catch common anti-patterns
that lead to data leakage, causality violations, or economically invalid models.

The tests are organized by anti-pattern category:
- test_lag0_competitor_detection.py: Catch lag-0 competitor features
- test_coefficient_sign_validation.py: Enforce economic sign constraints
- test_future_leakage.py: Detect future data in features
- test_economic_plausibility.py: Verify realistic coefficient magnitudes

These tests should be run as part of the pre-deployment leakage gate:
    make leakage-audit

Author: Claude Code
Date: 2026-01-30
"""
