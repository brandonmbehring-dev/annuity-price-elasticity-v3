"""
Shared Types for Multiple Testing Correction.

This module contains shared data structures used by all multiple testing
correction modules.

Used by: bonferroni_engine.py, fdr_engine.py, multiple_testing_correction.py
"""

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class MultipleTestingResults:
    """
    Container for multiple testing correction results.

    Attributes
    ----------
    method : str
        Correction method used ('bonferroni', 'fdr', 'reduced_space')
    original_alpha : float
        Original significance level (typically 0.05)
    corrected_alpha : float
        Corrected significance level
    n_tests : int
        Number of statistical tests performed
    significant_models : pd.DataFrame
        Models passing corrected significance threshold
    rejected_models : pd.DataFrame
        Models rejected by correction
    correction_impact : Dict[str, Any]
        Impact assessment of correction method
    statistical_power : Dict[str, float]
        Power analysis results
    """
    method: str
    original_alpha: float
    corrected_alpha: float
    n_tests: int
    significant_models: pd.DataFrame
    rejected_models: pd.DataFrame
    correction_impact: Dict[str, Any]
    statistical_power: Dict[str, float]
