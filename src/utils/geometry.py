"""
N-Sphere Volume Calculations

This module provides mathematical functions for calculating volumes of N-dimensional spheres.

Business Context:
    While not directly related to price elasticity modeling, this demonstrates proper
    function design patterns for mathematical operations following CODING_STANDARDS.md §3.
"""

import math
from typing import Union


def _validate_radius_type(radius: Union[int, float]) -> None:
    """Validate radius type is numeric.

    Parameters
    ----------
    radius : Union[int, float]
        Radius value to validate

    Raises
    ------
    TypeError
        If radius is not int or float
    """
    if not isinstance(radius, (int, float)):
        raise TypeError(
            f"Radius must be a number (int or float), got {type(radius).__name__}. "
            "Business impact: Cannot calculate volume with non-numeric radius. "
            "Action required: Provide numeric radius value."
        )


def _validate_dimension_type(dimension: int) -> None:
    """Validate dimension type is integer.

    Parameters
    ----------
    dimension : int
        Dimension value to validate

    Raises
    ------
    TypeError
        If dimension is not int
    """
    if not isinstance(dimension, int):
        raise TypeError(
            f"Dimension must be an integer, got {type(dimension).__name__}. "
            "Business impact: N-sphere dimension must be a whole number. "
            "Action required: Provide integer dimension value."
        )


def _validate_n_sphere_inputs(radius: float, dimension: int) -> None:
    """Validate N-sphere inputs for value constraints.

    Parameters
    ----------
    radius : float
        N-sphere radius
    dimension : int
        N-sphere dimension

    Raises
    ------
    ValueError
        If radius or dimension is negative
    """
    if radius < 0:
        raise ValueError(
            f"Radius must be non-negative, got {radius}. "
            "Business impact: Negative radius has no geometric meaning. "
            "Action required: Provide non-negative radius value."
        )

    if dimension < 0:
        raise ValueError(
            f"Dimension must be non-negative, got {dimension}. "
            "Business impact: Negative dimension has no geometric meaning. "
            "Action required: Provide non-negative dimension value."
        )


def _compute_n_sphere_volume(radius: float, dimension: int) -> float:
    """Compute N-sphere volume using gamma function formula.

    V(n, r) = (pi^(n/2) / Gamma(n/2 + 1)) * r^n

    Parameters
    ----------
    radius : float
        N-sphere radius
    dimension : int
        N-sphere dimension

    Returns
    -------
    float
        Computed volume
    """
    n_half = dimension / 2.0
    numerator = math.pi ** n_half
    denominator = math.gamma(n_half + 1)
    coefficient = numerator / denominator
    return coefficient * (radius ** dimension)


def calculate_n_sphere_volume(radius: float, dimension: int) -> float:
    """
    Calculate the volume of an N-dimensional sphere (N-sphere).

    V(n, r) = (pi^(n/2) / Gamma(n/2 + 1)) * r^n

    Parameters
    ----------
    radius : float
        The radius of the N-sphere. Must be non-negative.
    dimension : int
        The dimension N (N=2: circle, N=3: sphere, N>=4: hypersphere)

    Returns
    -------
    float
        The volume of the N-sphere.

    Raises
    ------
    ValueError
        If radius or dimension is negative.
    TypeError
        If radius is not a number or dimension is not an integer.

    Examples
    --------
    >>> calculate_n_sphere_volume(1.0, 2)  # Unit circle
    3.141592653589793

    >>> calculate_n_sphere_volume(2.0, 3)  # Sphere with radius 2
    33.510321638291124
    """
    _validate_radius_type(radius)
    _validate_dimension_type(dimension)
    _validate_n_sphere_inputs(radius, dimension)

    return _compute_n_sphere_volume(radius, dimension)
