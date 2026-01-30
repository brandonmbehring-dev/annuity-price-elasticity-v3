"""
Tests for N-sphere volume calculation utilities.

Tests cover:
- Type validation for radius and dimension
- Value validation for negative inputs
- Mathematical correctness of volume formula
- Edge cases (zero dimension, zero radius)
"""

import pytest
import math

from src.utils.geometry import (
    calculate_n_sphere_volume,
    _validate_radius_type,
    _validate_dimension_type,
    _validate_n_sphere_inputs,
    _compute_n_sphere_volume,
)


# =============================================================================
# Tests: _validate_radius_type
# =============================================================================


class TestValidateRadiusType:
    """Tests for radius type validation."""

    def test_accepts_int(self):
        """Test accepts integer radius."""
        _validate_radius_type(5)  # Should not raise

    def test_accepts_float(self):
        """Test accepts float radius."""
        _validate_radius_type(3.14)  # Should not raise

    def test_rejects_string(self):
        """Test rejects string radius."""
        with pytest.raises(TypeError, match="must be a number"):
            _validate_radius_type("five")

    def test_rejects_none(self):
        """Test rejects None radius."""
        with pytest.raises(TypeError, match="must be a number"):
            _validate_radius_type(None)

    def test_rejects_list(self):
        """Test rejects list radius."""
        with pytest.raises(TypeError, match="must be a number"):
            _validate_radius_type([5])


# =============================================================================
# Tests: _validate_dimension_type
# =============================================================================


class TestValidateDimensionType:
    """Tests for dimension type validation."""

    def test_accepts_int(self):
        """Test accepts integer dimension."""
        _validate_dimension_type(3)  # Should not raise

    def test_rejects_float(self):
        """Test rejects float dimension."""
        with pytest.raises(TypeError, match="must be an integer"):
            _validate_dimension_type(3.0)

    def test_rejects_string(self):
        """Test rejects string dimension."""
        with pytest.raises(TypeError, match="must be an integer"):
            _validate_dimension_type("three")


# =============================================================================
# Tests: _validate_n_sphere_inputs
# =============================================================================


class TestValidateNSphereInputs:
    """Tests for N-sphere input value validation."""

    def test_accepts_valid_inputs(self):
        """Test accepts valid positive values."""
        _validate_n_sphere_inputs(1.0, 3)  # Should not raise

    def test_accepts_zero_radius(self):
        """Test accepts zero radius."""
        _validate_n_sphere_inputs(0.0, 3)  # Should not raise

    def test_accepts_zero_dimension(self):
        """Test accepts zero dimension."""
        _validate_n_sphere_inputs(1.0, 0)  # Should not raise

    def test_rejects_negative_radius(self):
        """Test rejects negative radius."""
        with pytest.raises(ValueError, match="Radius must be non-negative"):
            _validate_n_sphere_inputs(-1.0, 3)

    def test_rejects_negative_dimension(self):
        """Test rejects negative dimension."""
        with pytest.raises(ValueError, match="Dimension must be non-negative"):
            _validate_n_sphere_inputs(1.0, -1)


# =============================================================================
# Tests: _compute_n_sphere_volume
# =============================================================================


class TestComputeNSphereVolume:
    """Tests for volume computation formula."""

    def test_unit_circle_area(self):
        """Test area of unit circle (2D, r=1) equals pi."""
        result = _compute_n_sphere_volume(1.0, 2)
        assert result == pytest.approx(math.pi, rel=1e-12)

    def test_unit_sphere_volume(self):
        """Test volume of unit sphere (3D, r=1) equals 4/3 * pi."""
        result = _compute_n_sphere_volume(1.0, 3)
        expected = (4.0 / 3.0) * math.pi
        assert result == pytest.approx(expected, rel=1e-12)

    def test_zero_dimension_volume_is_one(self):
        """Test 0D 'sphere' (point) has volume 1 regardless of radius."""
        # V(0, r) = pi^0 / Gamma(1) * r^0 = 1 / 1 * 1 = 1
        result = _compute_n_sphere_volume(5.0, 0)
        assert result == pytest.approx(1.0, rel=1e-12)

    def test_zero_radius_volume_is_zero(self):
        """Test zero radius gives zero volume (for dimension > 0)."""
        result = _compute_n_sphere_volume(0.0, 3)
        assert result == pytest.approx(0.0)

    def test_radius_scaling(self):
        """Test volume scales as r^n."""
        # For dimension n, V(r) = V(1) * r^n
        unit_volume = _compute_n_sphere_volume(1.0, 3)
        scaled_volume = _compute_n_sphere_volume(2.0, 3)
        expected = unit_volume * (2.0 ** 3)
        assert scaled_volume == pytest.approx(expected, rel=1e-12)


# =============================================================================
# Tests: calculate_n_sphere_volume (Integration)
# =============================================================================


class TestCalculateNSphereVolume:
    """Integration tests for main volume calculation function."""

    def test_unit_circle(self):
        """Test unit circle area matches pi."""
        result = calculate_n_sphere_volume(1.0, 2)
        assert result == pytest.approx(math.pi, rel=1e-12)

    def test_unit_sphere(self):
        """Test unit sphere volume matches 4/3 * pi."""
        result = calculate_n_sphere_volume(1.0, 3)
        expected = (4.0 / 3.0) * math.pi
        assert result == pytest.approx(expected, rel=1e-12)

    def test_sphere_radius_2(self):
        """Test sphere with radius 2 (from docstring example)."""
        result = calculate_n_sphere_volume(2.0, 3)
        # V = 4/3 * pi * r^3 = 4/3 * pi * 8 = 32/3 * pi
        expected = (4.0 / 3.0) * math.pi * 8
        assert result == pytest.approx(expected, rel=1e-12)
        # Also verify against docstring value
        assert result == pytest.approx(33.510321638291124, rel=1e-12)

    def test_4d_hypersphere(self):
        """Test 4D hypersphere volume."""
        result = calculate_n_sphere_volume(1.0, 4)
        # V(4, 1) = pi^2 / 2
        expected = (math.pi ** 2) / 2
        assert result == pytest.approx(expected, rel=1e-12)

    def test_integer_radius_accepted(self):
        """Test integer radius is accepted."""
        result = calculate_n_sphere_volume(2, 3)
        expected = (4.0 / 3.0) * math.pi * 8
        assert result == pytest.approx(expected, rel=1e-12)

    def test_type_error_string_radius(self):
        """Test raises TypeError for string radius."""
        with pytest.raises(TypeError, match="must be a number"):
            calculate_n_sphere_volume("1.0", 3)

    def test_type_error_float_dimension(self):
        """Test raises TypeError for float dimension."""
        with pytest.raises(TypeError, match="must be an integer"):
            calculate_n_sphere_volume(1.0, 3.0)

    def test_value_error_negative_radius(self):
        """Test raises ValueError for negative radius."""
        with pytest.raises(ValueError, match="non-negative"):
            calculate_n_sphere_volume(-1.0, 3)

    def test_value_error_negative_dimension(self):
        """Test raises ValueError for negative dimension."""
        with pytest.raises(ValueError, match="non-negative"):
            calculate_n_sphere_volume(1.0, -1)


# =============================================================================
# Tests: Mathematical Properties
# =============================================================================


class TestMathematicalProperties:
    """Tests verifying mathematical properties of N-sphere volumes."""

    def test_volume_increases_with_radius(self):
        """Test volume strictly increases with radius (for n > 0)."""
        v1 = calculate_n_sphere_volume(1.0, 3)
        v2 = calculate_n_sphere_volume(2.0, 3)
        v3 = calculate_n_sphere_volume(3.0, 3)

        assert v1 < v2 < v3

    def test_1d_length(self):
        """Test 1D 'sphere' (line segment) has length 2r."""
        # V(1, r) = 2r (diameter of interval [-r, r])
        result = calculate_n_sphere_volume(5.0, 1)
        expected = 2 * 5.0
        assert result == pytest.approx(expected, rel=1e-12)

    def test_high_dimension_convergence(self):
        """Test that unit sphere volume decreases for high dimensions."""
        # For fixed radius, volume eventually decreases with dimension
        volumes = [calculate_n_sphere_volume(1.0, n) for n in range(1, 20)]

        # Find the peak (occurs around n=5 for r=1)
        peak_idx = volumes.index(max(volumes))
        assert peak_idx > 0  # Peak not at dimension 1

        # After peak, volumes should decrease
        assert volumes[-1] < volumes[peak_idx]

    def test_precision_large_dimension(self):
        """Test precision is maintained for larger dimensions."""
        # Just verify it computes without error and returns positive value
        result = calculate_n_sphere_volume(1.0, 100)
        assert result > 0
        assert math.isfinite(result)
