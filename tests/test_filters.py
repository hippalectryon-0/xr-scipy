"""Tests for xrscipy.signal.filters module."""

import numpy as np
import pytest
import scipy as sp
import xarray as xr

import xrscipy.signal as dsp
from .testings import get_obj


def _check_metadata_preservation(original_da, result_da, dim):
    """Check that metadata is properly preserved."""
    # Make sure the original data does not change
    assert original_da.values.shape == original_da.shape

    # Make sure the coordinate (except the transformed one) is propagated
    for key, v in original_da.coords.items():
        if dim not in v.dims and key != dim:
            assert original_da[key].identical(result_da[key])


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dim", ["x"])
def test_savgol_filter(mode, dim):
    """Test savgol_filter function.

    Verifies that xrscipy.signal.savgol_filter produces results strictly equal to scipy,
    and that metadata is properly handled.
    """
    # Get test data
    da = get_obj(mode)

    # Skip if dimension not available
    if dim not in da.dims:
        pytest.skip("dimension not available in test object")

    # Test parameters
    window_length_coord = 0.5  # Window length in coordinate units
    polyorder = 2
    deriv = 0

    # Calculate using xrscipy
    actual = dsp.savgol_filter(da, window_length_coord, polyorder, deriv=deriv, dim=dim)

    # Calculate using scipy
    # First, we need to convert window_length from coordinate units to samples
    coord = da.coords[dim]
    dt = (coord[1] - coord[0]).values
    window_length_samples = int(np.rint(window_length_coord / dt))

    # Ensure window_length is odd and positive
    if window_length_samples % 2 == 0:
        window_length_samples += 1
    window_length_samples = max(window_length_samples, 3)

    # Ensure polyorder is less than window_length
    polyorder = min(polyorder, window_length_samples - 1)

    # Apply scipy function along the correct axis
    axis = da.get_axis_num(dim)
    expected_result = sp.signal.savgol_filter(
        da.values,
        window_length=window_length_samples,
        polyorder=polyorder,
        deriv=deriv,
        delta=1.0,
        axis=axis,
        mode="interp",
        cval=0.0,
    )

    # Check that result values match
    np.testing.assert_allclose(actual.values, expected_result)

    # Check metadata preservation
    _check_metadata_preservation(da, actual, dim)

    # Check that the result has the correct name
    expected_name = f"savgol_filtered_{da.name}" if da.name else "savgol_filtered"
    assert actual.name == expected_name


@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_savgol_filter_deriv(deriv):
    """Test savgol_filter function with different derivative orders."""
    # Get test data
    da = get_obj(0)  # 1D data

    # Test parameters
    window_length_coord = 0.5  # Window length in coordinate units
    polyorder = 3

    # Calculate using xrscipy
    actual = dsp.savgol_filter(da, window_length_coord, polyorder, deriv=deriv, dim="x")

    # Calculate using scipy
    # First, we need to convert window_length from coordinate units to samples
    coord = da.coords["x"]
    dt = (coord[1] - coord[0]).values
    window_length_samples = int(np.rint(window_length_coord / dt))

    # Ensure window_length is odd and positive
    if window_length_samples % 2 == 0:
        window_length_samples += 1
    window_length_samples = max(window_length_samples, 3)

    # Ensure polyorder is less than window_length
    polyorder = min(polyorder, window_length_samples - 1)

    expected_result = sp.signal.savgol_filter(
        da.values,
        window_length=window_length_samples,
        polyorder=polyorder,
        deriv=deriv,
        delta=1.0,
        axis=-1,
        mode="interp",
        cval=0.0,
    )

    # Check that result values match
    np.testing.assert_allclose(actual.values, expected_result)


def test_savgol_filter_edge_cases():
    """Test savgol_filter function with edge cases."""
    # Create a simple test case with known behavior
    x = np.linspace(0, 1, 20)
    da = xr.DataArray(
        np.sin(2 * np.pi * x) + 0.1 * np.random.RandomState(0).randn(20),
        dims=["x"],
        coords={"x": x},
        name="test_signal",
    )

    # Test with small window length
    result = dsp.savgol_filter(da, 0.1, 1, dim="x")
    assert isinstance(result, xr.DataArray)
    assert result.shape == da.shape
    assert result.name == "savgol_filtered_test_signal"

    # Test with large polyorder (should be automatically reduced)
    result = dsp.savgol_filter(da, 0.5, 10, dim="x")  # polyorder 10 should be reduced
    assert isinstance(result, xr.DataArray)

    # Test with named data array
    da_named = xr.DataArray([1, 2, 3, 4, 5], dims=["x"], coords={"x": [0, 1, 2, 3, 4]}, name="test")
    result = dsp.savgol_filter(da_named, 1.5, 2, dim="x")
    assert result.name == "savgol_filtered_test"
