"""Tests for xrscipy.signal.filters module."""

import numpy as np
import pytest
import scipy as sp
import xarray as xr

import xrscipy.signal as dsp
from .testings import get_obj


def _check_metadata_preservation(original_da, result_da, dim, preserve_filtered_dim=True):
    """Check that metadata is properly preserved.

    Parameters
    ----------
    original_da : xr.DataArray
        Original data array
    result_da : xr.DataArray
        Result data array
    dim : str
        The dimension along which the operation was performed
    preserve_filtered_dim : bool, optional
        Whether the coordinates along the filtered dimension should be preserved.
        For operations like FFT, this should be False. For operations like
        savgol_filter and decimate, this should be True. Default is True.
    """
    # Make sure the original data does not change
    assert original_da.values.shape == original_da.shape

    # Make sure coordinates are preserved
    for key, v in original_da.coords.items():
        if dim not in v.dims:  # Check coordinates not along the processed dimension
            assert original_da[key].identical(result_da[key])
        elif preserve_filtered_dim and key == dim:  # Check the filtered dimension if it should be preserved
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

    # Check metadata preservation (coordinates along filtered dimension should be preserved)
    _check_metadata_preservation(da, actual, dim, preserve_filtered_dim=True)

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


@pytest.mark.parametrize("q", [2, 5])
def test_decimate(q):
    """Test decimate function.

    Verifies that xrscipy.signal.decimate produces results strictly equal to scipy,
    and that metadata is properly handled.
    """
    # Create test data
    x = np.linspace(0, 1, 100)
    da = xr.DataArray(
        np.sin(2 * np.pi * 5 * x) + 0.1 * np.random.RandomState(0).randn(100),
        dims=["x"],
        coords={"x": x},
        name="test_signal",
    )

    # Calculate using xrscipy
    actual = dsp.decimate(da, q=q, dim="x")

    # Calculate using scipy
    expected_result = sp.signal.decimate(
        da.values,
        q=q,
        n=None,
        ftype="iir",
        axis=-1,
        zero_phase=True,
    )

    # Check that result values match
    np.testing.assert_allclose(actual.values, expected_result)

    # Check that coordinates are properly subsampled
    expected_x_coords = da.coords["x"].values[::q]
    actual_x_coords = actual.coords["x"].values
    np.testing.assert_allclose(actual_x_coords, expected_x_coords)

    # Check that non-filtered coordinates are preserved
    _check_metadata_preservation(da, actual, "x", preserve_filtered_dim=False)

    # Check that the result has the correct name
    expected_name = f"decimated_{da.name}" if da.name else "decimated"
    assert actual.name == expected_name

    # Check that the dimension has been properly decimated
    original_size = da.sizes["x"]
    expected_size = original_size // q
    assert actual.sizes["x"] == expected_size


def test_decimate_with_target_fs():
    """Test decimate function with target_fs parameter."""
    # Create test data
    x = np.linspace(0, 1, 100)
    da = xr.DataArray(
        np.sin(2 * np.pi * 5 * x) + 0.1 * np.random.RandomState(0).randn(100),
        dims=["x"],
        coords={"x": x},
        name="test_signal",
    )

    # Test parameters
    target_fs = 10.0

    # Calculate using xrscipy
    actual = dsp.decimate(da, target_fs=target_fs, dim="x")

    # Calculate expected q value
    coord = da.coords["x"]
    dt = (coord[1] - coord[0]).values
    current_fs = 1.0 / dt
    expected_q = int(np.rint(current_fs / target_fs))

    # Calculate using scipy with the same q
    expected_result = sp.signal.decimate(
        da.values,
        q=expected_q,
        n=None,
        ftype="iir",
        axis=-1,
        zero_phase=True,
    )

    # Check that result values match
    np.testing.assert_allclose(actual.values, expected_result)

    # Check that coordinates are properly subsampled
    expected_x_coords = da.coords["x"].values[::expected_q]
    actual_x_coords = actual.coords["x"].values
    np.testing.assert_allclose(actual_x_coords, expected_x_coords)

    # Check that non-filtered coordinates are preserved
    _check_metadata_preservation(da, actual, "x", preserve_filtered_dim=False)

    # Verify that the function worked with target_fs
    original_size = da.sizes["x"]
    expected_size = original_size // expected_q
    assert actual.sizes["x"] == expected_size

    # Check that the result has the correct name
    expected_name = f"decimated_{da.name}" if da.name else "decimated"
    assert actual.name == expected_name


def test_decimate_edge_cases():
    """Test decimate function with edge cases."""
    # Create a simple test case with known behavior
    x = np.linspace(0, 1, 30)
    da = xr.DataArray(
        np.sin(2 * np.pi * x) + 0.1 * np.random.RandomState(0).randn(30),
        dims=["x"],
        coords={"x": x},
        name="test_signal",
    )

    # Test with small q
    result = dsp.decimate(da, q=2, dim="x")
    expected = sp.signal.decimate(da.values, q=2, n=None, ftype="iir", axis=-1, zero_phase=True)
    np.testing.assert_allclose(result.values, expected)
    assert result.shape[0] == 15  # 30 // 2
    assert result.name == "decimated_test_signal"

    # Test with named data array (use FIR filter for short signals)
    da_named = xr.DataArray([1, 2, 3, 4, 5, 6], dims=["x"], coords={"x": [0, 1, 2, 3, 4, 5]}, name="test")
    result = dsp.decimate(da_named, q=2, dim="x")
    # For the comparison, we need to use the same parameters that xrscipy uses internally
    expected = sp.signal.decimate(da_named.values, q=2, n=4, ftype="fir", axis=-1, zero_phase=True)
    np.testing.assert_allclose(result.values, expected)
    assert result.name == "decimated_test"
    assert result.shape[0] == 3  # 6 // 2

    # Test error when neither q nor target_fs is provided
    with pytest.raises(ValueError, match="Either 'q' or 'target_fs' must be specified"):
        dsp.decimate(da, dim="x")
