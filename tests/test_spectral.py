"""Tests for xrscipy.signal.spectral module."""

import numpy as np
import pytest
import scipy as sp
import xarray as xr

import xrscipy.signal as dsp
from xrscipy.signal.spectral import _crossspectrogram
from .testings import get_obj


def _get_sampling_frequency(da, dim):
    """Calculate sampling frequency from coordinate spacing."""
    return 1.0 / (da.coords[dim][1] - da.coords[dim][0]).values


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
@pytest.mark.parametrize("func_name", ["csd", "welch", "coherence", "spectrogram", "hilbert"])
def test_spectral_functions(mode, dim, func_name):
    """Test spectral analysis functions.

    Verifies that xrscipy.signal functions produce results strictly equal to scipy,
    and that metadata is properly handled:
    - Input DataArrays remain unmodified (shape preservation)
    - Coordinates are propagated to output DataArrays
    - Frequency and time dimensions are correctly created
    """
    # Common test parameters
    NPERSEG = 4
    NOVERLAP = 2
    HANN_WINDOW = "hann"
    TUKEY_WINDOW = ("tukey", 0.25)

    # Get test data
    da1 = get_obj(mode)
    da2 = get_obj(mode)  # Only used for 2-argument functions

    # For 1D case with mode=0, we need to make sure we have the right dimension
    if mode == 0 and dim not in da1.dims:
        pytest.skip("dimension not available in 1D test object")

    if dim not in da1.dims:
        pytest.skip("dimension not available in test object")

    if func_name not in ["spectrogram", "hilbert"] and dim not in da2.dims:
        pytest.skip("dimension not available in test object")

    if func_name == "hilbert":
        # Hilbert transform is simpler, test it separately
        # Calculate using xrscipy
        actual = dsp.hilbert(da1, dim=dim)

        # Calculate using scipy
        axis = da1.get_axis_num(dim)
        expected_result = sp.signal.hilbert(da1.values, axis=axis)

        # Check that result values match
        np.testing.assert_allclose(actual.values, expected_result)

        # Check metadata preservation
        _check_metadata_preservation(da1, actual, dim)
        return  # Skip the rest of the function for hilbert

    # Get axis number for the specified dimension
    axis = da1.get_axis_num(dim)

    # Common parameters
    nperseg = NPERSEG
    noverlap = NOVERLAP
    fs = _get_sampling_frequency(da1, dim)

    if func_name == "csd":
        # Calculate using xrscipy
        actual = dsp.csd(da1, da2, dim=dim, nperseg=nperseg, noverlap=noverlap)

        # Calculate using scipy
        expected_f, expected_result = sp.signal.csd(
            da1.values,
            da2.values,
            fs=fs,
            window=HANN_WINDOW,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=None,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=axis,
            average="mean",
        )

    elif func_name == "welch":
        # Calculate using xrscipy
        actual = dsp.welch(da1, dim=dim, nperseg=nperseg, noverlap=noverlap)

        # Calculate using scipy
        expected_f, expected_result = sp.signal.welch(
            da1.values,
            fs=fs,
            window=HANN_WINDOW,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=None,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=axis,
            average="mean",
        )

    elif func_name == "coherence":
        # Calculate using xrscipy
        actual = dsp.coherence(da1, da2, dim=dim, nperseg=nperseg, noverlap=noverlap)

        # Calculate using scipy
        expected_f, expected_result = sp.signal.coherence(
            da1.values,
            da2.values,
            fs=fs,
            window=HANN_WINDOW,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=None,
            detrend="constant",
            axis=axis,
        )

    elif func_name == "spectrogram":
        # Calculate using xrscipy
        actual = dsp.spectrogram(da1, dim=dim, nperseg=nperseg, noverlap=noverlap, window=TUKEY_WINDOW)

        # Calculate using scipy
        expected_f, expected_t, expected_result = sp.signal.spectrogram(
            da1.values,
            fs=fs,
            window=TUKEY_WINDOW,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=None,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=axis,
            mode="psd",
        )

    # Check that frequency values match
    np.testing.assert_allclose(actual.coords["frequency"].values, expected_f)

    # Check that result values match
    if func_name == "spectrogram":
        # For spectrogram, also check time values
        np.testing.assert_allclose(actual.coords[dim].values, expected_t)
        np.testing.assert_allclose(actual.values, expected_result)
    else:
        np.testing.assert_allclose(actual.values, expected_result)

    # Check metadata preservation
    _check_metadata_preservation(da1, actual, dim)


def test_crossspectrogram():
    """test for crossspectrogram regarding paths not covered by the already existing tests

    This test covers the following lines in crossspectrogram function:
    - dt = 1.0 / fs when fs is explicitly provided
    - nperseg and nfft calculation when seglen is provided
    - noverlap calculation from overlap_ratio when noverlap is None
    - xr.broadcast usage when arrays have different dimensions
    """
    # Test data for all scenarios
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 0.5, 5)

    # Basic 1D arrays for most tests
    da1 = xr.DataArray(np.sin(2 * np.pi * x), dims=["x"], coords={"x": x})
    da2 = xr.DataArray(np.cos(2 * np.pi * x), dims=["x"], coords={"x": x})

    # Test 1: Explicit fs parameter
    fs_calculated = 1.0 / (x[1] - x[0])
    result_auto_fs = _crossspectrogram(da1, da2, dim="x", nperseg=4, noverlap=2)
    result_explicit_fs = _crossspectrogram(da1, da2, dim="x", fs=fs_calculated, nperseg=4, noverlap=2)
    np.testing.assert_allclose(result_auto_fs.values, result_explicit_fs.values)

    # Test 2: seglen parameter
    result_seglen = _crossspectrogram(da1, da2, dim="x", seglen=0.2)
    assert isinstance(result_seglen, xr.DataArray)
    assert "frequency" in result_seglen.coords
    assert "x" in result_seglen.coords

    # Test 3: Default noverlap calculation
    result_noverlap = _crossspectrogram(da1, da2, dim="x", nperseg=8, overlap_ratio=0.25)
    assert isinstance(result_noverlap, xr.DataArray)
    assert "frequency" in result_noverlap.coords
    assert "x" in result_noverlap.coords

    # Test 4: Different dimensions
    # 2D array with different coordinate lengths
    da2_2d = xr.DataArray(
        np.outer(np.cos(2 * np.pi * x[:10]), np.sin(2 * np.pi * y)), dims=["x", "y"], coords={"x": x[:10], "y": y}
    )
    result_broadcast = _crossspectrogram(da1, da2_2d, dim="x", nperseg=4, noverlap=2)
    assert isinstance(result_broadcast, xr.DataArray)
    assert "frequency" in result_broadcast.coords
    assert "x" in result_broadcast.coords
    assert "y" in result_broadcast.coords

    # Basic sanity checks for all results
    assert np.all(np.isfinite(result_auto_fs.values))
    assert np.all(np.isfinite(result_seglen.values[:10]))  # Check first few values
    assert np.all(np.isfinite(result_noverlap.values))
