"""Tests for xrscipy.signal.spectral module."""

import numpy as np
import pytest
import scipy as sp
import xarray as xr

import xrscipy.signal as dsp
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


def test_crossspectrogram_with_explicit_fs():
    """Test crossspectrogram function in some cases not tested by the other exposed functions"""
    # Create simple test data with known sampling
    x = np.linspace(0, 1, 10)
    da1 = xr.DataArray(np.sin(2 * np.pi * x), dims=["x"], coords={"x": x})
    da2 = xr.DataArray(np.cos(2 * np.pi * x), dims=["x"], coords={"x": x})

    # Calculate fs from the coordinate spacing
    fs_calculated = 1.0 / (x[1] - x[0])

    # Test with calculated fs (fs=None, let it calculate from coordinates)
    result_auto_fs = dsp.extra.crossspectrogram(da1, da2, dim="x", nperseg=4, noverlap=2)

    # Test with explicit fs parameter (this exercises line 289: dt = 1.0 / fs)
    result_explicit_fs = dsp.extra.crossspectrogram(da1, da2, dim="x", fs=fs_calculated, nperseg=4, noverlap=2)

    # Results should be identical
    np.testing.assert_allclose(result_auto_fs.values, result_explicit_fs.values)
    np.testing.assert_allclose(result_auto_fs.coords["frequency"].values, result_explicit_fs.coords["frequency"].values)
    np.testing.assert_allclose(result_auto_fs.coords["x"].values, result_explicit_fs.coords["x"].values)

    # Basic sanity checks
    assert np.all(np.isfinite(result_explicit_fs.values))
