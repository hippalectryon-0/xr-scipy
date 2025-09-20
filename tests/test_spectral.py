"""Tests for xrscipy.signal.spectral module."""

import numpy as np
import pytest
import scipy as sp

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
@pytest.mark.parametrize("func_name", ["csd", "welch", "coherence", "spectrogram"])
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

    if func_name != "spectrogram" and dim not in da2.dims:
        pytest.skip("dimension not available in test object")

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
    else:
        raise ValueError

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
