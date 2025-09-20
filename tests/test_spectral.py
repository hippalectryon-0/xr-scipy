"""Tests for xrscipy.signal.spectral module."""

import numpy as np
import pytest
import scipy as sp

import xrscipy.signal as dsp
from .testings import get_obj


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dim", ["x"])
def test_csd(mode, dim):
    """Test cross power spectral density function.

    Verifies that xrscipy.signal.csd produces results strictly equal to scipy.signal.csd,
    and that metadata is properly handled:
    - Input DataArrays remain unmodified (shape preservation)
    - Coordinates are propagated to output DataArrays
    - Frequency dimension is correctly created
    """
    da1 = get_obj(mode)
    da2 = get_obj(mode)

    # For 1D case with mode=0, we need to make sure we have the right dimension
    if mode == 0 and dim not in da1.dims:
        pytest.skip("dimension not available in 1D test object")

    if dim not in da1.dims or dim not in da2.dims:
        pytest.skip("dimension not available in test object")

    # Get axis number for the specified dimension
    axis = da1.get_axis_num(dim)

    # Use appropriate parameters for the signal length
    if mode == 0:
        # For short signals, use smaller nperseg
        actual = dsp.csd(da1, da2, dim=dim, nperseg=4, noverlap=2)

        # Calculate using scipy with matching parameters
        fs = 1.0 / (da1.coords[dim][1] - da1.coords[dim][0]).values  # Calculate sampling frequency
        expected_f, expected_csd = sp.signal.csd(
            da1.values,
            da2.values,
            fs=fs,
            window="hann",
            nperseg=4,
            noverlap=2,
            nfft=None,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=axis,
            average="mean",
        )
    else:
        # For longer signals along the x dimension (which has size 9), use appropriate parameters
        actual = dsp.csd(da1, da2, dim=dim, nperseg=4, noverlap=2)

        # Calculate using scipy with matching parameters
        fs = 1.0 / (da1.coords[dim][1] - da1.coords[dim][0]).values  # Calculate sampling frequency
        expected_f, expected_csd = sp.signal.csd(
            da1.values,
            da2.values,
            fs=fs,
            window="hann",
            nperseg=4,
            noverlap=2,
            nfft=None,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=axis,
            average="mean",
        )

    # Check that frequency values match
    np.testing.assert_allclose(actual.coords["frequency"].values, expected_f)

    # Check that csd values match
    np.testing.assert_allclose(actual.values, expected_csd)

    # Make sure the original data does not change
    assert da1.values.shape == get_obj(mode).shape
    assert da2.values.shape == get_obj(mode).shape

    # Make sure the coordinate (except the transformed one) is propagated
    for key, v in da1.coords.items():
        if dim not in v.dims and key != dim:
            assert da1[key].identical(actual[key])


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dim", ["x"])
def test_welch(mode, dim):
    """Test power spectral density function.

    Verifies that xrscipy.signal.welch produces results strictly equal to scipy.signal.welch,
    and that metadata is properly handled:
    - Input DataArrays remain unmodified (shape preservation)
    - Coordinates are propagated to output DataArrays
    - Frequency dimension is correctly created
    """
    da = get_obj(mode)

    # For 1D case with mode=0, we need to make sure we have the right dimension
    if mode == 0 and dim not in da.dims:
        pytest.skip("dimension not available in 1D test object")

    if dim not in da.dims:
        pytest.skip("dimension not available in test object")

    # Get axis number for the specified dimension
    axis = da.get_axis_num(dim)

    # Use appropriate parameters for the signal length to avoid the noverlap issue
    if mode == 0:
        # For short signals, use smaller nperseg
        actual = dsp.welch(da, dim=dim, nperseg=4, noverlap=2)

        # Calculate using scipy with matching parameters
        fs = 1.0 / (da.coords[dim][1] - da.coords[dim][0]).values  # Calculate sampling frequency
        expected_f, expected_psd = sp.signal.welch(
            da.values,
            fs=fs,
            window="hann",
            nperseg=4,
            noverlap=2,
            nfft=None,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=axis,
            average="mean",
        )
    else:
        # For longer signals along the x dimension (which has size 9), use appropriate parameters
        actual = dsp.welch(da, dim=dim, nperseg=4, noverlap=2)

        # Calculate using scipy with matching parameters
        fs = 1.0 / (da.coords[dim][1] - da.coords[dim][0]).values  # Calculate sampling frequency
        expected_f, expected_psd = sp.signal.welch(
            da.values,
            fs=fs,
            window="hann",
            nperseg=4,
            noverlap=2,
            nfft=None,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=axis,
            average="mean",
        )

    # Check that frequency values match
    np.testing.assert_allclose(actual.coords["frequency"].values, expected_f)

    # Check that psd values match
    np.testing.assert_allclose(actual.values, expected_psd)

    # Make sure the original data does not change
    assert da.values.shape == get_obj(mode).shape

    # Make sure the coordinate (except the transformed one) is propagated
    for key, v in da.coords.items():
        if dim not in v.dims and key != dim:
            assert da[key].identical(actual[key])


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dim", ["x"])
def test_coherence(mode, dim):
    """Test coherence function.

    Verifies that xrscipy.signal.coherence produces results that match scipy.signal.coherence,
    and that metadata is properly handled:
    - Input DataArrays remain unmodified (shape preservation)
    - Coordinates are propagated to output DataArrays
    - Frequency dimension is correctly created
    """
    da1 = get_obj(mode)
    da2 = get_obj(mode)

    # For 1D case with mode=0, we need to make sure we have the right dimension
    if mode == 0 and dim not in da1.dims:
        pytest.skip("dimension not available in 1D test object")

    if dim not in da1.dims or dim not in da2.dims:
        pytest.skip("dimension not available in test object")

    # Get axis number for the specified dimension
    axis = da1.get_axis_num(dim)

    # Use appropriate parameters for the signal length
    if mode == 0:
        # For short signals, use smaller nperseg
        actual = dsp.coherence(da1, da2, dim=dim, nperseg=4, noverlap=2)

        # Calculate using scipy with matching parameters
        fs = 1.0 / (da1.coords[dim][1] - da1.coords[dim][0]).values  # Calculate sampling frequency
        expected_f, expected_coherence = sp.signal.coherence(
            da1.values,
            da2.values,
            fs=fs,
            window="hann",
            nperseg=4,
            noverlap=2,
            nfft=None,
            detrend="constant",
            axis=axis,
        )
    else:
        # For longer signals along the x dimension (which has size 9), use appropriate parameters
        actual = dsp.coherence(da1, da2, dim=dim, nperseg=4, noverlap=2)

        # Calculate using scipy with matching parameters
        fs = 1.0 / (da1.coords[dim][1] - da1.coords[dim][0]).values  # Calculate sampling frequency
        expected_f, expected_coherence = sp.signal.coherence(
            da1.values,
            da2.values,
            fs=fs,
            window="hann",
            nperseg=4,
            noverlap=2,
            nfft=None,
            detrend="constant",
            axis=axis,
        )

    # Check that frequency values match
    np.testing.assert_allclose(actual.coords["frequency"].values, expected_f)

    # Check that coherence values match scipy
    np.testing.assert_allclose(actual.values, expected_coherence)

    # Make sure the original data does not change
    assert da1.values.shape == get_obj(mode).shape
    assert da2.values.shape == get_obj(mode).shape

    # Make sure the coordinate (except the transformed one) is propagated
    for key, v in da1.coords.items():
        if dim not in v.dims and key != dim:
            assert da1[key].identical(actual[key])
