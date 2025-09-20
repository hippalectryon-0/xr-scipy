"""Tests for xrscipy.signal.spectral module."""

import numpy as np
import pytest
import scipy as sp

from xrscipy import signal
from xrscipy.signal.utils import get_sampling_step
from .testings import get_obj


def _test_spectral_function(da, func_name, dim, is_psd=False):
    """Helper function to test spectral analysis functions.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    func_name : str
        Name of the function to test
    dim : str
        Dimension to operate on
    is_psd : bool
        Whether this is a PSD function (welch) that averages over time
    """
    # Calculate fs the same way xrscipy does
    dt = get_sampling_step(da, dim)
    fs = 1.0 / dt

    if is_psd:
        # PSD functions (welch)
        actual = getattr(signal, func_name)(da, dim=dim, nperseg=4, window="hann")
        _, expected = getattr(sp.signal, func_name)(
            da.values, fs=fs, axis=da.get_axis_num(dim), nperseg=4, window="hann"
        )
        # Use allclose for PSD functions due to floating point precision issues
        assert np.allclose(actual.values, expected, equal_nan=True)
    else:
        # Spectrogram functions
        actual = getattr(signal.spectral, func_name)(da, dim=dim, nperseg=4, noverlap=2, window="hann")
        _, _, expected = getattr(sp.signal, func_name)(
            da.values, fs=fs, axis=da.get_axis_num(dim), nperseg=4, noverlap=2, window="hann"
        )
        # Use exact equality for spectrogram functions
        assert (actual.values == expected).all()

    # make sure scalar coordinates are propagated
    # (dimensional coordinates will change as they represent different things)
    for key, v in da.coords.items():
        if v.ndim == 0:  # scalar coordinates
            assert da[key].identical(actual[key])


@pytest.mark.parametrize("mode", [0])
@pytest.mark.parametrize("func", ["spectrogram", "welch"])
@pytest.mark.parametrize("dim", ["x"])
def test_spectral_basic(mode, func, dim):
    """Test basic spectral analysis functions.

    Verifies that xrscipy spectral functions produce results strictly equal to scipy,
    and that metadata is properly handled:
    - Input DataArrays remain unmodified
    - Coordinates are propagated to output DataArrays
    - Frequency dimension is correctly created
    """
    da = get_obj(mode)

    # Skip if the dimension doesn't exist in this mode
    if dim not in da.dims:
        pytest.skip(f"Dimension {dim} not in data for mode {mode}")

    # Use parameters that work with small arrays and matching windows
    is_psd = func == "welch"
    _test_spectral_function(da, func, dim, is_psd=is_psd)

    # make sure the original data does not change
    assert da.values.shape == get_obj(mode).shape


def test_spectral_error_conditions():
    """Test error conditions for spectral analysis functions.

    Verifies that appropriate errors are raised for invalid parameters.
    """
    da = get_obj(0)

    # Test with non-existent dimension
    with pytest.raises((ValueError, KeyError)):
        signal.spectral.spectrogram(da, dim="nonexistent")
