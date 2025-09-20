"""Tests for xrscipy.fft and xrscipy.fftpack modules."""

import numpy as np
import pytest
import scipy as sp
import xarray as xr

from xrscipy import fft, fftpack
from .testings import get_obj


# TODO utest iftt(fft) to make sure we haven't messed up the freqs too badly


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("module", ["fft", "fftpack"])
@pytest.mark.parametrize("func", ["fft", "ifft", "rfft", "irfft", "dct", "dst", "idct", "idst"])
@pytest.mark.parametrize("dim", ["x", "time"])
@pytest.mark.parametrize("n", [None, 14])
def test_fft1d(mode, module, func, dim, n):
    """Test 1D FFT functions (fft, ifft, rfft, irfft, dct, dst, idct, idst).

    Verifies that xrscipy FFT functions produce results strictly equal to scipy,
    and that metadata is properly handled:
    - Input DataArrays remain unmodified (shape preservation)
    - Coordinates are propagated to output DataArrays
    - Indexing works correctly on output DataArrays
    """
    da = get_obj(mode)
    if module == "fft" and func in ["dct", "dst", "idct", "idst"]:
        pytest.skip("not implemented")

    axis = da.get_axis_num(da[dim].dims[0])
    fftlib, sp_fftlib = (fftpack, sp.fftpack) if module == "fftpack" else (fft, sp.fft)
    actual = getattr(fftlib, func)(da, dim, n=n).transpose(*da.dims)
    expected: xr.DataArray = getattr(sp_fftlib, func)(da.values, n=n, axis=axis)

    assert (actual.values == expected).all()

    # make sure the original data does not change
    assert da.values.shape == get_obj(mode).shape

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if "x" not in v.dims:
            assert da[key].identical(actual[key])

    # make sure it can be indexed
    d = da[dim].dims[0]
    assert len(da[dim]) == len(da[d])


@pytest.mark.parametrize("mode", [1])
@pytest.mark.parametrize("module", ["fftpack", "fft"])
@pytest.mark.parametrize("func", ["fftn", "ifftn", "rfftn", "irfftn"])
@pytest.mark.parametrize("coords", [["x"], ["time", "y"]])
@pytest.mark.parametrize("shape", [None, {"time": 14}])
def test_fftnd(mode, module, func, coords, shape):
    """Test multidimensional FFT functions (fftn, ifftn, rfftn, irfftn).

    Verifies that xrscipy ND FFT functions produce results strictly equal to scipy,
    and that metadata is properly handled:
    - Input DataArrays remain unmodified (shape preservation)
    - Coordinates are propagated to output DataArrays
    - Multiple axes are correctly handled
    - Shape parameter is properly applied
    """
    da = get_obj(mode)

    if module == "fftpack" and func in ["rfftn", "irfftn"]:
        pytest.skip("not implemented")

    if shape is not None and coords == ["x"]:
        pytest.skip("invalid shape")

    axes = [da.get_axis_num(da[c].dims[0]) for c in coords]
    shape_sp = [
        shape[c] if shape is not None and c in shape else da.values.shape[axes[i]] for i, c in enumerate(coords)
    ]

    fftlib, sp_fftlib = (fftpack, sp.fftpack) if module == "fftpack" else (fft, sp.fft)
    kwargs = {"shape" if module == "fftpack" else "s": shape}
    actual = getattr(fftlib, func)(da, *coords, **kwargs).transpose(*da.dims)
    kwargs_sp = {"shape" if module == "fftpack" else "s": shape_sp}
    expected: np.ndarray = getattr(sp_fftlib, func)(da.values, axes=axes, **kwargs_sp)

    assert (actual.values == expected).all()

    # make sure the original data does not change
    assert da.values.shape == get_obj(mode).shape

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if "x" not in v.dims:
            assert da[key].identical(actual[key])


@pytest.mark.parametrize("func", ["fftn", "ifftn", "rfftn", "irfftn"])
def test_fftnd_invalid_shape_type(func):
    """Test that ND FFT functions raise TypeError when shape/s parameter is not a dict."""
    da = get_obj(1)

    # TypeError when sizes is not a dict for ND functions
    with pytest.raises(TypeError, match="s should be a dict mapping from coord name to size"):
        getattr(fft, func)(da, "x", "y", s=[10, 20])  # Should be a dict, not a list
