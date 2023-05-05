import numpy as np
import pytest
import scipy
import scipy as sp

from xrscipy import fft, fftpack

from .testings import get_obj

# TODO utest iftt(fft) to make sure we haven't messed up the freqs too badly


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("module", ["fft"])  # TODO "fftpack"
@pytest.mark.parametrize("func", ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"])
@pytest.mark.parametrize("dim", ["x", "time"])
@pytest.mark.parametrize("n", [None, 14])
def test_fft1d(mode, module, func, dim, n):
    da = get_obj(mode)

    axis = da.get_axis_num(da[dim].dims[0])
    if module == "fftpack":
        actual = getattr(fftpack, func)(da, dim, n=n)
        expected = getattr(sp.fftpack, func)(da.values, n, axis=axis)
    else:
        actual = getattr(fft, func)(da, dim, n=n).transpose(*da.dims)
        expected: np.ndarray = getattr(scipy.fft, func)(da.values, n, axis=axis)

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
@pytest.mark.parametrize("module", ["fft"])  # TODO "fftpack"
@pytest.mark.parametrize("func", ["fftn", "ifftn", "rfftn", "irfftn"])
@pytest.mark.parametrize("coords", [["x"], ["time", "y"]])
@pytest.mark.parametrize("shape", [None, {"time": 14}])
def test_fftnd(mode, module, func, coords, shape):
    da = get_obj(mode)

    if shape is not None and coords == ["x"]:
        pytest.skip("invalid shape")

    axes = [da.get_axis_num(da[c].dims[0]) for c in coords]
    shape_sp = [
        shape[c] if shape is not None and c in shape else da.values.shape[axes[i]] for i, c in enumerate(coords)
    ]

    if module == "fftpack":
        actual = getattr(fftpack, func)(da, *coords, shape=shape)
        expected = getattr(sp.fftpack, func)(da.values, axes=axes, shape=shape_sp)
    else:
        actual = getattr(fft, func)(da, *coords, s=shape).transpose(*da.dims)
        expected: np.ndarray = getattr(scipy.fft, func)(da.values, axes=axes, s=shape_sp)

    assert (actual.values == expected).all()

    # make sure the original data does not change
    assert da.values.shape == get_obj(mode).shape

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if "x" not in v.dims:
            assert da[key].identical(actual[key])
