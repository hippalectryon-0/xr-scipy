from __future__ import absolute_import, division, print_function

import scipy as sp
import pytest

from xrscipy import fftpack
from .testings import get_obj


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('func', ['fft', 'rfft'])
@pytest.mark.parametrize('dim', ['x', 'time'])
@pytest.mark.parametrize('n', [None, 14])
def test_interpolate1d(mode, func, dim, n):
    da = get_obj(mode)

    axis = da.get_axis_num(da[dim].dims[0])
    actual = getattr(fftpack, func)(da, dim, 'freq', n=n)
    expected = getattr(sp.fftpack, func)(da, n, axis=axis)

    assert (actual.values == expected).all()

    # make sure the original data does not change
    da.values.ndim == get_obj(mode).ndim

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if 'x' not in v.dims:
            assert da[key].identical(actual[key])
