from __future__ import absolute_import, division, print_function

from textwrap import dedent
import scipy as sp
import pytest

from xrscipy import fftpack
from xrscipy.docs import DocParser
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


def test_doc():
    parser = DocParser(fftpack.fft.__doc__)

    not_included_keys = ['x', 'axis', 'overwrite_x']
    for k in not_included_keys:
        assert k not in parser.parameters.keys()

    actual = dedent(fftpack.fft.__doc__)
    expected = '''fft(obj, coord, n=None, outdim=None):

Return discrete Fourier transform of real or complex sequence.

The returned complex array contains ``y(0), y(1),..., y(n-1)`` where

``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``.

Parameters
----------
obj : xarray object
    Array to Fourier transform.
coord : string
    Coordinate along which the fft's are computed.
    The coordinate must be evenly spaced.
n : int, optional
    Length of the Fourier transform.  If ``n < x.shape[axis]``, `x` is
    truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
    default results in ``n = x.shape[axis]``.
outdim : string, optional
    Name of the output dimension; the default is obj[coord].dims[0].

Returns
--------
z : complex xarray object
    with the elements::

        [y(0),y(1),..,y(n/2),y(1-n/2),...,y(-1)]        if n is even
        [y(0),y(1),..,y((n-1)/2),y(-(n-1)/2),...,y(-1)]  if n is odd

    where::

        y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1


See Also
--------
scipy.fftpack.fft : Original scipy implementation
ifft : Inverse FFT
rfft : FFT of a real sequence
'''
    assert actual == expected
