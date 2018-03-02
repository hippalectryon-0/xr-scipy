from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from scipy import fftpack
import xarray as xr
from . import errors
from . import utils


def _get_spacing(x):
    if x.ndim != 1:
        raise ValueError('Coordinate for FFT should be one dimensional. '
                         'Axis {} is {}-dimensional.'.format(x.name, x.ndim))
    dx = np.diff(x)
    mean = dx.mean()
    jitter = dx.std()

    if np.abs(jitter / mean) > 1e-4:     # heuristic value
        raise ValueError('Coordinate for FFT should be evenly spaced.')

    return mean


def _wrap1d(func, freq_func, y, dim, outdim, **kwargs):
    """ Wrap function for fft1d """
    errors.raise_invalid_args(['axis', 'overwrite_x'], kwargs)
    errors.raise_not_sorted(y[dim])

    # In case of dim is a non-dimensional coordinate.
    x = y[dim]
    dim = x.dims[0]
    output_core_dim = [outdim]
    dx = _get_spacing(x)

    kwargs['axis'] = -1
    kwargs['overwrite_x'] = False

    def apply_func(v):
        # v: xr.Varaible
        result = xr.apply_ufunc(
            func, v, input_core_dims=[[dim]],
            output_core_dims=[output_core_dim], kwargs=kwargs)
        new_dims = [d if d != dim else outdim for d in v.dims]
        return result.set_dims(new_dims)

    ds = utils.wrap_dataset(apply_func, y, dim)

    # attach frequency coordinate
    freq = freq_func(len(ds[outdim]), dx)
    ds[outdim] = (outdim, ), freq
    return ds


fft = partial(_wrap1d, fftpack.fft, fftpack.fftfreq)
ifft = partial(_wrap1d, fftpack.ifft, fftpack.fftfreq)
rfft = partial(_wrap1d, fftpack.rfft, fftpack.rfftfreq)
irfft = partial(_wrap1d, fftpack.irfft, fftpack.rfftfreq)
dct = partial(_wrap1d, fftpack.dct, fftpack.rfftfreq)
dst = partial(_wrap1d, fftpack.dst, fftpack.rfftfreq)
idct = partial(_wrap1d, fftpack.idct, fftpack.rfftfreq)
idst = partial(_wrap1d, fftpack.idst, fftpack.rfftfreq)
