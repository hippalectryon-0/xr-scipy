from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np
import xarray as xr
from numpy import fft as fft_

from . import errors
from . import utils
from .docs import DocParser


def _get_spacing(x):
    if x.ndim != 1:
        raise ValueError('Coordinate for FFT should be one dimensional. '
                         'Axis {} is {}-dimensional.'.format(x.name, x.ndim))
    dx = np.diff(x)
    mean = dx.mean()
    jitter = dx.std()

    if np.abs(jitter / mean) > 1e-4:  # heuristic value
        raise ValueError('Coordinate for FFT should be evenly spaced.')

    return mean


def _wrap1d(func, freq_func, y, coord, **kwargs):
    """ Wrap function for fft1d """
    errors.raise_invalid_args(['axis', 'overwrite_x'], kwargs)
    errors.raise_not_sorted(y[coord])

    # In case of dim is a non-dimensional coordinate.
    x = y[coord]
    dim = x.dims[0]
    output_core_dim = [dim]
    dx = _get_spacing(x)

    kwargs['axis'] = -1

    def apply_func(v):
        # v: xr.Varaible
        result = xr.apply_ufunc(
            func, v, input_core_dims=[[dim]],
            output_core_dims=[output_core_dim], kwargs=kwargs)
        return result.set_dims(v.dims)

    ds = utils.wrap_dataset(apply_func, y, dim, keep_coords='drop')

    # attach frequency coordinate
    size = kwargs.pop('n', None)
    if size is None:
        size = len(y[dim])
    freq = freq_func(size, dx)
    ds[coord] = (dim,), freq
    return ds


def _wrapnd(func, freq_func, y, *coords, **kwargs):
    """ Wrap function for fftnd """
    errors.raise_invalid_args(['axes', 'overwrite_x'], kwargs)
    shape = kwargs.pop('s', None)

    if shape is not None and not isinstance(shape, dict):
        raise TypeError('shape should be a dict mapping from coord name to '
                        'size. Given {}.'.format(shape))

    for c in coords:
        errors.raise_not_sorted(y[c])

    # In case of dim is a non-dimensional coordinate.
    xs = [y[c] for c in coords]
    dims = [x.dims[0] for x in xs]
    shape = {d: len(y[d]) if shape is None or c not in shape else shape[c]
             for d, c in zip(dims, coords)}
    dxs = [_get_spacing(x) for x in xs]

    def apply_func(v):
        # v: xr.Varaible
        kwargs_tmp = kwargs.copy()
        kwargs_tmp.pop('s', None)
        input_core_dims = [d for d in dims if d in v.dims]
        kwargs_tmp['axes'] = -np.arange(len(input_core_dims))[::-1] - 1
        if shape is not None:
            kwargs_tmp['s'] = [shape[d] for d in input_core_dims]
        result = xr.apply_ufunc(
            func, v, input_core_dims=[input_core_dims],
            output_core_dims=[input_core_dims], kwargs=kwargs_tmp)
        return result.set_dims(v.dims)

    ds = utils.wrap_dataset(apply_func, y, *dims, keep_coords='drop')

    # attach frequency coordinate
    for c, d, dx in zip(coords, dims, dxs):
        size = kwargs.pop(d, None)
        if size is None:
            size = len(y[d])
        freq = freq_func(size, dx)
        ds[c] = (d,), freq
    return ds


def _inject_docs(func, func_name, description=None, nd=False):
    try:
        doc = DocParser(getattr(fft_, func_name).__doc__)
    except errors.NoDocstringError:
        return

    if not nd:
        doc.replace_params(
            a='a : xarray object\n    The data tp transform.',
            axis='coord : string\n' +
                 '    The axis along which the transform is applied. ' +
                 '.\n    The coordinate must be evenly spaced.\n')
    else:
        doc.replace_params(
            a='a : xarray object\n' +
              '    Object which the transform is applied.\n',
            axes='coords : string\n' +
                 '    Coordinates along which the transform is applied.\n'
                 '    The coordinate must be evenly spaced.\n',
            s='s: mapping from coords to size, optional\n'
              '    The shape of the result.')

    doc.reorder_params('a', 'coord')
    doc.remove_sections('Notes', 'Examples')

    # update return statement
    returns = doc.returns.copy()
    for key, item in doc.returns.items():
        returns[key] = [it.replace('ndarray', 'xarray object')
                        .replace('axes', 'coords').replace('axis', 'coord')
                        for it in item]
    doc.returns = returns

    if description is not None:
        doc.insert_description(description)

    doc.insert_see_also(**{
        'numpy.fft.' + func_name:
            'numpy.fft.' + func_name + ' : Original numpy implementation\n'})

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


fft = partial(_wrap1d, fft_.fft, fft_.fftfreq)
_inject_docs(fft, 'fft', description='fft(a, coord, n=None, norm=None)')

ifft = partial(_wrap1d, fft_.ifft, fft_.fftfreq)
_inject_docs(ifft, 'ifft',
             description='ifft(a, coord, n=None, norm=None)')

rfft = partial(_wrap1d, fft_.rfft, fft_.rfftfreq)
_inject_docs(rfft, 'rfft',
             description='rfft(a, coord, n=None, norm=None)')

irfft = partial(_wrap1d, fft_.irfft, fft_.rfftfreq)
_inject_docs(irfft, 'irfft',
             description='irfft(a, coord, n=None, norm=None)')

fftn = partial(_wrapnd, fft_.fftn, fft_.fftfreq)
_inject_docs(fftn, 'fftn', nd=True,
             description='fftn(a, *coords, shape=None, norm=None)')

ifftn = partial(_wrapnd, fft_.ifftn, fft_.fftfreq)
_inject_docs(ifftn, 'ifftn', nd=True,
             description='ifftn(a, *coords, shape=None, norm=None)')

rfftn = partial(_wrapnd, fft_.rfftn, fft_.rfftfreq)
_inject_docs(rfftn, 'rfftn', nd=True,
             description='rfftn(a, *coords, shape=None, norm=None)')

irfftn = partial(_wrapnd, fft_.irfftn, fft_.rfftfreq)
_inject_docs(irfftn, 'irfftn', nd=True,
             description='irfftn(a, *coords, shape=None, norm=None)')

hfft = partial(_wrap1d, fft_.hfft, fft_.rfftfreq)
_inject_docs(hfft, 'hfft', description='hfft(a, coord, n=None, norm=None)')

ihfft = partial(_wrap1d, fft_.ihfft, fft_.rfftfreq)
_inject_docs(ihfft, 'ihfft',
             description='ihfft(a, coord, n=None, norm=None)')
