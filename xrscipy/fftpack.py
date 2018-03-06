from __future__ import absolute_import, division, print_function
from functools import partial

import numpy as np
from scipy import fftpack
import xarray as xr
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

    if np.abs(jitter / mean) > 1e-4:     # heuristic value
        raise ValueError('Coordinate for FFT should be evenly spaced.')

    return mean


def _wrap1d(func, freq_func, y, coord, outdim=None, **kwargs):
    """ Wrap function for fft1d """
    errors.raise_invalid_args(['axis', 'overwrite_x'], kwargs)
    errors.raise_not_sorted(y[coord])

    outdim = outdim if outdim is not None else coord

    # In case of dim is a non-dimensional coordinate.
    x = y[coord]
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


def _inject_docs(func, func_name, description=None):
    doc = DocParser(getattr(fftpack, func_name).__doc__)
    doc.replace_params(
        x='obj : xarray object\n' + doc.parameters['x'][1],
        axis='coord : string\n'
        + doc.parameters['axis'][1].split(';')[0].replace('Axis', 'Coordinate')
        + '.\n    The coordinate must be evenly spaced.\n')
    doc.remove_params('overwrite_x')
    doc.add_params(
        outdim='''outdim : string, optional\n    Name of the output '''
               '''dimension; the default is obj[coord].dims[0].\n''')
    doc.reorder_params('obj', 'coord')

    doc.remove_sections('Notes', 'Examples')

    # update return statement
    returns = doc.returns.copy()
    for key, item in doc.returns.items():
        returns[key] = [it.replace('ndarray', 'xarray object') for it in item]
    doc.returns = returns

    if description is not None:
        doc.insert_description(description)

    doc.insert_see_also(**{
        'scipy.fftpack.' + func_name:
        'scipy.fftpack.' + func_name + ' : Original scipy implementation\n'})

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


fft = partial(_wrap1d, fftpack.fft, fftpack.fftfreq)
_inject_docs(fft, 'fft',
             description='fft(obj, coord, n=None, outdim=None)')

ifft = partial(_wrap1d, fftpack.ifft, fftpack.fftfreq)
_inject_docs(ifft, 'ifft',
             description='ifft(obj, coord, n=None, outdim=None)')

rfft = partial(_wrap1d, fftpack.rfft, fftpack.rfftfreq)
_inject_docs(rfft, 'rfft',
             description='rfft(obj, coord, n=None, outdim=None)')

irfft = partial(_wrap1d, fftpack.irfft, fftpack.rfftfreq)
_inject_docs(irfft, 'irfft',
             description='irfft(obj, coord, n=None, outdim=None)')

dct = partial(_wrap1d, fftpack.dct, fftpack.rfftfreq)
_inject_docs(dct, 'dct',
             description='dct(obj, coord, type=2, n=None, norm=None, '
             'outdim=None)')

dst = partial(_wrap1d, fftpack.dst, fftpack.rfftfreq)
_inject_docs(dst, 'dst',
             description='dst(obj, coord, type=2, n=None, norm=None, '
             'outdim=None)')

idct = partial(_wrap1d, fftpack.idct, fftpack.rfftfreq)
_inject_docs(idct, 'idct',
             description='idct(obj, coord, type=2, n=None, norm=None, '
             'outdim=None)')

idst = partial(_wrap1d, fftpack.idst, fftpack.rfftfreq)
_inject_docs(idst, 'idst',
             description='idst(obj, coord, type=2, n=None, norm=None, '
             'outdim=None)')
