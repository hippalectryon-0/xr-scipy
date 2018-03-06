from __future__ import absolute_import, division, print_function

from functools import partial
from textwrap import dedent
from scipy import integrate

import xarray as xr
from . import errors
from . import utils
from .docs import DocParser


def _wrap(func, reduces, y, coord, **kwargs):
    """
    Wrap scipy functions for xarray data objects
    """
    errors.raise_invalid_args(['x', 'dx', 'axis'], kwargs)
    errors.raise_not_sorted(y[coord])

    # In case of dim is a non-dimensional coordinate.
    x = y[coord]
    dim = x.dims[0]
    output_core_dim = [] if reduces else [dim]

    kwargs['x'] = x
    kwargs['axis'] = -1

    def apply_func(v):
        # v: xr.Varaible
        result = xr.apply_ufunc(
            func, v, input_core_dims=[[dim]],
            output_core_dims=[output_core_dim], kwargs=kwargs)
        if not reduces:
            return result.transpose(*y.dims)
        else:
            return result

    return utils.wrap_dataset(apply_func, y, dim)


def inject_docs(func, func_name, description=None):
    doc = DocParser(getattr(integrate, func_name).__doc__)

    doc.replace_params(
        y='obj : xarray object\n' + doc.parameters['y'][1],
        axis='coord : string\n    The coordinate along which to integrate.\n')
    doc.remove_params('dx', 'x')

    doc.remove_sections('Notes', 'Examples')

    # update return statement
    returns = doc.returns.copy()
    for key, item in doc.returns.items():
        returns[key] = [it.replace('ndarray', 'xarray object').replace(
            'axis', 'coordiante'
        ) for it in item]
    doc.returns = returns

    if description is not None:
        doc.insert_description(description)

    doc.description = [item.replace('axis', 'coordinate') for item in
                       doc.description]

    doc.insert_see_also(**{
        'scipy.integrate.' + func_name:
        'scipy.integrate.' + func_name + ' : Original scipy implementation\n'})

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


trapz = partial(_wrap, integrate.trapz, True)
inject_docs(trapz, 'trapz', description='trapz(obj, coord):')

simps = partial(_wrap, integrate.simps, True)
inject_docs(simps, 'simps', description='simps(obj, coord, even=\'avg\'):')

romb = partial(_wrap, integrate.romb, True)
inject_docs(romb, 'romb', description='romb(obj, coord, show=False):')

cumtrapz = partial(_wrap, integrate.cumtrapz, False, initial=0)
inject_docs(cumtrapz, 'cumtrapz', description='cumtrapz(obj, coord):')
