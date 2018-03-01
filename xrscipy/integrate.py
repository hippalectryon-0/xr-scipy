from __future__ import absolute_import, division, print_function
from functools import partial
import scipy.integrate as integrate
import xarray as xr
from . import errors


def _wrap(func, reduces, y, dim, **kwargs):
    """
    Wrap scipy functions for xarray data objects
    """
    errors.raise_invalid_args(['x', 'dx', 'axis'], kwargs)
    errors.raise_not_sorted(y[dim])

    output_core_dim = [] if reduces else [dim]

    if isinstance(y, xr.DataArray):
        result = xr.apply_ufunc(
            func, y, input_core_dims=[[dim]],
            output_core_dims=[output_core_dim],
            kwargs={'x': y[dim], 'axis': -1})
        if not reduces:
            return result.transpose(*y.dims)
        else:
            return result

    elif isinstance(y, xr.Dataset):
        y = y.copy()
        for key in y.data_vars.keys():
            if dim in y[key].dims:
                y[key] = _wrap(func, reduces, y[key], dim, **kwargs)
        return y

    else:
        raise TypeError('Invalid data type {} is given.'.format(type(y)))


trapz = partial(_wrap, integrate.trapz, True)
sims = partial(_wrap, integrate.simps, True)
romb = partial(_wrap, integrate.romb, True)
cumtrapz = partial(_wrap, integrate.cumtrapz, False)
