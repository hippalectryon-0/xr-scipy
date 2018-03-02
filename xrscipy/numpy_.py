import numpy as np
import xarray as xr
from . import errors


def gradient(f, dim, edge_order=1):
    errors.raise_not_sorted(f[dim])

    output_core_dim = [dim]

    if isinstance(f, xr.DataArray):
        result = xr.apply_ufunc(
                    np.gradient, f, input_core_dims=[[dim]],
                    output_core_dims=[output_core_dim],
                    kwargs={'edge_order': edge_order, 'axis': -1})
        return result.transpose(*f.dims)

    else:
        raise TypeError('Invalid data type {} is given.'.format(type(f)))
