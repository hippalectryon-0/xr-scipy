import numpy as np
import xarray as xr
from . import errors


def gradient(f, dim, edge_order=1):
    errors.raise_not_sorted(f[dim])

    x = f[dim]
    dim = x.dims[0]
    output_core_dim = [dim]

    if isinstance(f, xr.DataArray):
        # TODO use apply_ufunc
        result = np.gradient(f.values, x.values, axis=f.get_axis_num(dim),
                             edge_order=edge_order)
        return xr.DataArray(result, dims=f.dims, coords=f.coords)
    else:
        raise TypeError('Invalid data type {} is given.'.format(type(f)))
