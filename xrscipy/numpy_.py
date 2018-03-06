import numpy as np
import xarray as xr
from . import errors


def gradient(f, coord, edge_order=1):
    """
    Return the gradient of an xarray object.

    The gradient is computed using second order accurate central differences in
    the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries. The returned gradient
    hence has the same shape as the input array.

    Parameters
    ----------
    f: xarray object
        An xarray object containing samples of a scalar function.
    coord: str
        Coordinate along which the gradient is computed.
    edge_order: {1, 2}, optional
        Gradient is calculated using N-th order accurate differences at the
        boundaries. Default: 1.

    Returns
    -------
    gradient: xarray object
    """
    errors.raise_not_sorted(f[coord])

    x = f[coord]
    dim = x.dims[0]

    if isinstance(f, xr.DataArray):
        # TODO use apply_ufunc
        result = np.gradient(f.values, x.values, axis=f.get_axis_num(dim),
                             edge_order=edge_order)
        return xr.DataArray(result, dims=f.dims, coords=f.coords)
    else:
        raise TypeError('Invalid data type {} is given.'.format(type(f)))
