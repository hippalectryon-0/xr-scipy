from distutils.version import LooseVersion

import numpy as np
import xarray as xr

from . import errors, utils


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
    if LooseVersion(np.__version__) < LooseVersion("1.13"):
        raise ImportError("Gradient requires numpy >= 1.13")

    errors.raise_not_sorted(f[coord])

    x = f[coord]
    dim = x.dims[0]

    def gradient(f):
        return np.gradient(f, x.values, axis=-1, edge_order=edge_order)

    def func(v):
        # noinspection PyProtectedMember
        result = xr.apply_ufunc(gradient, v, input_core_dims=[[dim]], output_core_dims=[[utils._TEMP_DIM]])
        # noinspection PyProtectedMember
        dims = [d if d != dim else utils._TEMP_DIM for d in v.dims]
        result = result.transpose(*dims)
        result.dims = v.dims
        return result

    return utils.wrap_dataset(func, f, dim, keep_coords="keep")
