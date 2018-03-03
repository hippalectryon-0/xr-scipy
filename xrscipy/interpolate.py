from __future__ import absolute_import, division, print_function
from functools import partial
from collections import OrderedDict
from scipy import interpolate
import numpy as np
import xarray as xr
from . import errors
from . import utils


class _Interp1d(object):
    _interp_cls = None

    def __init__(self, obj, dim, **kwargs):
        # TODO consider dask array
        errors.raise_invalid_args(['x', 'axis'], kwargs)
        errors.raise_not_1d(obj[dim])

        x = obj[dim]
        dim = x.dims[0]

        if isinstance(obj, xr.DataArray):
            self._dataarray_name = obj.name
            self._obj = obj._to_temp_dataset()
        elif isinstance(obj, xr.Dataset):
            self._obj = obj
        else:
            raise TypeError('Invalid object {} is passed.'.format(type(obj)))

        self._interp_objs = OrderedDict()
        for k, v in self._obj.variables.items():
            if dim in v.dims:
                self._interp_objs[k] = self._interp_cls(
                    x, v.data, axis=v.get_axis_num(dim), **kwargs)

    def __call__(self, new_x):
        # TODO consider the dimension of new_x
        ds = xr.Dataset({})
        for k, v in self._obj.variables.items():
            if k in self._interp_objs:
                new_v = self._interp_objs[k](new_x)
                ds[k] = xr.Variable(v.dims, new_v)
            else:
                ds[k] = v
        ds = ds.set_coords(self._obj.coords)

        if hasattr(self, '_dataarray_name'):
            da = ds[list(ds.data_vars.keys())[0]]
            da.name = self._dataarray_name
            return da
        return ds


class interp1d(_Interp1d):
    """
    interp1d(y, dim, kind='linear', bounds_error=None, fill_value=nan,
             assume_sorted=False):

    Interpolate a 1-D function.

    This class returns a function whose call method uses interpolation to
    find the value of new points.

    Note that calling interp1d with NaNs present in input values results in
    undefined behaviour.
    """
    _interp_cls = interpolate.interp1d


class PchipInterpolator(_Interp1d):
    """
    PchipInterpolator(y, dim, extrapolate):

    PCHIP 1-d monotonic cubic interpolation.
    """
    _interp_cls = interpolate.PchipInterpolator


class Akima1DInterpolator(_Interp1d):
    """
    Akima1DInterpolator(y, dim):

    Akima interpolator.
    """
    _interp_cls = interpolate.Akima1DInterpolator


class CubicSpline(_Interp1d):
    """
    CubicSpline(y, dim, bc_type='not-a-knot', extrapolate=None):

    Cubic spline data interpolator
    """
    _interp_cls = interpolate.CubicSpline


def _wrap_griddata(func, obj, coords, new_coords, **kwargs):
    """
    Wrapper for griddata.
    coords: sequence of strings.
    new_coords: the same length of xr.DataArrays.
    """
    assert isinstance(obj, xr.DataArray)

    dims = set()
    for c in coords:
        dims = dims.union(set(obj[c].dims))
    obj = obj.stack(_points=list(dims)).reset_index('_points')  # broadcast
    # Original coordinate. Sized [N, D], where N is the sample size,
    # D is number of dimension
    points = np.stack([obj[c].values for c in coords], axis=-1)
    assert points.ndim == 2
    obj = obj.drop(coords)

    # new coordinates
    # TODO support numpy arrays
    assert all(isinstance(c, xr.DataArray) for c in new_coords)

    new_dims = [c.name if c.name is not None else c_old for c, c_old
                in zip(new_coords, coords)]
    dest_ds = xr.Dataset({}, coords={d: c for d, c in
                                     zip(new_dims, new_coords)})

    dest = dest_ds.stack(_points2=list(dest_ds.dims))
    dest_arrays = np.stack([dest[d] for d in new_dims], axis=-1)

    target_func = func
    if len(coords) == 1:
        def func_sqeeze(points, values, xi, **kwargs):
            # the 1 dimensional interpolation gives 2-dimensional result.
            res = func(points, values, xi, **kwargs)
            return np.squeeze(res, axis=-1)

        target_func = func_sqeeze

    if obj.ndim > 1:
        target_func_copy = target_func

        def func_vectorized(points, values, xi, **kwargs):
            return target_func_copy(
                np.array(points), np.array(values), np.array(xi), **kwargs)

        target_func = np.vectorize(
            func_vectorized, signature='(m,d),(m),(n,d)->(n)')

    result = xr.apply_ufunc(target_func, points, obj, dest_arrays,
                            input_core_dims=[[], ['_points'], []],
                            output_core_dims=[['_points2']])
    # append new coordinates
    result.coords.update(dest.coords)
    if len(coords) > 1:
        result = result.set_index('_points2').unstack('_points2')
        result.coords.update(dest_ds.coords)
    else:
        del result['_points2']
        result = result.rename({'_points2': new_coords[0].dims[0]})

    # drop coordinate that is not coordinate in new_coords
    drop_coords = [c for c in dest.reset_index('_points2').coords
                   if c not in new_dims and c in result.coords]
    for c in drop_coords:
        del result[c]
    return result


griddata = partial(_wrap_griddata, interpolate.griddata)
