from __future__ import absolute_import, division, print_function
from functools import partial
from collections import OrderedDict
from scipy import interpolate
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
