from __future__ import absolute_import, division, print_function
from functools import partial
from collections import OrderedDict
from scipy import interpolate
import numpy as np
import xarray as xr
from . import errors


_THIS_ARRAY = xr.core.utils.ReprObject('<this-array>')


def _get_safename(base):
    pass


class _VaribaleInterp(object):
    """ Base class for _Variable1dInterp and VariableNdInterp """


class _VariableIdentity(_VaribaleInterp):
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, xi):
        return xr.Variable(self.dims, xi)


class _Variable1dInterp(_VaribaleInterp):
    def __init__(self, interp_cls, variable, dim, x, **kwargs):
        """ Interp object for xr.Variable

        Parameters
        ----------
        interp_cls: scipy's interpolate class
        variable: xr.Variable
            Variable to be interpolated
        dim: dimension to which interpolate variable along
        x: coordinate of dim
        kwargs:
            kwargs for interp_cls
        """
        self.interp_obj = interp_cls(
            x, variable.data, axis=variable.get_axis_num(dim), **kwargs)
        self.dims = variable.dims
        self.interp_dim = dim

    def __call__(self, x):
        value = self.interp_obj(x)
        return xr.Variable(self.dims, value)


class DataArrayInterp(object):
    def __init__(self, variable, coords, name=None):
        self.variable = variable
        self._coords = coords
        self.name = name

    @property
    def dims(self):
        return self.variable.dims

    def __getitem__(self, key):
        variable = self._coords[key]
        if variable.dims == self.dims:
            coords = self._coords
        else:
            allowed_dims = set(variable.dims)
            coords = OrderedDict((k, v) for k, v in self._coords.items()
                                 if set(v.dims) <= allowed_dims)

        return type(self)(variable, coords, name=self.name)

    def _to_temp_dataset(self):
        variables = OrderedDict()
        variables[_THIS_ARRAY] = self.variable
        variables.update(self._coords)
        return DatasetInterp(variables, list(self._coords.keys()))

    def __call__(self, xi):
        dataset = self._to_temp_dataset()(xi)
        variable = dataset._variables.pop(_THIS_ARRAY)
        coords = dataset._variables
        return xr.DataArray(variable, dims=variable.dims, coords=coords,
                            name=self.name)


class DatasetInterp(object):
    def __init__(self, variables, coords):
        """
        variables: mapping from names to _VaribaleInterp
        coordnames: names of coordinates
        """
        self._variables = variables
        self._coords = coords

    @property
    def coords(self):
        coords = OrderedDict()
        for k in self._coords:
            coords[k] = self[k]
        # TODO maybe we need DatasetCoordinate class?
        return coords

    def __getitem__(self, key):
        variable = self._variables[key]
        coords = OrderedDict()
        needed_dims = set(variable.dims)
        for k in self._coords:
            if set(self._variables[k].dims) <= needed_dims:
                coords[k] = self._variables[k]

        return DataArrayInterp(variable, coords, name=key)

    def __call__(self, xi):
        """ Get interpolated xarray object at new coordinate xi """
        # TODO consider the dimension of new_x
        variables = OrderedDict()
        coords = OrderedDict()
        for k, v in self._variables.items():
            v = v(xi) if isinstance(v, _VaribaleInterp) else v.copy()
            if k in self._coords:
                coords[k] = v
            else:
                variables[k] = v
        return xr.Dataset(variables, coords=coords)


def _wrap_interp1d(interp_cls, obj, coord, **kwargs):
    # TODO consider dask array
    errors.raise_invalid_args(['x', 'axis'], kwargs)
    errors.raise_not_1d(obj[coord])

    x = obj[coord]
    dim = x.dims[0]

    if isinstance(obj, xr.DataArray):
        variable = _Variable1dInterp(interp_cls, obj.variable, dim, x,
                                     **kwargs)
        coords = OrderedDict()
        for k, v in obj.coords.items():
            if dim in v.dims and k != coord:
                coords[k] = _Variable1dInterp(interp_cls, v, dim, x, **kwargs)
            elif k == coord:
                coords[k] = _VariableIdentity([dim])
            else:
                coords[k] = v
        return DataArrayInterp(variable, coords, obj.name)

    if isinstance(obj, xr.Dataset):
        variables = OrderedDict()
        for k, v in obj.variables.items():
            if dim in v.dims and k != coord:
                variables[k] = _Variable1dInterp(interp_cls, v, dim, x,
                                                 **kwargs)
            elif k == coord:
                variables[k] = _VariableIdentity([dim])
            else:
                variables[k] = v
        return DatasetInterp(variables, obj.coords)


interp1d = partial(_wrap_interp1d, interpolate.interp1d)
interp1d.__doc__ = """
    interp1d(y, coord, kind='linear', bounds_error=None, fill_value=nan,
             assume_sorted=False):

    Interpolate a 1-D function.

    This class returns a function whose call method uses interpolation to
    find the value of new points.

    Note that calling interp1d with NaNs present in input values results in
    undefined behaviour.
    """

PchipInterpolator = partial(_wrap_interp1d, interpolate.PchipInterpolator)
PchipInterpolator.__doc__ = """
    PchipInterpolator(y, coord, extrapolate):

    PCHIP 1-d monotonic cubic interpolation.
    """


Akima1DInterpolator = partial(_wrap_interp1d, interpolate.Akima1DInterpolator)
Akima1DInterpolator.__doc__ = """
    Akima1DInterpolator(y, coord):

    Akima interpolator.
    """


CubicSpline = partial(_wrap_interp1d, interpolate.CubicSpline)
CubicSpline.__doc__ = """
    CubicSpline(y, coord, bc_type='not-a-knot', extrapolate=None):

    Cubic spline data interpolator
    """


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
