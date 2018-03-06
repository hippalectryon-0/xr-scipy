from __future__ import absolute_import, division, print_function
from functools import partial
from collections import OrderedDict
from scipy import interpolate
import numpy as np
import xarray as xr
from . import errors
from .docs import DocParser


_THIS_ARRAY = xr.core.utils.ReprObject('<this-array>')
_SAMPLE_DIM = "__sample_dim__"
_DIMENSION_DIM = "__dimension_dim__"


def _get_safename(base):
    pass


class _VaribaleInterp(object):
    """ Base class for _Variable1dInterp and VariableNdInterp """


class _VariableIdentity(_VaribaleInterp):
    def __init__(self, dims, coord_num=0):
        self.dims = dims
        self.coord_num = coord_num

    def __call__(self, *xi):
        v = xi[self.coord_num]
        if hasattr(v, 'dims'):
            return getattr(v, 'variable', v)
        return xr.Variable(self.dims, v)


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

    def __call__(self, *x):
        if len(x) != 1:
            raise ValueError('Only one coordinate should be given. '
                             'Given {}.'.format(len(x)))
        value = self.interp_obj(x[0])
        # consider x's dimension
        if hasattr(x[0], 'dims'):
            new_dim = x[0].dims[0]
            dims = list(self.dims)
            dims[self.dims.index(self.interp_dim)] = new_dim
        else:
            dims = self.dims
        return xr.Variable(dims, value)


def _concat_and_stack(*variables):
    """ Concatenate multiple variables, stack other dimensions and returns
    a 2D variable sized [_SAMPLE_DIM, _DIMENSION_DIM]
    """
    variables = [getattr(v, 'variable', v) for v in variables]
    broadcasted = xr.core.variable.broadcast_variables(*variables)
    x = xr.Variable.concat(broadcasted, dim=_DIMENSION_DIM)
    x = x.stack(**{_SAMPLE_DIM: broadcasted[0].dims}).transpose(
        _SAMPLE_DIM, _DIMENSION_DIM)
    return x, broadcasted[0].shape, broadcasted[0].dims


class _VariableNdInterp(_VaribaleInterp):
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
        self.dims = variable.dims
        self._shapes = {d: s for d, s in zip(variable.dims, variable.shape)}
        if isinstance(variable, xr.DataArray):
            variable = variable.variable

        # choose dim and x based on self.dims
        dim = [d for d in dim if d in self.dims]
        x = [x1 for x1 in x if len(set(x1.dims) & set(dim)) >= 1]
        # stack all the dims other than dim
        self._non_stack_dims = [d for d in self.dims if d not in dim]
        stacked = variable.stack(**{_SAMPLE_DIM: dim}).set_dims(
            [_SAMPLE_DIM] + self._non_stack_dims)
        x, _, _ = _concat_and_stack(*x)
        self.interp_obj = interp_cls(x, stacked.data, **kwargs)
        self.interp_dim = dim

    def __call__(self, *x):
        # TODO consider non-xarray object
        if isinstance(x[0], xr.DataArray):
            assert all(isinstance(xi, xr.DataArray) for xi in x)
            stacked_x, coord_shape, coord_dims = _concat_and_stack(*x)
            assert stacked_x.ndim == 2
            keep_dimorder = False if self.interp_dim != coord_dims else True
        else:
            raise TypeError('Invalid coordinate passed.')

        value = self.interp_obj(stacked_x)  # [_SAMPLE_DIM, _DIMENSION_DIM]
        variable = xr.Variable([_SAMPLE_DIM] + self._non_stack_dims, value)
        # unstack variable
        unstack_dims = OrderedDict()
        for d, s in zip(coord_dims, coord_shape):
            unstack_dims[d] = s

        result = variable.unstack(**{_SAMPLE_DIM: unstack_dims})
        if keep_dimorder:
            result = result.set_dims(self.dims)
        return result


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

    def __call__(self, *xi):
        dataset = self._to_temp_dataset()(*xi)
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

    def __call__(self, *xi):
        """ Get interpolated xarray object at new coordinate xi """
        # TODO consider the dimension of new_x
        variables = OrderedDict()
        coords = OrderedDict()
        for k, v in self._variables.items():
            v = v(*xi) if isinstance(v, _VaribaleInterp) else v.copy()
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


def _inject_doc_1d(func, func_name, description=None):
    try:
        doc = DocParser(getattr(interpolate, func_name).__doc__)
    except errors.NoDocstringError:
        return

    doc.replace_params(y='obj : xarray object\n')
    doc.add_params(
        coord='coord : string\n    Coordinate along which to interpolate.\n')
    doc.remove_params('x', 'axis')
    doc.reorder_params('obj', 'coord')

    doc.remove_sections('Notes', 'Examples')

    if description is not None:
        doc.insert_description(description)

    doc.insert_see_also(**{
        'scipy.interpolate.' + func_name:
        'scipy.interpolate.' + func_name +
        ' : Original scipy implementation\n'})

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


interp1d = partial(_wrap_interp1d, interpolate.interp1d)
_inject_doc_1d(interp1d, 'interp1d',
               description='interp1d(obj, coord, kind=\'linear\', copy=True, '
               'bounds_error=None, fill_value=nan, assume_sorted=False)')

PchipInterpolator = partial(_wrap_interp1d, interpolate.PchipInterpolator)
_inject_doc_1d(PchipInterpolator, 'PchipInterpolator',
               description='PchipInterpolator(obj, coord, extrapolate=None)')

Akima1DInterpolator = partial(_wrap_interp1d, interpolate.Akima1DInterpolator)
_inject_doc_1d(Akima1DInterpolator, 'Akima1DInterpolator',
               description='Akima1DInterpolator(obj, coord)')

CubicSpline = partial(_wrap_interp1d, interpolate.CubicSpline)
_inject_doc_1d(CubicSpline, 'CubicSpline',
               description='CubicSpline(obj, coord, bc_type=\'not-a-knot\', '
               'extrapolate=None)')


def _wrap_interp_nd(interp_cls, obj, *coords, **kwargs):
    # TODO consider dask array
    errors.raise_invalid_args(['x', 'axis'], kwargs)

    x = [obj[c] for c in coords]
    dim = xr.broadcast(*x)[0].dims

    if isinstance(obj, xr.DataArray):
        variable = _VariableNdInterp(interp_cls, obj.variable, dim, x,
                                     **kwargs)
        new_coords = OrderedDict()
        for k, v in obj.coords.items():
            if set(dim) <= set(v.dims) and k not in coords:
                new_coords[k] = _VariableNdInterp(interp_cls, v, dim, x,
                                                  **kwargs)
            elif k in coords:
                new_coords[k] = _VariableIdentity([dim])
            else:
                new_coords[k] = v
        return DataArrayInterp(variable, new_coords, obj.name)

    if isinstance(obj, xr.Dataset):
        variables = OrderedDict()
        for k, v in obj.variables.items():
            if dim in v.dims and k not in coords:
                variables[k] = _VariableNdInterp(interp_cls, v, dim, x,
                                                 **kwargs)
            elif k in coords:
                variables[k] = _VariableIdentity([dim])
            else:
                variables[k] = v
        return DatasetInterp(variables, obj.coords)


def _inject_doc_nd(func, func_name, description=None):
    try:
        doc = DocParser(getattr(interpolate, func_name).__doc__)
    except errors.NoDocstringError:
        return

    doc.add_params(
        obj='obj : xarray object\n',
        coord='*coord : strings\n    '
        'Coordinates along which to interpolate.\n')
    doc.reorder_params('obj', 'coord')
    doc.remove_params('points', 'values', 'x', 'y')

    doc.remove_sections('Examples')

    if description is not None:
        doc.insert_description(description)

    doc.insert_see_also(**{
        'scipy.interpolate.' + func_name:
        'scipy.interpolate.' + func_name +
        ' : Original scipy implementation\n'})

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


LinearNDInterpolator = partial(_wrap_interp_nd,
                               interpolate.LinearNDInterpolator)
_inject_doc_nd(LinearNDInterpolator, 'LinearNDInterpolator',
               description='LinearNDInterpolator(obj, *coords, '
               'fill_value=np.nan, rescale=False)')

NearestNDInterpolator = partial(_wrap_interp_nd,
                                interpolate.NearestNDInterpolator)
_inject_doc_nd(NearestNDInterpolator, 'NearestNDInterpolator',
               description='NearestNDInterpolator(obj, *coords)')

CloughTocher2DInterpolator = partial(_wrap_interp_nd,
                                     interpolate.CloughTocher2DInterpolator)
_inject_doc_nd(CloughTocher2DInterpolator, 'CloughTocher2DInterpolator',
               description='CloughTocher2DInterpolator(obj, *coords, '
               'fill_value=np.nan, tol=False, maxiter, rescale)')


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
