import xarray
import numpy as np
import scipy.interpolate
from .utils import get_maybe_last_dim_axis

def interp1d(darray, other_darray=None, fill_value=np.nan, mask_null=True,
             return_interpolator=False,
             interp_class=scipy.interpolate.UnivariateSpline, **class_kwargs):
    if other_darray is None:
        other_darray = darray
    if darray.ndim != 1:
        raise ValueError(
            'Signal has %i dimensions, need 1 for interpolation' % darray.ndim)
    if mask_null:
        darray = darray[darray.notnull()]
    dim = darray.dims[0]
    axis_data = np.asarray(darray.coords[dim])
    interpolator = interp_class(axis_data, np.asarray(darray.data),
                                **class_kwargs)
    if return_interpolator:
        return interpolator
    interp_x = np.asarray(other_darray.coords[dim])
    ret_data = interpolator(interp_x)
    if fill_value is not None:
        left = interp_x < axis_data[0]
        right = axis_data[-1] < interp_x
        outside = left | right
        if fill_value == 'edge':
            data_inside = ret_data[~outside]
            ret_data[left] = data_inside[0]
            ret_data[right] = data_inside[-1]
        elif isinstance(fill_value, (tuple, list)):
            ret_data[left] = fill_value[0]
            ret_data[right] = fill_value[1]
        else:
            ret_data[outside] = fill_value
    return xarray.DataArray(ret_data, coords=[(dim, other_darray.coords[dim])],
                            attrs=darray.attrs, name=darray.name)


@xarray.register_dataarray_accessor('interp')
class InterpolationAccessor(object):
    """Accessor for interpolation services"""

    def __init__(self, darray):
        self.darray = darray

    def _check_same_ndim(self, other):
        if isinstance(other, xarray.DataArray) and self.darray.ndim != other.ndim:
            raise ValueError('{}D interpolated array does not fit onto {}D target'.format(
                self.darray.ndim, other.ndim))
        elif isinstance(other, np.ndarray) and other.ndim == 2 and self.darray.ndim != other.shape[1]:
            raise ValueError('interpolation points table has {} columns, but interpolated array is {}D'.format(other.shape[1], self.darray.ndim))


    def __call__(self, other, method='linear', bounds_error=False, fill_value=None):
        if self.darray.ndim == 1:
            return self.d1(other, method, bounds_error, fill_value)
        else:
            return self.n(other, method, bounds_error, fill_value)

    def d1(self, other, kind='linear', dim=None, bounds_error=None, fill_value='extrapolate', copy=False, assume_sorted=False, return_interpolator=False):
        dim, axis = get_maybe_last_dim_axis(self.darray, dim)
        darray = self.darray.dropna(dim=dim)
        x = darray.coords[dim]
        interp = scipy.interpolate.interp1d(x, darray, kind=kind,
                                            axis=axis, copy=copy, bounds_error=bounds_error, fill_value=fill_value,
                                            assume_sorted=assume_sorted)
        if return_interpolator:
            return interp
        try:
            xi = other.coords[dim]
            coords = other.coords
        except KeyError:
            xi = other
            coords = {dim: xi}
        yi = interp(xi)
        ret = xarray.DataArray(yi, name=darray.name,
                               dims=darray.dims, coords=coords, attrs=darray.attrs)
        return ret

    def n(self, other, method='linear', bounds_error=False, fill_value=None, return_interpolator=False):
        if not return_interpolator:
            self._check_same_ndim(other)
        points = self.darray.coords.values()
        interp = scipy.interpolate.RegularGridInterpolator(points, self.darray.values,
                                                           method=method, bounds_error=bounds_error,
                                                           fill_value=fill_value,
        )
        if return_interpolator:
            return interp
        if isinstance(other, xarray.DataArray):
            other_darray = other
            xi_m = np.meshgrid(*other_darray.coords.values())
            xi = np.transpose([d.ravel() for s in xi_m])
            new_coords = other_darray.coords
        else:
            xi = other
            new_coords = {d: xi[i] for (i,d) in enumerate(delf.darray.dims)}
        values_x = interp(xi)
        ret = xarray.DataArray(values_x.reshape(xi_m[0]),
                               coords=new_coords,
                               dims=self.darray.dims,
                               attrs=self.darray.attrs,
                               name=self.darray.name
        )
        return ret
