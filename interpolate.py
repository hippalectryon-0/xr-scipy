import xarray
import numpy as np
import scipy.interpolate

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
    return xarray.DataArray(ret_data, {dim: other_darray.coords[dim]},
                            attrs=darray.attrs, name=darray.name)
