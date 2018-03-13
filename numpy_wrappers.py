"""xarray-aware wrappers for some numpy functions"""
import numpy as np
import xarray


def angle(darray, unwrap=False):
    angle = np.angle(darray)
    if unwrap:
        angle = np.unwrap(angle)
    return darray.__array_wrap__(angle)


def unwrap(darray, discont=np.pi, dim=None):
    if dim is None:
        axis = -1
    else:
        axis = darray.get_axis_num(dim)
    return darray.__array_wrap__(np.unwrap(darray, discont, axis))

def gradient(darray, div_by_dim=True, dim=None, **kwargs):
    if div_by_dim:
        dx = [np.gradient(darray.coords[d].values) for d in darray.dims]
    else:
        dx = []
    if dim is not None:
        axis = darray.get_axis_num(dim)
        kwargs.setdefault('axis', axis)
        dx = [dx[axis]]
    ret = np.gradient(darray.values, *dx, **kwargs)
    if isinstance(ret, list):
        ds = xarray.Dataset({d: darray.__array_wrap__(deriv)
                             for (d, deriv) in zip(darray.dims, ret)},
                            attrs={'name': darray.name}
        )
        return ds
    else:
        return darray.__array_wrap__(ret)

