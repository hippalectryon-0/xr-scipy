import xarray
import numpy as np
import scipy.signal
from .utils import get_maybe_last_dim_axis, get_sampling_step

def _create_lags(length, dt):
    return (np.arange(length) - length / 2.0) * dt


def correlate(darray, other_darray, mode='full'):
    darray, other_darray = xarray.align(darray, other_darray, join='outer', copy=True)
    corr = scipy.signal.correlate(darray, other_darray, mode=mode)
    new_coords = [(dim, _create_lags(corr.shape[i], get_sampling_step(darray, dim)))
                  for (i, dim) in enumerate(darray.dims)]
    return xarray.DataArray(corr, new_coords,
                            name='correlate_{}_{}'.format(
                                darray.name, other_darray.name))


def relextrema(darray, comparator, dim=None, order=1, distance=None,
               mode='clip'):
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    if distance is not None:
        dt = get_sampling_step(darray, dim)
        order = int(np.rint(distance / dt))
    idxs = scipy.signal.argrelextrema(np.asarray(darray), comparator, axis,
                                      order, mode)
    return darray[idxs]


def relmax(darray, dim=None, order=1, distance=None, mode='clip'):
    return relextrema(darray, np.greater, dim, order, distance, mode)


def relmin(darray, dim=None, order=1, distance=None, mode='clip'):
    return relextrema(darray, np.less, dim, order, distance, mode)
