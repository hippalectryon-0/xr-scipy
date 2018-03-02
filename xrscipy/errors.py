import numpy as np


def raise_invalid_args(keys, kwargs):
    for key in keys:
        if kwargs.pop(key, None) is not None:
            raise ValueError(
                '{} is not valid key for xr-scipy. Given {}.'.format(
                    key, kwargs[key]))


def raise_not_sorted(coord):
    raise_not_1d(coord)
    if not (np.diff(coord) > 0).all() and not (np.diff(coord) < 0).all():
        raise ValueError('Coordinate should be sorted first. See xr.sortby.')
    

def raise_not_1d(coord):
    if coord.ndim != 1:
        raise ValueError('Coordinate should be 1-dimensional. {}-d array is '
                         'given.'.format(coord.ndim))
