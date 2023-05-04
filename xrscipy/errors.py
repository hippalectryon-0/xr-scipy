import numpy as np


def raise_invalid_args(keys, kwargs):
    for key in keys:
        if kwargs.pop(key, None) is not None:
            raise ValueError(
                f"{key} is not valid key for xr-scipy. Given {kwargs[key]}."
            )


def raise_not_sorted(coord):
    raise_not_1d(coord)
    if not (np.diff(coord) > 0).all() and not (np.diff(coord) < 0).all():
        raise ValueError("Coordinate should be sorted first. See xr.sortby.")


def raise_not_1d(coord):
    if coord.ndim != 1:
        raise ValueError(
            f"Coordinate should be 1-dimensional. {coord.ndim}-d array is given."
        )


class NoDocstringError(Exception):
    pass
