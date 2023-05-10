import functools
from typing import Callable, TypeVar

import numpy as np
import xarray as xr

from xrscipy.errors import raise_not_1d

_DAS = TypeVar("_DAS", xr.DataArray, xr.Dataset)


def apply_func_to_DAS(func: Callable, x: _DAS, *dims, **kwargs) -> _DAS:
    """
    Apply func to each DataArray of x (or to x if x is a DataArray).

    keep_coords: 'apply' | 'keep' | 'drop'
    """
    keep_coords = kwargs.pop("keep_coords", "apply")

    if not isinstance(x, (xr.DataArray, xr.Dataset)):
        raise TypeError(f"Got invalid data type {type(x)}.")

    for d in dims:
        if d not in x.dims:
            raise ValueError(f"{d} is not a valid dimension for the object. The valid dimension are {x.dims}.")

    if isinstance(x, xr.DataArray):
        """apply to mock dataset"""
        # noinspection PyProtectedMember
        result = apply_func_to_DAS(func, x._to_temp_dataset(), *dims, keep_coords=keep_coords)
        # Drop unnecessary coordinate.
        da = next(iter(result.data_vars.values()))
        da.name = x.name
        return da

    ds = xr.Dataset({})
    if keep_coords in ["keep", "drop"]:
        for key in x.data_vars:
            ds[key] = func(x[key].variable) if any(d in x[key].dims for d in dims) else x[key]
        for key in x.coords:
            if keep_coords != "drop" or all(d not in dims for d in x[key].dims):
                ds.coords[key] = x[key]

    else:  # also applied to coord
        for key in x.variables:
            ds[key] = func(x[key].variable) if any(d in x[key].dims for d in dims) else x[key]
        ds = ds.set_coords(list(x.coords.keys()))

    return ds


_F = TypeVar("_F", bound=Callable)


def partial(f0: Callable, f1: _F, *args, **kwargs) -> _F:
    """wrapper around partial that conserves the name of the second function"""
    f = functools.partial(f0, f1, *args, **kwargs)
    f.__name__ = f1.__name__
    return f


def get_1D_spacing(x: xr.DataArray) -> float:
    """get avg. spacing from the da"""
    raise_not_1d(x)
    dx = np.diff(x)
    mean, std = dx.mean(), dx.std()

    if np.abs(std / mean) > 1e-4:  # heuristic value
        raise ValueError("Coordinate for FFT should be evenly spaced.")

    return mean
