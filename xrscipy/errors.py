"""xr-scipy-specific errors"""

import numpy as np
import xarray as xr


def raise_invalid_args(keys: list[str], kwargs: dict) -> None:
    """makes sure keys of <keys> are not in <kwargs>"""
    for key in keys:
        if kwargs.get(key) is not None:
            raise ValueError(f"{key} is not valid key for xr-scipy. Given {kwargs[key]}.")


def raise_not_sorted(coord) -> None:
    """makes sure the array is 1D and sorted"""
    raise_not_1d(coord)
    if (np.diff(coord) > 0).all() or (np.diff(coord) < 0).all():
        return
    raise ValueError("Coordinate should be sorted first. See xr.sortby.")


def raise_not_1d(coord: xr.DataArray) -> None:
    """make sure the array is 1D"""
    if coord.ndim != 1:
        raise ValueError(f"Coordinate should be 1-dimensional. {coord.ndim}-d array is given.")
