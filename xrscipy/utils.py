from typing import Callable

import xarray as xr

# Used as the key corresponding to a DataArray's variable when converting
# arbitrary DataArray objects to datasets
_TEMP_DIM = xr.core.utils.ReprObject('<temporal-dim>')


def wrap_dataset(func: Callable, y: xr.Dataset | xr.DataArray, *dims, **kwargs) -> xr.Dataset:
    """
    Wrap Dataset for Array func. If y is Dataset, the func is applied for all
    the data vars if it has dim in its dimension.

    keep_coords: 'apply' | 'keep' | 'drop'
    """
    keep_coords = kwargs.pop('keep_coords', 'apply')

    if not isinstance(y, (xr.DataArray, xr.Dataset)):
        raise TypeError(f'Got invalid data type {type(y)}.')

    for d in dims:
        if d not in y.dims:
            raise ValueError(f'{d} is not a valid dimension for the object. The valid dimension is {y.dims}.')

    if isinstance(y, xr.DataArray):
        # noinspection PyProtectedMember
        result = wrap_dataset(func, y._to_temp_dataset(), *dims, keep_coords=keep_coords)
        # Drop unnecessary coordinate.
        da = result[list(result.data_vars.keys())[0]]
        da.name = y.name
        return da

    ds = xr.Dataset({})
    if keep_coords in ['keep', 'drop']:
        for key in y.data_vars:
            ds[key] = (func(y[key].variable) if any(d in y[key].dims for d in dims) else y[key])
        for key in y.coords:
            if keep_coords != 'drop' or all(d not in dims for d in y[key].dims):
                ds.coords[key] = y[key]

    else:  # also applied to coord
        for key in y.variables:
            ds[key] = (func(y[key].variable) if any(d in y[key].dims for d in dims) else y[key])
        ds = ds.set_coords(list(y.coords.keys()))

    return ds
