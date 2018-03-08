import xarray as xr


def wrap_dataset(func, y, *dims, keep_coords='apply'):
    """
    Wrap Dataset for Array func. If y is Dataset, the func is applied for all
    the data vars if it has dim in its dimension.

    keep_coords: 'apply' | 'keep' | 'drop'
    """
    if not isinstance(y, (xr.DataArray, xr.Dataset)):
        raise TypeError('Invalid data type {} is given.'.format(type(y)))

    for d in dims:
        if d not in y.dims:
            raise ValueError('{} is not a valid dimension for the object. '
                             'The valid dimension is {}.'.format(d, y.dims))

    if isinstance(y, xr.DataArray):
        result = wrap_dataset(func, y._to_temp_dataset(), *dims,
                              keep_coords=keep_coords)
        # Drop unnecessary coordinate.
        da = result[list(result.data_vars.keys())[0]]
        da.name = y.name
        return da

    ds = xr.Dataset({})
    if keep_coords in ['keep', 'drop']:
        for key in y.data_vars:
            if any(d in y[key].dims for d in dims):
                ds[key] = func(y[key].variable)
            else:
                ds[key] = y[key]

        for key in y.coords:
            if (keep_coords != 'drop'
                    or not any(d in dims for d in y[key].dims)):
                ds.coords[key] = y[key]

    else:  # also applied to coord
        for key in y.variables:
            if any(d in y[key].dims for d in dims):
                ds[key] = func(y[key].variable)
            else:
                ds[key] = y[key]
        ds = ds.set_coords(list(y.coords.keys()))

    return ds
