import xarray as xr


def wrap_dataset(func, y, dim, apply_to_coord=False):
    """
    Wrap Dataset for Array func. If y is Dataset, the func is applied for all
    the data vars if it has dim in its dimension.
    """
    if not isinstance(y, (xr.DataArray, xr.Dataset)):
        raise TypeError('Invalid data type {} is given.'.format(type(y)))

    if dim not in y.dims:
        raise ValueError('{} is not a valid dimension for the object. The '
                         'valid dimension is {}.'.format(dim, y.dims))

    if isinstance(y, xr.DataArray):
        result = wrap_dataset(func, y._to_temp_dataset(), dim,
                              apply_to_coord=apply_to_coord)
        # Drop unnecessary coordinate.
        da = result[list(result.data_vars.keys())[0]]
        da.name = y.name
        return da

    ds = xr.Dataset({})
    if not apply_to_coord:
        for key in y.data_vars.keys():
            if dim in y[key].dims:
                ds[key] = func(y[key].variable)
            else:
                ds[key] = y[key]
        ds.coords.update(y.coords)

    else:  # also applied to coord
        for key in y.variables.keys():
            if dim in y[key].dims:
                ds[key] = func(y[key])
            else:
                ds[key] = y[key]
        ds = ds.set_coords(*y.coords.name())

    return ds
