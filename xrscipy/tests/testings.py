from __future__ import absolute_import, division, print_function

import numpy as np
import xarray as xr


def get_obj(mode):
    rng = np.random.RandomState(0)

    shapes = [10, 12, 15]
    if mode in [0, 1]:
        ndim = 1 if mode == 0 else 3
        dims = ['x', 'y', 'z']
        coords = {}
        coords['x'] = np.arange(shapes[0]) * 0.2
        if ndim >= 2:
            coords['z'] = np.linspace(0, 1, shapes[2])
        coords['time'] = ('x', ), np.linspace(0, 1, shapes[0])
        da = xr.DataArray(rng.randn(*shapes[:ndim]), dims=dims[:ndim],
                          coords=coords)
        da.attrs['comment'] = 'dummy comment.'
        # scalar coordinate
        da['scalar'] = 3.141592
        return da

    if mode == 2:  # dask
        da = get_obj(mode=1)
        return da.chunk({'x': 5, 'y': 4, 'z': 5})

    if mode == 3:  # dataset
        ds = xr.Dataset({})
        ds['a'] = get_obj(mode=0)
        ds['b'] = get_obj(mode=1)
        return ds

    if mode == 4:  # Lazy array from netCDF
        pass

    if mode == 5:  # With time coordinate
        da = get_obj(mode=1)
        # da['time'] = pd.TimeRange()
