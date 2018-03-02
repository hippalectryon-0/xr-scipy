from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import pytest
import xarray as xr

from xrscipy import interpolate


def get_da(ndim):
    shapes = [10, 11, 12]
    dims = ['x', 'y', 'z']
    coords = {}
    coords['x'] = np.arange(shapes[0])
    if ndim >= 2:
        coords['z'] = np.linspace(0, 1, shapes[2])
    coords['time'] = ('x', ), np.linspace(0, 1, shapes[0])
    da = xr.DataArray(np.random.randn(*shapes[:ndim]), dims=dims[:ndim],
                      coords=coords)
    da.attrs['comment'] = 'dummy comment.'
    return da


@pytest.mark.parametrize('ndim', [1, 3])
@pytest.mark.parametrize('func', ['interp1d', 'PchipInterpolator',
                                  'Akima1DInterpolator', 'CubicSpline'])
@pytest.mark.parametrize('dim', ['x'])
def test_interpolate1d(ndim, func, dim):
    da = get_da(ndim)
    new_x = np.linspace(1, 8, 13)

    axis = da.get_axis_num(da[dim].dims[0])
    actual = getattr(interpolate, func)(da, dim)(new_x)
    expected = getattr(sp.interpolate, func)(x=da[dim].values, y=da.values,
                                             axis=axis)(new_x)
    assert (actual.values == expected).all()

    # make sure the original data does not change
    da.values.ndim == ndim

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if 'x' not in v.dims:
            assert da[key].identical(actual[key])
