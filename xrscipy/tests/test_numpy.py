from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import xarray as xr

from xrscipy import gradient


def get_da(ndim, ascend=False):
    shapes = [10, 11, 12]
    dims = ['x', 'y', 'z']
    coords = {}
    coords['x'] = np.arange(shapes[0]) * 0.2
    if ndim >= 2:
        coords['z'] = np.linspace(0, 1, shapes[2])
    coords['time'] = ('x', ), np.linspace(0, 1, shapes[0])
    da = xr.DataArray(np.random.randn(*shapes[:ndim]), dims=dims[:ndim],
                      coords=coords)
    da.attrs['comment'] = 'dummy comment.'
    return da


@pytest.mark.parametrize('ndim', [1, 3])
@pytest.mark.parametrize('edge_order', [1, 2])
@pytest.mark.parametrize('dim', ['x', 'time'])
def test_gradient(ndim, edge_order, dim):
    da = get_da(ndim)

    axis = da.get_axis_num(da[dim].dims[0])
    actual = gradient(da, dim, edge_order=edge_order)
    expected = np.gradient(da.values, da[dim].values, edge_order=edge_order,
                           axis=axis)

    assert (actual.values == expected).all()

    # make sure the original data does not change
    da.values.ndim == ndim

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if 'x' not in v.dims:
            assert da[key].identical(actual[key])
