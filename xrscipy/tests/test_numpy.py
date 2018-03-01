from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import xarray as xr

from xrscipy import gradient


def get_da(ndim, ascend=False):
    shapes = [10, 11, 12]
    dims = ['x', 'y', 'z']
    coords = {'x': np.arange(shapes[0]), 'z': np.linspace(0, 1, shapes[2])}
    da = xr.DataArray(np.random.randn(*shapes[:ndim]), dims=dims[:ndim],
                      coords={k: v for k, v in coords.items() if k in
                              dims[:ndim]})
    da.attrs['comment'] = 'dummy comment.'
    return da


@pytest.mark.parametrize('ndim', [1, 3])
@pytest.mark.parametrize('edge_order', [1, 2])
def test_gradient(ndim, edge_order):
    da = get_da(ndim)
    dim = 'x'
    axis = da.get_axis_num(dim)
    actual = gradient(da, dim, edge_order=edge_order)
    expected = np.gradient(da.values, da[dim].values, edge_order=edge_order,
                           axis=axis)
    assert (actual.values == expected).all()

    # make sure the original data does not change
    da.values.ndim == ndim

    # make sure the coordinate is propagated
    for key in da.coords:
        if key != 'x':
            assert da[key].identical(actual[key])
