from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import pytest
import xarray as xr

from xrscipy import integrate


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
@pytest.mark.parametrize('func', ['trapz', 'cumtrapz'])
def test_integrate(ndim, func):
    da = get_da(ndim)
    dim = 'x'
    axis = da.get_axis_num(dim)
    actual = getattr(integrate, func)(da, dim)
    expected = getattr(sp.integrate, func)(da.values, x=da[dim].values,
                                           axis=axis)
    assert (actual.values == expected).all()

    # make sure the original data does not change
    da.values.ndim == ndim

    # make sure the coordinate is propagated
    for key in da.coords:
        if key != 'x':
            assert da[key].identical(actual[key])


def test_integrate_dataset():
    ds = xr.Dataset({})
    ds['a'] = get_da(1)
    ds['b'] = get_da(3)

    actual = integrate.trapz(ds, dim='z')
    assert actual['a'].identical(ds['a'])
    assert actual['b'].identical(integrate.trapz(ds['b'], dim='z'))
