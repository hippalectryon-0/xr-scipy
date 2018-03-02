from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import pytest
import xarray as xr

from xrscipy import integrate


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
@pytest.mark.parametrize('func', ['trapz', 'cumtrapz'])
@pytest.mark.parametrize('dim', ['x', 'time'])
def test_integrate(ndim, func, dim):
    da = get_da(ndim)

    axis = da.get_axis_num(da[dim].dims[0])
    actual = getattr(integrate, func)(da, dim)
    kwargs = {}
    if func == 'cumtrapz':
        kwargs['initial'] = 0
    expected = getattr(sp.integrate, func)(da.values, x=da[dim].values,
                                           axis=axis, **kwargs)
    assert (actual.values == expected).all()

    # make sure the original data does not change
    da.values.ndim == ndim

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if 'x' not in v.dims:
            assert da[key].identical(actual[key])


def test_integrate_dataset():
    ds = xr.Dataset({})
    ds['a'] = get_da(1)
    ds['b'] = get_da(3)

    actual = integrate.trapz(ds, dim='z')
    assert actual['a'].identical(ds['a'])
    assert actual['b'].identical(integrate.trapz(ds['b'], dim='z'))


def test_integrate_error():
    # not sorted
    da = xr.DataArray([0, 1, 2], dims=['x'], coords={'x': [2, 3, 0]})
    with pytest.raises(ValueError):
        integrate.trapz(da, 'x')

    # wrong argument
    with pytest.raises(TypeError):
        integrate.trapz(da, axis='x')
