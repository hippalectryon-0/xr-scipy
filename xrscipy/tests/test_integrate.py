from __future__ import absolute_import, division, print_function

import scipy as sp
import pytest
import xarray as xr

from xrscipy import integrate
from .testings import get_obj


@pytest.mark.parametrize('mode', [1, 1])
@pytest.mark.parametrize('func', ['trapz', 'cumtrapz'])
@pytest.mark.parametrize('dim', ['x', 'time'])
def test_integrate(mode, func, dim):
    da = get_obj(mode)

    axis = da.get_axis_num(da[dim].dims[0])
    actual = getattr(integrate, func)(da, dim)
    kwargs = {}
    if func == 'cumtrapz':
        kwargs['initial'] = 0
    expected = getattr(sp.integrate, func)(da.values, x=da[dim].values,
                                           axis=axis, **kwargs)
    assert (actual.values == expected).all()

    # make sure the original data does not change
    da.values.ndim == get_obj(mode).ndim

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if 'x' not in v.dims:
            assert da[key].identical(actual[key])


def test_integrate_dataset():
    ds = get_obj(mode=3)

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
