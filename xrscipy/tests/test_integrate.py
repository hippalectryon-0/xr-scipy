from __future__ import absolute_import, division, print_function

import scipy as sp
import pytest
import xarray as xr

from xrscipy import integrate
from xrscipy.docs import DocParser
from .testings import get_obj


_trapz_funcs = [integrate.trapz]
_trapz_names = ['trapz']
_cumtrapz_names = ['cumtrapz']
_simps_names = ['simps']
_romb_names = ['romb']
if hasattr(integrate, 'trapezoid'):  # scipy >= 1.6.0
    _trapz_funcs += [integrate.trapezoid]
    _trapz_names += ['trapezoid']
    _cumtrapz_names += ['cumulative_trapezoid']
    _simps_names += ['simpson']


@pytest.mark.parametrize('mode', [1, 1])
@pytest.mark.parametrize('func', _trapz_names + _cumtrapz_names)
@pytest.mark.parametrize('dim', ['x', 'time'])
def test_integrate(mode, func, dim):
    da = get_obj(mode)

    axis = da.get_axis_num(da[dim].dims[0])
    actual = getattr(integrate, func)(da, dim)
    kwargs = {}
    if func in _cumtrapz_names:
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


@pytest.mark.parametrize('trapz_func', _trapz_funcs)
def test_integrate_dataset(trapz_func):
    ds = get_obj(mode=3)

    actual = trapz_func(ds, coord='z')
    assert actual['a'].identical(ds['a'])
    assert actual['b'].identical(integrate.trapz(ds['b'], coord='z'))


@pytest.mark.parametrize('trapz_func', _trapz_funcs)
def test_integrate_error(trapz_func):
    # not sorted
    da = xr.DataArray([0, 1, 2], dims=['x'], coords={'x': [2, 3, 0]})
    with pytest.raises(ValueError):
        trapz_func(da, 'x')

    # wrong argument
    with pytest.raises(TypeError):
        trapz_func(da, axis='x')


@pytest.mark.parametrize(
    'func',
    _trapz_names + _cumtrapz_names + _simps_names + _romb_names)
def test_doc_all(func):
    parser = DocParser(func.__doc__)

    not_included_keys = ['x', 'axis', 'dx']
    for k in not_included_keys:
        assert k not in parser.parameters.keys()


@pytest.mark.parametrize('trapz_func', _trapz_funcs)
def test_doc(trapz_func):
    parser = DocParser(trapz_func.__doc__)

    not_included_keys = ['x', 'axis', 'dx']
    for k in not_included_keys:
        assert k not in parser.parameters.keys()
