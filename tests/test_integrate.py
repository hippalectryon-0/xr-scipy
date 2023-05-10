"""tests for integrate"""

import numpy as np
import pytest
import scipy as sp

from xrscipy import integrate

from .testings import get_obj


@pytest.mark.parametrize("mode", [1])
@pytest.mark.parametrize("dx", [1, 0.1, 10])
@pytest.mark.parametrize("dim", ["x", "time"])
def test_integrate_romb(mode, dx, dim):
    da = get_obj(mode)

    axis = da.get_axis_num(da[dim].dims[0])
    actual = integrate.romb(da, dim, dx=dx)
    expected: np.ndarray = sp.integrate.romb(da.values, dx=dx, axis=axis)
    assert (actual.values == expected).all()

    # make sure the original data does not change
    assert da.values.ndim == get_obj(mode).ndim

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if "x" not in v.dims:
            assert da[key].identical(actual[key])


@pytest.mark.parametrize("mode", [1])
@pytest.mark.parametrize("func", ["trapezoid", "cumulative_trapezoid", "simpson"])
@pytest.mark.parametrize("dim", ["x", "time"])
def test_integrate(mode, func, dim):
    da = get_obj(mode)

    axis = da.get_axis_num(da[dim].dims[0])
    actual = getattr(integrate, func)(da, dim)
    kwargs = {}
    if func == "cumulative_trapezoid":
        kwargs["initial"] = 0
        actual = actual.transpose(*da.dims)
    expected: np.ndarray = getattr(sp.integrate, func)(da.values, x=da[dim].values, axis=axis, **kwargs)
    assert (actual.values == expected).all()

    # make sure the original data does not change
    assert da.values.ndim == get_obj(mode).ndim

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if "x" not in v.dims:
            assert da[key].identical(actual[key])
