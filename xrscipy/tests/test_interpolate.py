from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import pytest

from xrscipy import interpolate
from .testings import get_obj


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('func', ['interp1d', 'PchipInterpolator',
                                  'Akima1DInterpolator', 'CubicSpline'])
@pytest.mark.parametrize('dim', ['x', 'time'])
def test_interpolate1d(mode, func, dim):
    da = get_obj(mode)
    new_x = np.linspace(1, 8, 13) * 0.1

    axis = da.get_axis_num(da[dim].dims[0])
    actual = getattr(interpolate, func)(da, dim)(new_x)
    expected = getattr(sp.interpolate, func)(x=da[dim].values, y=da.values,
                                             axis=axis)(new_x)
    assert (actual.values == expected).all()

    # make sure the original data does not change
    da.values.ndim == get_obj(mode).ndim

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if 'x' not in v.dims:
            assert da[key].identical(actual[key])
