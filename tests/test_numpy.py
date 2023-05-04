from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from xrscipy import gradient
from .testings import get_obj


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("edge_order", [1, 2])
@pytest.mark.parametrize("dim", ["x", "time"])
def test_gradient(mode, edge_order, dim):
    da = get_obj(mode)

    axis = da.get_axis_num(da[dim].dims[0])
    actual = gradient(da, dim, edge_order=edge_order)
    expected = np.gradient(da.values, da[dim].values, edge_order=edge_order, axis=axis)

    assert (actual.values == expected).all()

    # make sure the original data does not change
    assert da.values.ndim == get_obj(mode).ndim

    # make sure the coordinate is propagated
    for key, v in da.coords.items():
        if "x" not in v.dims:
            assert da[key].identical(actual[key])
