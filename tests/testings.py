"""utilities for tests"""

import numpy as np
import xarray as xr


def get_obj(mode: int) -> xr.DataArray | xr.Dataset:
    rng = np.random.RandomState(0)

    if mode in {0, 1}:  # 0 = 1D, 1 = 3D
        ndim = 1 if mode == 0 else 3
        dims = ["x", "y", "z"]
        shapes = [9, 12, 15]
        coords = {"x": np.arange(shapes[0]) * 0.2}
        if ndim >= 2:
            coords["z"] = np.linspace(0, 1, shapes[2])
        coords["time"] = ("x",), np.linspace(0, 1, shapes[0])
        da = xr.DataArray(rng.randn(*shapes[:ndim]), dims=dims[:ndim], coords=coords)
        da.attrs["comment"] = "dummy comment."
        # scalar coordinate
        da["scalar"] = 3.141592
        return da
    elif mode == 2:
        da = get_obj(mode=1)
        return da.chunk({"x": 5, "y": 4, "z": 5})
    elif mode == 3:
        ds = xr.Dataset({})
        ds["a"] = get_obj(mode=0)
        ds["b"] = get_obj(mode=1)
        return ds
    elif mode == 5:
        return get_obj(mode=1)
