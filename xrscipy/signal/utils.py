r"""utils for signal"""

from __future__ import annotations

import typing
import warnings

import xarray as xr


class UnevenSamplingWarning(Warning):
    pass


class FilteringNaNWarning(Warning):
    pass


class DecimationWarning(Warning):
    pass


# always (not just once) show decimation warnings to see the responsible signal
warnings.filterwarnings("always", category=DecimationWarning)
# always (not just once) show filtering NaN warnings to see the responsible signal
warnings.filterwarnings("always", category=FilteringNaNWarning)
# always (not just once) show decimation warnings to see the responsible signal
warnings.filterwarnings("always", category=UnevenSamplingWarning)


def get_maybe_only_dim(darray: xr.DataArray, dim: str) -> str:
    """
    Returns <dim>, or the only dimension of the array

    Parameters
    ----------
    darray : DataArray
        An xarray DataArray.
    dim : string
        Specifies the dimension.
    """
    if dim is not None:
        return dim
    if len(darray.dims) != 1:
        raise ValueError("Specify the dimension")
    if not isinstance(res := darray.dims[0], str):
        raise ValueError("Got unexpectedly many dims")
    return typing.cast(str, res)


def get_maybe_last_dim_axis(darray: xr.DataArray, dim: str = None) -> tuple[str, int | tuple[int, ...]]:
    """get the axis associated with dim, where by default dim is the dimension of the last axis"""
    if dim is None:
        axis = darray.ndim - 1
        dim = darray.dims[axis]
    else:
        axis = darray.get_axis_num(dim)
    return dim, axis


def get_sampling_step(darray: xr.DataArray, dim: str = None, rtol: float = 1e-3) -> float:
    """get avg step along dimension"""
    dim = get_maybe_only_dim(darray, dim)
    coord = darray.coords[dim].data
    step_avg = (coord[-1] - coord[0]) / (len(coord) - 1)  # N-1 segments
    step_first = coord[1] - coord[0]

    if abs(step_avg - step_first) > rtol * min(step_first, step_avg):
        # show warning at caller level to see which signal it is related to
        warnings.warn(
            f"Average sampling {step_avg:.3g} != first sampling step {step_first:.3g}",
            UnevenSamplingWarning,
            stacklevel=2,
        )
    return step_avg  # should be more precise
