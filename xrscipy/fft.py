"""mirrors numpy.fft"""
import functools
from typing import Callable, TypeVar

import numpy as np
import xarray as xr
from numpy import fft as fft_

from . import errors, utils
from .docs import CDParam, DocParser

_F = TypeVar("_F", bound=Callable)


def partial(f0: Callable, f1: _F, *args, **kwargs) -> _F:
    """wrapper around partial that conserves the name of the second function"""
    f = functools.partial(f0, f1, *args, **kwargs)
    f.__name__ = f1.__name__
    return f


def _get_spacing(x: xr.DataArray):
    if x.ndim != 1:
        raise ValueError(f"Coordinate for FFT should be one dimensional. Axis {x.name} is {x.ndim}-dimensional.")
    dx = np.diff(x)
    mean = dx.mean()
    jitter = dx.std()

    if np.abs(jitter / mean) > 1e-4:  # heuristic value
        raise ValueError("Coordinate for FFT should be evenly spaced.")

    return mean


def _wrap1d(func: Callable, freq_func: Callable, y: xr.DataArray, coord: str, **kwargs) -> xr.Dataset:
    """Wrap function for fft1d"""
    errors.raise_invalid_args(["axis", "overwrite_x"], kwargs)
    errors.raise_not_sorted(y[coord])

    # In case of dim is a non-dimensional coordinate.
    x = y[coord]
    dim = x.dims[0]

    kwargs["axis"] = -1

    def apply_func(v: xr.DataArray) -> xr.DataArray:
        """
        Parameters
        ----------
        v: xr.Variable
        """
        result = xr.apply_ufunc(
            func,
            v,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            kwargs=kwargs,
            exclude_dims={dim},
        )
        return result.set_dims(v.dims)

    ds = utils.wrap_dataset(apply_func, y, dim, keep_coords="drop")

    # attach frequency coordinate
    size = kwargs.pop("n", None)
    if size is None:
        size = len(y[dim])
    freq = freq_func(size, _get_spacing(x))
    ds[coord] = (dim,), freq
    return ds


def _wrapnd(func: Callable, freq_func: Callable, y: xr.DataArray, *coords, **kwargs) -> xr.Dataset:
    """Wrap function for fftnd"""
    errors.raise_invalid_args(["axes", "overwrite_x"], kwargs)
    shape = kwargs.pop("s", None)

    if shape is not None and not isinstance(shape, dict):
        raise TypeError(f"shape should be a dict mapping from coord name to size. Given {shape}.")

    for c in coords:
        errors.raise_not_sorted(y[c])

    # In case of dim is a non-dimensional coordinate.
    xs = [y[c] for c in coords]
    dims = [x.dims[0] for x in xs]
    shape = {d: len(y[d]) if shape is None or c not in shape else shape[c] for d, c in zip(dims, coords)}
    dxs = [_get_spacing(x) for x in xs]

    def apply_func(v: xr.DataArray) -> xr.DataArray:
        """
        Parameters
        ----------
        v: xr.Variable
        """
        kwargs_tmp = kwargs.copy()
        kwargs_tmp.pop("s", None)
        input_core_dims = [d for d in dims if d in v.dims]
        kwargs_tmp["axes"] = -np.arange(len(input_core_dims))[::-1] - 1
        if shape is not None:
            kwargs_tmp["s"] = [shape[d] for d in input_core_dims]
        result = xr.apply_ufunc(
            func,
            v,
            input_core_dims=[input_core_dims],
            output_core_dims=[input_core_dims],
            kwargs=kwargs_tmp,
            exclude_dims={*input_core_dims},
        )
        return result.set_dims(v.dims)

    ds = utils.wrap_dataset(apply_func, y, *dims, keep_coords="drop")

    # attach frequency coordinate
    for c, d, dx in zip(coords, dims, dxs):
        size = kwargs.pop(d, None)
        if size is None:
            size = len(y[d])
        freq = freq_func(size, dx)
        ds[c] = (d,), freq
    return ds


def _inject_docs(func: Callable, description: str = None, nd: bool = False) -> None:
    """inject xr docs into fft docs

    Parameters
    ----------
    func : object
        The function to modify
    description : object
    nd : object
        Whether the function acts on n-dimentional arrays
    """
    func_name = func.__name__
    doc = DocParser(fun=getattr(fft_, func_name))

    if not nd:
        doc.replace_params(
            a=CDParam("a", "The data to transform.", "xarray object"),
            axis=CDParam(
                "coord",
                "The axis along which the transform is applied.\n    The coordinate must be evenly spaced.",
                "string",
            ),
        )
    else:
        doc.replace_params(
            a=CDParam("a", "Object which the transform is applied.", "xarray object"),
            axes=CDParam(
                "coord",
                "The axis along which the transform is applied.\n    The coordinate must be evenly spaced.",
                "string",
            ),
            s=CDParam("s", "the shape of the result.", "mapping from coords to size", optional=True),
        )

    doc.reorder_params("a", "coord")
    doc.remove_sections("Notes", "Examples")
    doc.replace_strings_returns(("ndarray", "xarray object"), ("axes", "coords"), ("axis", "coord"))

    doc.insert_description(description)
    doc.insert_see_also(f"numpy.fft.{func_name} : numpy.fft.{func_name} : Original numpy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


fft = partial(_wrap1d, fft_.fft, fft_.fftfreq)
_inject_docs(fft, description="fft(a, coord, n=None, norm=None)")

ifft = partial(_wrap1d, fft_.ifft, fft_.fftfreq)
_inject_docs(ifft, description="ifft(a, coord, n=None, norm=None)")

rfft = partial(_wrap1d, fft_.rfft, fft_.rfftfreq)
_inject_docs(rfft, description="rfft(a, coord, n=None, norm=None)")

irfft = partial(_wrap1d, fft_.irfft, fft_.rfftfreq)
_inject_docs(irfft, description="irfft(a, coord, n=None, norm=None)")

fftn = partial(_wrapnd, fft_.fftn, fft_.fftfreq)
_inject_docs(fftn, nd=True, description="fftn(a, *coords, shape=None, norm=None)")

ifftn = partial(_wrapnd, fft_.ifftn, fft_.fftfreq)
_inject_docs(ifftn, nd=True, description="ifftn(a, *coords, shape=None, norm=None)")

rfftn = partial(_wrapnd, fft_.rfftn, fft_.rfftfreq)
_inject_docs(rfftn, nd=True, description="rfftn(a, *coords, shape=None, norm=None)")

irfftn = partial(_wrapnd, fft_.irfftn, fft_.rfftfreq)
_inject_docs(irfftn, nd=True, description="irfftn(a, *coords, shape=None, norm=None)")

hfft = partial(_wrap1d, fft_.hfft, fft_.rfftfreq)
_inject_docs(hfft, description="hfft(a, coord, n=None, norm=None)")

ihfft = partial(_wrap1d, fft_.ihfft, fft_.rfftfreq)
_inject_docs(ihfft, description="ihfft(a, coord, n=None, norm=None)")
