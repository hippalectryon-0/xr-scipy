r"""mirrors scipy.fftpack"""
from typing import Callable

import numpy as np
import xarray as xr
from scipy import fftpack

from xrscipy import errors
from xrscipy.docs import CDParam, DocParser
from xrscipy.utils import _DAS, get_1D_spacing, partial


def _wrap1d(func: Callable, freq_func: Callable, x: _DAS, coord: str, **kwargs) -> _DAS:
    """Wrap function for fft1d

    Parameters
    ----------
    coord : str
        the coord along which to do the transform. Must map to a single dimension.
    """
    errors.raise_invalid_args(["axis", "overwrite_x", "workers", "plan"], kwargs)
    errors.raise_not_sorted(x[coord])

    # In case of dim is a non-dimensional coordinate.
    coord_da = x[coord]
    if len(coord_da.dims) > 1:
        raise ValueError(f"coord {coord} coresponds to more than one dimension: {coord_da.dims}")
    dim = coord_da.dims[0]

    kwargs["axis"] = -1
    ds = xr.apply_ufunc(
        func,
        x,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=kwargs,
        exclude_dims={dim},
    )

    # attach frequency coordinate
    size = kwargs.get("n", None) or len(ds[dim])
    freq = freq_func(size, get_1D_spacing(coord_da))
    if freq.size != ds[dim].size:  # functions such as rfft, hfft, irfft, ihfft in scipy.fft modify the output shape
        size = (len(ds[dim]) - 1) * 2
        freq = freq_func(size, get_1D_spacing(coord_da))
    ds[coord] = (dim,), freq
    return ds


def _wrapnd(func: Callable, freq_func: Callable, x: _DAS, *coords: str, **kwargs) -> _DAS:
    """Wrap function for fftnd

    Parameters
    ----------
    coords : list[str]
        the coords along which to do the transform. Each coord must map to a single dimension."""
    errors.raise_invalid_args(["axes", "overwrite_x", "workers", "plan"], kwargs)
    sizes = kwargs.pop("s", None)
    _fftfreq = kwargs.pop("_fftfreq", freq_func)

    if sizes is not None and not isinstance(sizes, dict):
        raise TypeError(f"shape should be a dict mapping from coord name to size. Given {sizes}.")

    for c in coords:
        errors.raise_not_sorted(x[c])

    # In case of dim is a non-dimensional coordinate.
    coords_data = [x[c] for c in coords]
    for c in coords_data:
        if len(c.dims) > 1:
            raise ValueError(f"coord {c.name} coresponds to more than one dimension: {c.dims}")
    dims = [x.dims[0] for x in coords_data]
    sizes = {d: len(x[d]) if sizes is None or c not in sizes else sizes[c] for d, c in zip(dims, coords)}
    dxs = [get_1D_spacing(x) for x in coords_data]

    input_core_dims = [d for d in dims if d in x.dims]
    ds = xr.apply_ufunc(
        func,
        x,
        input_core_dims=[input_core_dims],
        output_core_dims=[input_core_dims],
        kwargs={
            **kwargs,
            "s": [sizes[d] for d in input_core_dims] if sizes else None,
            "axes": np.arange(-len(input_core_dims), 0) if len(input_core_dims) else None,
        },
        exclude_dims={*input_core_dims},
    )

    # attach frequency coordinate
    for coord, dim, dx in zip(coords, dims, dxs):
        size = sizes[dim]
        freq = (
            freq_func((size - 1) * 2 if func.__name__ == "irfftn" and dim == dims[-1] else size, dx)
            if dim == dims[-1]
            else _fftfreq(size, dx)
        )  # /!\ in n dims, rfftn does fft over all axis except last one where it does rfft + irfftn changes output dims n -> ~2*n
        ds[coord] = (dim,), freq
    return ds


def _wrapfftpack(func: Callable, freq_func: Callable, x: _DAS, *coords: str, **kwargs) -> _DAS:
    """wrapper around _wrapnd that changes s<->shape"""
    kwargs["s"] = kwargs.pop("shape", None)

    # noinspection PyMissingOrEmptyDocstring
    def s_to_shape(xx: _DAS, **kwargs) -> _DAS:
        kwargs["shape"] = kwargs.pop("s", None)
        return func(xx, **kwargs)

    return _wrapnd(s_to_shape, freq_func, x, *coords, **kwargs)


def _inject_docs(func: Callable, description: str = None, nd: bool = False) -> None:
    """inject xr docs into fftpack docs

    Parameters
    ----------
    func : callable
        The function to modify
    description : str
    nd : bool
        Whether the function acts on n-dimentional arrays
    """
    func_name = func.__name__
    doc = DocParser(fun=getattr(fftpack, func_name))

    if not nd:
        doc.replace_params(
            x=CDParam("obj", doc.get_parameter("x").description, "xarray object"),
            axis=CDParam(
                "coord",
                f"{doc.get_parameter('axis').description.split(';')[0].replace('Axis', 'Coordinate')}. The coordinate must be evenly spaced.",
                "string",
            ),
        )
    else:
        doc.replace_params(
            x=CDParam("a", "Object on which the transform is applied.", "xarray object"),
            axes=CDParam(
                "coord",
                "The axis along which the transform is applied. The coordinate must be evenly spaced.",
                "string",
            ),
            s=CDParam("s", "the shape of the result.", "mapping from coords to size", optional=True),
        )

    doc.remove_params("overwrite_x")
    doc.reorder_params("obj", "coord")
    doc.remove_sections("Notes", "Examples")
    doc.replace_strings_returns(("ndarray", "xarray object"))

    doc.insert_description(description)
    doc.insert_see_also(f"numpy.fftpack.{func_name} : scipy.fft.{func_name} : Original scipy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


fft = partial(_wrap1d, fftpack.fft, fftpack.fftfreq)
_inject_docs(fft, description="fft(obj, coord, n=None)")

ifft = partial(_wrap1d, fftpack.ifft, fftpack.fftfreq)
_inject_docs(ifft, description="ifft(obj, coord, n=None)")

rfft = partial(_wrap1d, fftpack.rfft, fftpack.rfftfreq)
_inject_docs(rfft, description="rfft(obj, coord, n=None)")

irfft = partial(_wrap1d, fftpack.irfft, fftpack.rfftfreq)
_inject_docs(irfft, description="irfft(obj, coord, n=None)")

dct = partial(_wrap1d, fftpack.dct, fftpack.rfftfreq)
_inject_docs(dct, description="dct(obj, coord, type=2, n=None, norm=None)")

dst = partial(_wrap1d, fftpack.dst, fftpack.rfftfreq)
_inject_docs(dst, description="dst(obj, coord, type=2, n=None, norm=None)")

idct = partial(_wrap1d, fftpack.idct, fftpack.rfftfreq)
_inject_docs(idct, description="idct(obj, coord, type=2, n=None, norm=None)")

idst = partial(_wrap1d, fftpack.idst, fftpack.rfftfreq)
_inject_docs(idst, description="idst(obj, coord, type=2, n=None, norm=None)")

fftn = partial(_wrapfftpack, fftpack.fftn, fftpack.fftfreq)
_inject_docs(fftn, nd=True, description="fftn(obj, *coords, s=None, axes=None)")

ifftn = partial(_wrapfftpack, fftpack.ifftn, fftpack.fftfreq)
_inject_docs(ifftn, nd=True, description="ifftn(obj, *coords, s=None, axes=None)")
