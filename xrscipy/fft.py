r""".. _fft:

Fourier Transform
-----------------

.. ipython:: python
   :suppress:

    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import xrscipy
    np.random.seed(123456)

xr-scipy wraps ``numpy.fft``, for more convenient data analysis with
xarray.
Let us consider an example DataArray

.. ipython:: python

    arr = xr.DataArray(np.sin(np.linspace(0, 15.7, 30)) ** 2,
                       dims=('x'), coords={'x': np.linspace(0, 5, 30)})
    arr

Our :py:func:`~xrscipy.fft.fft` takes an xarray object
(possibly high dimensional) and a coordinate name which direction we compute
the Fourier transform.

.. ipython:: python

    fft = xrscipy.fft.fft(arr, 'x')
    fft

The coordinate `x` is also converted to frequency.

.. ipython:: python
    :okwarning:

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    arr.plot()
    plt.subplot(1, 2, 2)
    np.abs(fft).plot()
    @savefig fft.png width=4in
    plt.show()


.. note::

  The coordinate values must be evenly spaced for FFT.


Multidimensional Fourier transform
----------------------------------

xr-scipy also wraps the multidimensional Fourier transform,
such as :py:func:`~xrscipy.fft.rfftn`

Their usage is very similar to the above, where we just need to specify
coordinates.

.. ipython:: python

    arr = xr.DataArray(np.random.randn(30, 20) ** 2,
                       dims=('x', 'y'),
                       coords={'x': np.linspace(0, 5, 30),
                               'y': np.linspace(0, 5, 20)})
    fftn = xrscipy.fft.fftn(arr, 'x', 'y')
    fftn

.. ipython:: python
    :okwarning:

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    arr.plot()
    plt.subplot(1, 2, 2)
    np.abs(fftn.sortby('x').sortby('y')).plot()
    @savefig fftn.png width=4in
    plt.show()
"""
from typing import Callable

import numpy as np
import xarray as xr
from scipy import fft as sp_fft

import xrscipy.errors as errors
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


def _inject_docs(func: Callable, description: str = None, nd: bool = False) -> None:
    """inject xr docs into fft docs

    Parameters
    ----------
    func : callable
        The function to modify
    description : str
    nd : bool
        Whether the function acts on n-dimentional arrays
    """
    func_name = func.__name__
    doc = DocParser(fun=getattr(sp_fft, func_name))

    if not nd:
        doc.replace_params(
            x=CDParam("a", "The data to transform.", "xarray object"),
            axis=CDParam(
                "coord",
                "The axis along which the transform is applied. The coordinate must be evenly spaced.",
                "string",
            ),
        )
    else:
        doc.replace_params(
            x=CDParam("a", "Object which the transform is applied.", "xarray object"),
            axes=CDParam(
                "coord",
                "The axis along which the transform is applied. The coordinate must be evenly spaced.",
                "string",
            ),
            s=CDParam("s", "the shape of the result.", "mapping from coords to size", optional=True),
        )

    doc.reorder_params("a", "n", "coord")
    doc.remove_params("overwrite_x", "workers", "plan")
    doc.remove_sections("Notes", "Examples")
    doc.replace_strings_returns(("array_like", "xarray object"), ("axes", "coords"), ("axis", "coord"))

    doc.insert_description(description)
    doc.insert_see_also(f"scipy.fft.{func_name} : scipy.fft.{func_name} : Original scipy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


fft = partial(_wrap1d, sp_fft.fft, sp_fft.fftfreq)
_inject_docs(fft, description="fft(a, coord, n=None, norm=None)")

ifft = partial(_wrap1d, sp_fft.ifft, sp_fft.fftfreq)
_inject_docs(ifft, description="ifft(a, coord, n=None, norm=None)")

rfft = partial(_wrap1d, sp_fft.rfft, sp_fft.rfftfreq)
_inject_docs(rfft, description="rfft(a, coord, n=None, norm=None)")

irfft = partial(_wrap1d, sp_fft.irfft, sp_fft.rfftfreq)
_inject_docs(irfft, description="irfft(a, coord, n=None, norm=None)")

fftn = partial(_wrapnd, sp_fft.fftn, sp_fft.fftfreq)
_inject_docs(fftn, nd=True, description="fftn(a, *coords, shape=None, norm=None)")

ifftn = partial(_wrapnd, sp_fft.ifftn, sp_fft.fftfreq)
_inject_docs(ifftn, nd=True, description="ifftn(a, *coords, shape=None, norm=None)")

rfftn = partial(_wrapnd, sp_fft.rfftn, sp_fft.rfftfreq, _fftfreq=sp_fft.fftfreq)
_inject_docs(rfftn, nd=True, description="rfftn(a, *coords, shape=None, norm=None)")

irfftn = partial(_wrapnd, sp_fft.irfftn, sp_fft.rfftfreq, _fftfreq=sp_fft.fftfreq)
_inject_docs(irfftn, nd=True, description="irfftn(a, *coords, shape=None, norm=None)")

hfft = partial(_wrap1d, sp_fft.hfft, sp_fft.rfftfreq)
_inject_docs(hfft, description="hfft(a, coord, n=None, norm=None)")

ihfft = partial(_wrap1d, sp_fft.ihfft, sp_fft.rfftfreq)
_inject_docs(ihfft, description="ihfft(a, coord, n=None, norm=None)")
