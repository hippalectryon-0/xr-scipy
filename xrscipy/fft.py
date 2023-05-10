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


def _wrap(func: Callable, freq_func: Callable, x: _DAS, *coords: str, **kwargs) -> _DAS:
    """Wrap function for fft

    Parameters
    ----------
    func: Callable
        the function to wrap
    freq_func: Callable
        the function that yields the frequencies associated with the output
    x: Dataset or DataArray
        the input object
    coords : list[str]
        the coords along which to do the transform. Each coord must map to a single dimension.
    _nd: bool
        If the function is 1D or n-dimensional
    _fftfreq: Callable
        Alternative fftfreq function to use in some special cases
    """
    nd = kwargs.pop("_nd", False)
    axes_arg = "axes" if nd else "axis"
    size_arg = "s" if nd else "n"
    errors.raise_invalid_args(["overwrite_x", "workers", "plan", axes_arg], kwargs)

    sizes = kwargs.pop(size_arg, None)
    _fftfreq = kwargs.pop("_fftfreq", freq_func)

    for c in coords:
        errors.raise_not_sorted(x[c])
    if nd and (sizes is not None and not isinstance(sizes, dict)):
        raise TypeError(f"s should be a dict mapping from coord name to size. Given {sizes}.")

    # coord to dim
    coords_da = [x[c] for c in coords]
    for c in coords_da:
        if len(c.dims) > 1:
            raise ValueError(f"coord {c.name} corresponds to more than one dimension: {c.dims}")
    dims = [x.dims[0] for x in coords_da]

    if nd:
        sizes = {d: len(x[d]) if sizes is None or c not in sizes else sizes[c] for d, c in zip(dims, coords)}

    input_core_dims = [d for d in dims if d in x.dims]
    ds = xr.apply_ufunc(
        func,
        x,
        input_core_dims=[input_core_dims],
        output_core_dims=[input_core_dims],
        kwargs={
            **kwargs,
            size_arg: ([sizes[d] for d in input_core_dims] if sizes else None) if nd else sizes,
            axes_arg: np.arange(-len(input_core_dims), 0) if nd else -1,
        },
        exclude_dims={*input_core_dims},
    )

    # attach frequency coordinate  # TODO merge those two ugly if branches
    if nd:
        dxs = [get_1D_spacing(x) for x in coords_da]
        for coord, dim, dx in zip(coords, dims, dxs):
            size = sizes[dim]
            freq = (
                freq_func((size - 1) * 2 if func.__name__ == "irfftn" and dim == dims[-1] else size, dx)
                if dim == dims[-1]
                else _fftfreq(size, dx)
            )  # /!\ in n dims, rfftn does fft over all axis except last one where it does rfft + irfftn changes output dims n -> ~2*n
            ds[coord] = (dim,), freq
    else:
        coord, dim, coord_da = coords[0], dims[0], coords_da[0]
        size = kwargs.get("n", None) or len(ds[dim])
        freq = freq_func(size, get_1D_spacing(coord_da))
        if freq.size != ds[dim].size:  # functions such as rfft, hfft, irfft, ihfft in scipy.fft modify the output shape
            size = (len(ds[dim]) - 1) * 2
            freq = freq_func(size, get_1D_spacing(coord_da))
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


fft = partial(_wrap, sp_fft.fft, sp_fft.fftfreq)
_inject_docs(fft, description="fft(a, coord, n=None, norm=None)")

ifft = partial(_wrap, sp_fft.ifft, sp_fft.fftfreq)
_inject_docs(ifft, description="ifft(a, coord, n=None, norm=None)")

rfft = partial(_wrap, sp_fft.rfft, sp_fft.rfftfreq)
_inject_docs(rfft, description="rfft(a, coord, n=None, norm=None)")

irfft = partial(_wrap, sp_fft.irfft, sp_fft.rfftfreq)
_inject_docs(irfft, description="irfft(a, coord, n=None, norm=None)")

fftn = partial(_wrap, sp_fft.fftn, sp_fft.fftfreq, _nd=True)
_inject_docs(fftn, nd=True, description="fftn(a, *coords, s=None, norm=None)")

ifftn = partial(_wrap, sp_fft.ifftn, sp_fft.fftfreq, _nd=True)
_inject_docs(ifftn, nd=True, description="ifftn(a, *coords, s=None, norm=None)")

rfftn = partial(_wrap, sp_fft.rfftn, sp_fft.rfftfreq, _nd=True, _fftfreq=sp_fft.fftfreq)
_inject_docs(rfftn, nd=True, description="rfftn(a, *coords, s=None, norm=None)")

irfftn = partial(_wrap, sp_fft.irfftn, sp_fft.rfftfreq, _nd=True, _fftfreq=sp_fft.fftfreq)
_inject_docs(irfftn, nd=True, description="irfftn(a, *coords, s=None, norm=None)")

hfft = partial(_wrap, sp_fft.hfft, sp_fft.rfftfreq)
_inject_docs(hfft, description="hfft(a, coord, n=None, norm=None)")

ihfft = partial(_wrap, sp_fft.ihfft, sp_fft.rfftfreq)
_inject_docs(ihfft, description="ihfft(a, coord, n=None, norm=None)")
