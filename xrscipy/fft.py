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
    """
    nd = kwargs.pop("_nd", False)
    axes_arg = "axes" if nd else "axis"
    size_arg = "s" if nd else "n"
    errors.raise_invalid_args(["overwrite_x", "workers", "plan", axes_arg], kwargs)

    sizes = kwargs.pop(size_arg, None)

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

    # attach frequency coordinate
    dxs = [get_1D_spacing(x) for x in coords_da]
    for coord, dim, dx in zip(coords, dims, dxs):
        size = sizes[dim] if nd else (kwargs.get("n", None) or len(ds[dim]))
        freq = freq_func(size, dx)
        if (
            freq.size != ds[dim].size
        ):  # functions such as rfft, hfft, irfft, ihfft in scipy.fft modify the output shape # /!\ in n dims, rfftn does fft over all axis except last one where it does rfft + irfftn changes output dims n -> ~2*n
            size = (len(ds[dim]) - 1) * 2
            freq = freq_func(size, dx)
        ds[coord] = (dim,), freq
    return ds


def _inject_docs(func: Callable, description: str = None, _nd: bool = False) -> None:
    """inject xr docs into fft docs

    Parameters
    ----------
    func : callable
        The function to modify
    description : str
    _nd : bool
        Whether the function acts on n-dimentional arrays
    """
    func_name = func.__name__
    doc = DocParser(fun=getattr(sp_fft, func_name))

    doc.replace_params(
        x=CDParam("x", "The data to transform.", "xarray object"),
        axis=CDParam(
            "coord",
            "The axis along which the transform is applied. The coordinate must be evenly spaced.",
            "string",
        ),
    )

    if _nd:
        doc.replace_params(
            s=CDParam("s", "the shape of the result.", "mapping from coords to size", optional=True),
        )

    doc.reorder_params("x", "coord", "n", "s")
    doc.remove_params("overwrite_x", "workers", "plan")
    doc.remove_sections("Notes", "Examples")
    doc.replace_strings_returns(("array_like", "xarray object"), ("axes", "coords"), ("axis", "coord"))

    doc.insert_description(description)
    doc.insert_see_also(f"scipy.fft.{func_name} : scipy.fft.{func_name} : Original scipy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


def _partial_and_doc(f_orig: Callable, *args, description: str = "(x, coord, n=None, norm=None)", **kwargs) -> Callable:
    """apply partial and docs"""
    f = partial(_wrap, f_orig, *args, **kwargs)

    _inject_docs(f, description=f"{f_orig.__name__}{description}", _nd=kwargs.get("_nd"))
    return f


fft = _partial_and_doc(sp_fft.fft, sp_fft.fftfreq)
ifft = _partial_and_doc(sp_fft.ifft, sp_fft.fftfreq)
rfft = _partial_and_doc(sp_fft.rfft, sp_fft.rfftfreq)
irfft = _partial_and_doc(sp_fft.irfft, sp_fft.rfftfreq)
fftn = _partial_and_doc(sp_fft.fftn, sp_fft.fftfreq, _nd=True)
ifftn = _partial_and_doc(sp_fft.ifftn, sp_fft.fftfreq, _nd=True)
rfftn = _partial_and_doc(sp_fft.rfftn, sp_fft.rfftfreq, _nd=True)
irfftn = _partial_and_doc(sp_fft.irfftn, sp_fft.rfftfreq, _nd=True)
hfft = _partial_and_doc(sp_fft.hfft, sp_fft.rfftfreq)
ihfft = _partial_and_doc(sp_fft.ihfft, sp_fft.rfftfreq)
