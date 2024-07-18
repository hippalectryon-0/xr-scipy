r""".. _integrate:

Integration
------------------------

.. ipython:: python
   :suppress:

    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import xrscipy
    np.random.seed(123456)


xr-scipy wraps some of ``scipy.integrate`` functions.
Let's create a simple example DataArray:

.. ipython:: python

    arr = xr.DataArray(np.sin(np.linspace(0, 6.28, 30)) ** 2,
                       dims=('x'), coords={'x': np.linspace(0, 5, 30)})
    arr

Our integration function takes an xarray object and coordinate name
along which the array to be integrated.
The return type is also a DataArray,

.. ipython:: python
    :okwarning:

    # trapz computes definite integration
    xrscipy.integrate.trapezoid(arr, coord='x')

    # cumurative integration returns a same shaped array
    integ = xrscipy.integrate.cumulative_trapezoid(arr, 'x')
    integ

    arr.plot(label='arr')
    integ.plot(label='integration')
    plt.legend()
    @savefig cumulative_trapezoid.png width=4in
    plt.show()



See :py:func:`~xrscipy.integrate.trapezoid` for other options.


.. Note::

  There are slight difference from the original implementations.
  Our :py:func:`~xrscipy.integrate.cumulative_trapezoid` always assume ``initial=0``.
"""

from typing import Callable

import xarray as xr
from scipy import integrate

from xrscipy import errors
from xrscipy.docs import CDParam, DocParser
from xrscipy.utils import _DAS, partial


def _wrap_integrate(func: Callable, reduces: bool, y: _DAS, coord: str, **kwargs) -> _DAS:
    """
    Wrap scipy functions for xarray data objects

    Parameters
    ----------
    reduces : bool
        whether to return a single value
    coord : str
        the coord along which to do the transform. Must map to a single dimension.
    """
    errors.raise_not_sorted(y[coord])

    # In case of dim is a non-dimensional coordinate.
    coord_da = y[coord]
    if len(coord_da.dims) > 1:
        raise ValueError(f"coord {coord} coresponds to more than one dimension: {coord_da.dims}")
    dim = coord_da.dims[0]
    output_core_dim = [] if reduces else [dim]

    kwargs["axis"] = -1

    return xr.apply_ufunc(
        func,
        y,
        input_core_dims=[[dim]],
        output_core_dims=[output_core_dim],
        kwargs=kwargs,
    )


def _wrap_integrate_trapz(func: Callable, reduces: bool, y: _DAS, coord: str, **kwargs) -> _DAS:
    """
    adapted to trapz signature
    """
    errors.raise_invalid_args(["x", "dx", "axis"], kwargs)
    kwargs["x"] = y[coord]
    return _wrap_integrate(func, reduces, y, coord, **kwargs)


def _inject_docs(func: Callable, description: str = None) -> None:
    """inject xr docs into integrate docs

    Parameters
    ----------
    func : callable
        The function to modify
    description : str
    """
    func_name = func.__name__
    doc = DocParser(fun=getattr(integrate, func_name))

    if y_doc := doc.get_parameter("y"):
        doc.replace_params(y=CDParam("obj", y_doc.description, "xarray object"))
    doc.replace_params(axis=CDParam("coord", "The coordinate along which to integrate.", "string"))

    doc.remove_params("dx", "x")
    doc.remove_sections("Notes", "Examples")
    doc.replace_strings_returns(("ndarray", "xarray object"), ("axis", "coordinate"))

    doc.insert_description(description)
    doc.replace_strings_description(("axis", "coordinate"))

    doc.insert_see_also(f"scipy.integrate.{func_name}: scipy.integrate.{func_name} : Original scipy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


romb = partial(_wrap_integrate, integrate.romb, True)
_inject_docs(romb, description="romb(obj, coord, show=False)")

trapezoid = partial(_wrap_integrate_trapz, integrate.trapezoid, True)
_inject_docs(trapezoid, description="trapezoid(obj, coord)")

cumulative_trapezoid = partial(_wrap_integrate_trapz, integrate.cumulative_trapezoid, False, initial=0)
_inject_docs(
    cumulative_trapezoid,
    description="cumulative_trapezoid(obj, coord)",
)

simpson = partial(_wrap_integrate_trapz, integrate.simpson, True)
_inject_docs(simpson, description="simpson(obj, coord, even='avg')")
