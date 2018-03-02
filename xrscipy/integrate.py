from __future__ import absolute_import, division, print_function
from functools import partial
from scipy import integrate
import xarray as xr
from . import errors
from . import utils


def _wrap(func, reduces, y, dim, **kwargs):
    """
    Wrap scipy functions for xarray data objects
    """
    errors.raise_invalid_args(['x', 'dx', 'axis'], kwargs)
    errors.raise_not_sorted(y[dim])

    # In case of dim is a non-dimensional coordinate.
    x = y[dim]
    dim = x.dims[0]
    output_core_dim = [] if reduces else [dim]

    kwargs['x'] = x
    kwargs['axis'] = -1

    def apply_func(v):
        # v: xr.Varaible
        result = xr.apply_ufunc(
            func, v, input_core_dims=[[dim]],
            output_core_dims=[output_core_dim], kwargs=kwargs)
        if not reduces:
            return result.transpose(*y.dims)
        else:
            return result

    return utils.wrap_dataset(apply_func, y, dim)


DOCS = """

    Integrate an xarray object along the given dimension using {0:s}.

    Parameters
    ----------
    y: xr.DataArray or xr.Dataset
        Input object to intergrate
    dim: string
        Name of dimension or coordinate of the object.
    {1:s}

    See also
    --------
    scipy.integrate.{2:s}
"""


trapz = partial(_wrap, integrate.trapz, True)
trapz.__doc__ = """
    trapz(y, dim):
""" + DOCS.format('the composite trapezoidal rule', '', 'trapz')

simps = partial(_wrap, integrate.simps, True)
simps.__doc__ = """
    simps(y, dim, even='avg'):
""" + DOCS.format(
    'the composite trapezoidal rule',
    'even: str{`avg`, `first`, `last`}, optional', 'simps')

romb = partial(_wrap, integrate.romb, True)
romb.__doc__ = """
    robs(y, dim, show=False):
""" + DOCS.format('Roberg integration', 'show: bool, optional', 'romb')

cumtrapz = partial(_wrap, integrate.cumtrapz, False, initial=0)
cumtrapz.__doc__ = """
    cumtrapz(y, dim):
"""
