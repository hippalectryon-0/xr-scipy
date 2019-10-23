.. _integrate:

Gradient and Integration
------------------------

.. ipython:: python
   :suppress:

    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import xrscipy
    np.random.seed(123456)


xr-scipy wraps ``scipy.gradient`` and some of ``scipy.integrate`` functions.
Let's create a simple example DataArray:

.. ipython:: python

    arr = xr.DataArray(np.sin(np.linspace(0, 6.28, 30)) ** 2,
                       dims=('x'), coords={'x': np.linspace(0, 5, 30)})
    arr

Our :py:func:`~xrscipy.gradient` takes an xarray object
(possibly high dimensional) and a coordinate name
which direction we compute the gradient of the array,

.. ipython:: python

    grad = xrscipy.gradient(arr, 'x')
    grad

The return type is also a DataArray with coordinates.

.. ipython:: python
    :okwarning:

    arr.plot(label='arr')
    grad.plot(label='gradient')
    plt.legend()
    @savefig grad.png width=4in
    plt.show()

The other options (edge_order) are the same to ``numpy.gradient``.
See :py:func:`~xrscipy.gradient`.

Similar to :py:func:`~xrscipy.gradient`, xr-scipy wraps some functions
in ``scipy.integrate`` module.
Our integration function also takes an xarray object and coordinate name
along which the array to be integrated.
The return type is also a DataArray,

.. ipython:: python
    :okwarning:

    # trapz computes definite integration
    xrscipy.integrate.trapz(arr, coord='x')

    # cumurative integration returns a same shaped array
    integ = xrscipy.integrate.cumtrapz(arr, 'x')
    integ

    arr.plot(label='arr')
    integ.plot(label='integration')
    plt.legend()
    @savefig cumtrapz.png width=4in
    plt.show()



See :py:func:`~xrscipy.integrate.trapz` for other options.


.. Note::

  There are slight difference from the original implementations.
  Our :py:func:`~xrscipy.gradient` does not accept multiple coordinates.
  Our :py:func:`~xrscipy.integrate.cumtrapz` always assume ``initial=0``.
