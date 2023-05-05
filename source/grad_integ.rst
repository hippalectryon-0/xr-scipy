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
