.. _fft:

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

Our :py:func:`~xrscipy.fft.rfft` takes an xarray object
(possibly high dimensional) and a coordinate name which direction we compute
the Fourier transform.

.. ipython:: python

    rfft = xrscipy.fft.rfft(arr, 'x')
    rfft

The coordinate `x` is also converted to frequency.

.. ipython:: python

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    arr.plot()
    plt.subplot(1, 2, 2)
    np.abs(rfft).plot()
    @savefig rfft.png width=4in
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

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    arr.plot()
    plt.subplot(1, 2, 2)
    np.abs(fftn.sortby('x').sortby('y')).plot()
    @savefig fftn.png width=4in
    plt.show()
