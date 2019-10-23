.. _integrate:

Filters
------------------------

.. ipython:: python
   :suppress:

    import warnings
    import xarray
    import scipy.signal
    import numpy as np

    try:
        from scipy.signal import sosfiltfilt
    except ImportError:
        sosfiltfilt = None

    from .utils import get_maybe_last_dim_axis, get_sampling_step


xr-scipy wraps some of scipy filtering functions such as ``scipy.signal.decimate``, ``scipy.signal.savgol_filter`` etc.

To demonstrate basic functionality of ``dsp.decimate``, let's create a simple example DataArray:

.. ipython:: python

    arr = xarray.DataArray(np.sin(np.linspace(0, 6.28, 300)) ** 2,
                       dims=('x'), coords={'x': np.linspace(0, 5, 300)})
    arr

Our :py:func:`~xrscipy.decimate` takes an xarray object
(possibly high dimensional) and a coordinate name
which should be decimated,

.. ipython:: python

    arr_decimated = dsp.decimate(arr, 40)
    arr_decimated

The return type is also a DataArray with coordinates.

.. ipython:: python

    arr.plot(label='arr', color='r')
    arr_decimated.plot.line('s--', label='decimated', color='b')
    plt.legend()
    #@savefig decimated_signal.png width=4in
    plt.show()

The other options (lowpass_kwargs) are the same to ``numpy.decimate``.
See :py:func:`~xrscipy.decimate`.


To demonstrate basic functionality of ``dsp.savgol_filter``, let's create a simple example DataArray of the quadratic shape and add a noise there:

.. ipython:: python

    arr = xarray.DataArray((np.linspace(0, 5, 50)) ** 2,
                       dims=('x'), coords={'x': np.linspace(0, 5, 50)})
    noise = np.random.normal(0,3,50)
    arr_noisy = arr + noise
    arr

Our :py:func:`~xrscipy.savgol_filter` takes an xarray object
(possibly high dimensional) and a coordinate name
which should be filtered,

.. ipython:: python

    arr_savgol2 = dsp.savgol_filter(arr_noisy, 2, 2)
    arr_savgol5 = dsp.savgol_filter(arr_noisy, 5, 2)
    arr_savgol2
    arr_savgol5

The return type is also a DataArray with coordinates.

.. ipython:: python

    arr.plot(label='arr', color='r')
    arr_noisy.plot.line('s', label='nosiy and decimated', color='b')
    arr_savgol2.plot(label='quadratic fit on 2 units of x', color='k', linewidth=2)
    arr_savgol5.plot.line('--',label='quadratic fit on 5 units of x', linewidth=2, color='lime')
    plt.legend()
    #@savefig savgol_signal.png width=4in
    plt.show()

The other options are the same to ``numpy.savgol_filter``.
See :py:func:`~xrscipy.savgol_filter`.