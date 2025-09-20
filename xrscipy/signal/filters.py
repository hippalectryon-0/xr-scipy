r""".. _filters:

Digital filters
---------------

.. ipython:: python
   :suppress:

    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import xrscipy.signal as dsp
    import xrscipy.signal.extra as dsp_extra


``xr-scipy`` wraps some of SciPy functions for constructing frequency filters using functions such as :py:func:`scipy.signal.firwin` and  :py:func:`scipy.signal.iirfilter`. Wrappers for convenient functions such as :py:func:`scipy.signal.decimate` and :py:func:`scipy.signal.savgol_filter` are also provided.
For convenience, the ``xrscipy.signal`` namespace will be imported under the alias ``dsp``:

.. ipython:: python

    import xrscipy.signal as dsp
    import xrscipy.signal.extra as dsp_extra


Frequency filters
^^^^^^^^^^^^^^^^^

The main wrapper for frequency filters is the :py:func:`~xrscipy.signal.extra.frequency_filter` wrapper. Its many arguments enable one to specify the type of filter, e.g. frequency band, FIR or IIR, response type family, filter order, forward-backward filtering, etc. By default a Butterworth IIR 4-th order filter with second-order-series (numerically stable even for high orders) forward-backward application (zero phase shift, but double order) is used, because such a filter typically offers a good performance for most time-series analysis applications.

Convenience functions such as :py:func:`~xrscipy.signal.extra.lowpass`, :py:func:`~xrscipy.signal.extra.highpass` and :py:func:`~xrscipy.signal.extra.bandpass` are provided which wrap :py:func:`~xrscipy.signal.extra.frequency_filter` with a predefined response type and offer a more convenient interface for the cutoff frequency specification.

The cutoff frequency is specified in the inverse usint to the filtered dimension's coordinates (typically time). The wrapper automatically checks the sampling of those coordinates and normalizes the supplied frequency by the Nyquist frequency.

In the following example a simple low-pass filter is applied to a noisy signal. Because the cutoff frequency is close to 0 (relative to the Nyquist frequency) a high filter order is used for sufficient noise attenuation.


.. ipython:: python
    :okwarning:

    t = np.linspace(0, 1, 1000)  # seconds
    sig = xr.DataArray(np.sin(16*t) + np.random.normal(0, 0.1, t.size),
                       coords=[('time', t)], name='signal')
    sig.plot(label='noisy')
    low = dsp_extra.lowpass(sig, 20, order=8)  # cutoff at 20 Hz
    low.plot(label='lowpass', linewidth=5)
    plt.legend()
    @savefig freq_filters.png width=4in
    plt.show()


Decimation
^^^^^^^^^^


To demonstrate basic functionality of :py:func:`~xrscipy.signal.decimate`, let's create a simple example DataArray:

.. ipython:: python

    arr = xr.DataArray(np.sin(np.linspace(0, 6.28, 300)) ** 2,
                       dims=('x'), coords={'x': np.linspace(0, 5, 300)})
    arr

Our :py:func:`~xrscipy.signal.decimate` takes an xarray object
(possibly high dimensional) and a dimension name (if not 1D)
along which the signal should be decimated. Decimation means

1. Apply a lowpass filter to remove frequencies above the new Nyquist frequency in order to prevent aliasing
2. Take every `q`-th point

.. ipython:: python

    arr_decimated = dsp.decimate(arr, q=40)
    arr_decimated

An alternative parameter to ``q`` is ``target_fs`` which is the new target sampling frequency to obtain, ``q = np.rint(current_fs / target_fs)``.

The return type is also a DataArray with coordinates.

.. ipython:: python
    :okwarning:

    arr.plot(label='arr', color='r')
    arr_decimated.plot.line('s--', label='decimated', color='b')
    plt.legend()
    @savefig decimated_signal.png width=4in
    plt.show()

The other keyword arguments are passed on to :py:func:`~xrscipy.signal.extra.lowpass`.


Savitzky-Golay LSQ filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Savitzky-Golay filter as a special type of a FIR filter which is equivalent to replacing filtered values by least-square fits of polynomials (or their derivatives) of a given order within a rolling window. For details see `their Wikipedia page`_ Such a filter is very useful when temporal or spatial features in the signal are of greater interest than frequency or wavenumber bands, respectively.

.. _`their Wikipedia page`: https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

To demonstrate basic functionality of :py:func:`~xrscipy.signal.savgol_filter`, let's create a simple example DataArray of the quadratic shape and add some noise:

.. ipython:: python

    arr = xr.DataArray(np.linspace(0, 5, 50) ** 2,
                       dims=('x'), coords={'x': np.linspace(0, 5, 50)})
    noise = np.random.normal(0,3,50)
    arr_noisy = arr + noise
    arr

Our :py:func:`~xrscipy.signal.savgol_filter` takes an xarray object
(possibly high dimensional) and a dimension name (if not 1D)
along which the signal should be filtered.
The window length is given in the units of the dimension coordinates.

.. ipython:: python

    arr_savgol2 = dsp.savgol_filter(arr_noisy, 2, 2)
    arr_savgol5 = dsp.savgol_filter(arr_noisy, 5, 2)
    arr_savgol2
    arr_savgol5

The return type is also a DataArray with coordinates.

.. ipython:: python
    :okwarning:

    arr.plot(label='arr', color='r')
    arr_noisy.plot.line('s', label='nosiy and decimated', color='b')
    arr_savgol2.plot(label='quadratic fit on 2 units of x', color='k', linewidth=2)
    arr_savgol5.plot.line('--',label='quadratic fit on 5 units of x', linewidth=2, color='lime')
    plt.legend()
    @savefig savgol_signal.png width=4in
    plt.show()

The other options (polynomial and derivative order) are the same as for :py:func:`scipy.signal.savgol_filter`, see :py:func:`~xrscipy.signal.savgol_filter` for details.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import savgol_filter as sp_savgol_filter

from xrscipy.signal.utils import get_maybe_only_dim, get_sampling_step


def savgol_filter(
    darray: xr.DataArray,
    window_length: float,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    dim: str = None,
    mode: str = "interp",
    cval: float = 0.0,
) -> xr.DataArray:
    """
    Apply a Savitzky-Golay filter to an array.

    This is a 1-D filter. If `darray` has dimension greater than 1, `dim`
    determines the axis along which the filter is applied.

    Parameters
    ----------
    darray : xarray.DataArray
        The data to be filtered.
    window_length : float
        The length of the filter window in the units of the specified dimension.
        This will be converted to the number of samples based on the coordinate spacing.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    dim : str, optional
        The dimension of the array `darray` along which the filter is to be applied.
        Default is the only dimension if 1D, otherwise must be specified.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This
        determines the type of extension to use for the padded signal to
        which the filter is applied.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.

    Returns
    -------
    y : xarray.DataArray
        The filtered data with the same coordinates as the input.
    """
    dim = get_maybe_only_dim(darray, dim)

    # Convert window_length from coordinate units to samples
    dt = get_sampling_step(darray, dim)
    window_length_samples = int(np.rint(window_length / dt))

    # Ensure window_length is odd and positive
    if window_length_samples % 2 == 0:
        window_length_samples += 1
    window_length_samples = max(window_length_samples, 3)

    # Ensure polyorder is less than window_length
    polyorder = min(polyorder, window_length_samples - 1)

    # Apply the filter
    result = xr.apply_ufunc(
        sp_savgol_filter,
        darray,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=dict(
            window_length=window_length_samples,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            axis=-1,
            mode=mode,
            cval=cval,
        ),
        exclude_dims={dim},
    )

    # Reorder dimensions to match input order
    result = result.transpose(*darray.dims)

    result.name = f"savgol_filtered_{darray.name}" if darray.name else "savgol_filtered"
    return result
