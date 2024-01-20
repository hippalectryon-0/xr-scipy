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


``xr-scipy`` wraps some of SciPy functions for constructing frequency filters using functions such as :py:func:`scipy.signal.firwin` and  :py:func:`scipy.signal.iirfilter`. Wrappers for convenient functions such as  :py:func:`scipy.signal.decimate` and :py:func:`scipy.signal.savgol_filter` are also provided.
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

import warnings
from typing import Any, TypeVar

import numpy as np
import scipy.signal
import xarray as xr
from numpy import ndarray

# noinspection PyProtectedMember
from numpy._typing import ArrayLike
from scipy.signal import sosfiltfilt

from xrscipy.signal.utils import DecimationWarning, FilteringNaNWarning, get_maybe_only_dim, get_sampling_step


def _firwin_ba(*args, **kwargs) -> tuple[np.ndarray, ndarray]:
    if not kwargs.get("pass_zero"):
        args = (args[0] + 1,) + args[1:]  # numtaps must be odd
    return scipy.signal.firwin(*args, **kwargs), np.array([1])


_BA_FUNCS = {
    "iir": scipy.signal.iirfilter,
    "fir": _firwin_ba,
}

_ORDER_DEFAULTS = {
    "iir": 4,
    "fir": 29,
}


def frequency_filter(
    darray: xr.DataArray,
    f_crit: ArrayLike,
    order: int = None,
    irtype: str = "iir",
    filtfilt: bool = True,
    apply_kwargs: dict = None,
    in_nyq: bool = False,
    dim: str = None,
    **kwargs,
) -> xr.DataArray:
    """Applies given frequency filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by `dim`. If `darray` is multi dimensional then axis along which the filter
    is applied has to be specified by `dim` string.

    The type of the filter is chosen by `irtype` and then `filtfilt` states is
    the filter is applied both ways, forward and backward. Additional parameters
    passed to filter function specified by `apply_kwargs`.

    If 'iir' is chosen as `irtype`, then if `filtfilt` is True then the filter
    scipy.signal.filtfilt is used, if False scipy.signal.lfilter applies.

    If 'fir' is chosen as `irtype`, then if `filtfilt` is True then the filter
    scipy.signal.sosfiltfilt is used, if False scipy.signal.sosfilt applies.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_crit : array_like
        A scalar or length-2 sequence giving the critical frequencies.
    order : int, optional
        The order of the filter. If Default then it takes order defaults
        from `_ORDER_DEFAULTS`, which is `irtype` specific.
        Default is None.
    irtype : string, optional
        A string specifying the impulse response of the filter, has to be
        either "fir" then finite impulse response (FIR) is used, or "iir"
        then infinite impulse response (IIR) filter is applied. ValueError
        is raised otherwise.
        Default is "iir".
    filtfilt: bool, optional
        When True the filter is applied both forwards and backwards, otherwise
        only one way, from left to right, is applied.
        Default is True.
    apply_kwargs : dict, optional
        Specifies kwargs, which are passed to the filter function given by
        `irtype` and `filtfilt`.
        Default is None.
    in_nyq : bool, optional
        If True, then the critical frequencies given by `f_crit` are normalized
        by Nyquist frequency.
        Default is False.
    dim : string, optional
        A string specifing the dimension along which the filter is applied.
        If `darray` is 1-d then the dimension is found if not specified by `dim`.
        For multi dimensional `darray` has to be specified, otherwise ValueError
        is raised.
        Default is None.
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed,
        either to scipy.signal.iirfilter if "iir" method for `irtype` is chosen,
        or scipy.signal.firwin.
    """
    if irtype not in _BA_FUNCS:
        raise ValueError(f"Wrong argument for irtype: {irtype}, must be one of {_BA_FUNCS.keys()}")
    if order is None:
        order = _ORDER_DEFAULTS[irtype]
    if apply_kwargs is None:
        apply_kwargs = {}
    dim = get_maybe_only_dim(darray, dim)
    f_crit_norm = np.asarray(f_crit, dtype=float)
    if not in_nyq:  # normalize by Nyquist frequency
        f_crit_norm *= 2 * get_sampling_step(darray, dim)
    if np.any(np.isnan(np.asarray(darray))):  # only warn since simple forward-filter or FIR is valid
        warnings.warn(
            "data contains NaNs, filter will propagate them",
            FilteringNaNWarning,
            stacklevel=2,
        )
    if sosfiltfilt and irtype == "iir":  # TODO merge with other if branch
        sos = scipy.signal.iirfilter(order, f_crit_norm, output="sos", **kwargs)
        if filtfilt:
            ret = xr.apply_ufunc(
                sosfiltfilt,
                sos,
                darray,
                input_core_dims=[[], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
        else:
            ret = xr.apply_ufunc(
                scipy.signal.sosfilt,
                sos,
                darray,
                input_core_dims=[[], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
    else:
        b, a = _BA_FUNCS[irtype](order, f_crit_norm, **kwargs)
        if filtfilt:
            ret = xr.apply_ufunc(
                scipy.signal.filtfilt,
                b,
                a,
                darray,
                input_core_dims=[[], [], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
        else:
            ret = xr.apply_ufunc(
                scipy.signal.lfilter,
                b,
                a,
                darray,
                input_core_dims=[[], [], [dim]],
                output_core_dims=[[dim]],
                kwargs=apply_kwargs,
            )
    return ret


_D = TypeVar("_D", bound=dict)


def _update_ftype_kwargs(kwargs: _D, iirvalue: Any, firvalue: Any) -> _D:
    if kwargs.get("irtype", "iir") == "iir":
        kwargs.setdefault("btype", iirvalue)
    else:  # fir
        kwargs.setdefault("pass_zero", firvalue)
    return kwargs


def lowpass(darray: xr.DataArray, f_cutoff: ArrayLike, *args, **kwargs) -> xr.DataArray:
    """Applies lowpass filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by an arg `dim`. If `darray` is multi dimensional then axis along which
    the filter is applied has to be specified by an additional argument `dim`
    string.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_cutoff : array_like
        A scalar specifying the cut-off frequency for the lowpass filter.
    args :
        Additional arguments passed to frequency_filter function to further
        specify the filter. The following parameters can be passed:
        (order, irtype, filtfilt, apply_kwargs, in_nyq, dim)
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed.
        See frequency_filter documentation for furhter information.
    """
    kwargs = _update_ftype_kwargs(kwargs, "lowpass", True)
    return frequency_filter(darray, f_cutoff, *args, **kwargs)


def highpass(darray: xr.DataArray, f_cutoff: ArrayLike, *args, **kwargs) -> xr.DataArray:
    """Applies highpass filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by an arg `dim`. If `darray` is multi dimensional then axis along which
    the filter is applied has to be specified by an additional argument `dim`
    string.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_cutoff: array_like
        A scalar specifying the cut-off frequency for the highpass filter.
    args :
        Additional arguments passed to frequency_filter function to further
        specify the filter. The following parameters can be passed:
        (order, irtype, filtfilt, apply_kwargs, in_nyq, dim)
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed.
        See frequency_filter documentation for furhter information.
    """
    kwargs = _update_ftype_kwargs(kwargs, "highpass", False)
    return frequency_filter(darray, f_cutoff, *args, **kwargs)


def bandpass(darray: xr.DataArray, f_low: ArrayLike, f_high: ArrayLike, *args, **kwargs) -> xr.DataArray:
    """Applies bandpass filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by an arg `dim`. If `darray` is multi dimensional then axis along which
    the filter is applied has to be specified by an additional argument `dim`
    string.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_low : array_like
        A scalar specifying the lower cut-off frequency for the bandpass filter.
    f_high : array_like
        A scalar specifying the higher cut-off frequency for the bandpass filter.
    args :
        Additional arguments passed to frequency_filter function to further
        specify the filter. The following parameters can be passed:
        (order, irtype, filtfilt, apply_kwargs, in_nyq, dim)
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed.
        See frequency_filter documentation for furhter information.
    """
    kwargs = _update_ftype_kwargs(kwargs, "bandpass", False)
    return frequency_filter(darray, [f_low, f_high], *args, **kwargs)


def bandstop(darray: xr.DataArray, f_low: ArrayLike, f_high: ArrayLike, *args, **kwargs) -> xr.DataArray:
    """Applies bandstop filter to a darray.

    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by an arg `dim`. If `darray` is multi dimensional then axis along which
    the filter is applied has to be specified by an additional argument `dim`
    string.

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.
    f_low : array_like
        A scalar specifying the lower cut-off frequency for the bandstop filter.
    f_high : array_like
        A scalar specifying the higher cut-off frequency for the bandstop filter.
    args :
        Additional arguments passed to frequency_filter function to further
        specify the filter. The following parameters can be passed:
        (order, irtype, filtfilt, apply_kwargs, in_nyq, dim)
    kwargs :
        Arbitrary keyword arguments passed when the filter is being designed.
        See frequency_filter documentation for furhter information.
    """
    kwargs = _update_ftype_kwargs(kwargs, "bandstop", True)
    return frequency_filter(darray, [f_low, f_high], *args, **kwargs)


def decimate(
    darray: xr.DataArray, q: ArrayLike = None, target_fs: ArrayLike = None, dim: str = None, **lowpass_kwargs
) -> xr.DataArray:
    """Decimate signal by given (int) factor or to closest possible target_fs
    along the specified dimension.

    Decimation: lowpass to new nyquist frequency and then downsample by factor `q`
    `lowpass_kwargs` are given to the lowpass method.

    If `q` is not given, it is approximated as the closest integer ratio
    of fs / `target_fs`, so `target_fs` must be smaller than current sampling
    frequency fs.

    If `q` < 2, decimation is skipped and a DecimationWarning is emitted

    Parameters
    ----------
    darray : DataArray
        An xarray type data to be decimated.
    q : array_like
        A scalar specifying the factor by which the signal should be decimated.
        If not given, it is approximated as the closest integer ratio of
        fs / `target_fs`. If set lower than 2, decimation is skipped and
        a DecimationWarning is emitted.
        Default is None.
    target_fs : array_like, optional
        A scalar specifying target sampling frequency of returning data.
        Must be smaller than current sampling frequency.
        Default is None.
    dim : string, optional
        A string specifing the dimension along which the filter is applied.
        If `darray` is 1-d then the dimension is found if not specified by `dim`.
        For multi dimensional `darray` has to be specified, otherwise ValueError
        is raised.
        Default is None.
    lowpass_kwargs :
        Arbitrary keyword arguments passed to the lowpass method. See lowpass
        method for further details.
    """
    dim = get_maybe_only_dim(darray, dim)
    if q is None:
        if target_fs is None:
            raise ValueError("either q or target_fs must be given")
        dt = get_sampling_step(darray, dim)
        q = int(np.rint(1.0 / (dt * target_fs)))
    if q < 2:  # decimation not possible or useless
        # show warning at caller level to see which signal it is related to
        warnings.warn("q factor %i < 2, skipping decimation" % q, DecimationWarning, stacklevel=2)
        return darray
    new_f_nyq = 1.0 / q
    lowpass_kwargs.setdefault("dim", dim)
    lowpass_kwargs.setdefault("in_nyq", True)
    ret = lowpass(darray, new_f_nyq, **lowpass_kwargs)
    ret = ret.isel(**{dim: slice(None, None, q)})
    return ret


def savgol_filter(
    darray: xr.DataArray,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = None,
    dim: str = None,
    mode: str = "interp",
    cval: float = 0.0,
) -> xr.DataArray:  # TODO augment scipy doc directly
    """Apply a Savitzky-Golay filter to an array.

    This is a 1-d filter.  If `darray` has dimension greater than 1, `dim`
    determines the dimension along which the filter is applied.
    Parameters
    ----------
    darray : DataArray
        An xarray type data to be filtered.  If values of `darray` are not
        a single or double precision floating point array, it will be converted
        to type ``numpy.float64`` before filtering.
    window_length : int
        The length of the filter window (i.e. the number of coefficients).
        `window_length` must be a positive odd integer. If `mode` is 'interp',
        `window_length` must be less than or equal to the size of `darray`.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute.  This must be a
        nonnegative integer.  The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.  Default is 1.0.
    dim : string, optional
        Specifies the dimension along which the filter is applied. For 1-d
        darray finds the only dimension, if not specified. For multi
        dimensional darray, the dimension for the filtering has to be
        specified, otherwise raises ValueError.
        Default is None.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.
    Returns
    -------
    y : DataArray, same shape as `darray`
        The filtered data.
    """
    dim = get_maybe_only_dim(darray, dim)
    if delta is None:
        delta = get_sampling_step(darray, dim)
        window_length = int(np.rint(window_length / delta))
        if window_length % 2 == 0:  # must be odd
            window_length += 1
    return xr.apply_ufunc(
        scipy.signal.savgol_filter,
        darray,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=dict(
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            mode=mode,
            cval=cval,
        ),
    )


@xr.register_dataarray_accessor("filt")
class FilterAccessor(object):
    """Accessor exposing common frequency and other filtering methods"""

    def __init__(self, darray: xr.DataArray):
        self.darray = darray

    @property
    def dt(self) -> float:
        """Sampling step of last axis"""
        return get_sampling_step(self.darray)

    @property
    def fs(self) -> float:
        """Sampling frequency in inverse units of self.dt"""
        return 1.0 / self.dt

    @property
    def dx(self) -> np.ndarray:
        """Sampling steps for all axes as array"""
        return np.array([get_sampling_step(self.darray, dim) for dim in self.darray.dims])

    # NOTE: the arguments are coded explicitly for tab-completion to work,
    # using a decorator wrapper with *args would not expose them
    def low(self, f_cutoff: ArrayLike, *args, **kwargs) -> xr.DataArray:
        """Lowpass filter, wraps lowpass"""
        return lowpass(self.darray, f_cutoff, *args, **kwargs)

    def high(self, f_cutoff: ArrayLike, *args, **kwargs) -> xr.DataArray:
        """Highpass filter, wraps highpass"""
        return highpass(self.darray, f_cutoff, *args, **kwargs)

    def bandpass(self, f_low: ArrayLike, f_high: ArrayLike, *args, **kwargs) -> xr.DataArray:
        """Bandpass filter, wraps bandpass"""
        return bandpass(self.darray, f_low, f_high, *args, **kwargs)

    def bandstop(self, f_low: ArrayLike, f_high: ArrayLike, *args, **kwargs) -> xr.DataArray:
        """Bandstop filter, wraps bandstop"""
        return bandstop(self.darray, f_low, f_high, *args, **kwargs)

    def freq(
        self,
        f_crit: ArrayLike,
        order: int = None,
        irtype: str = "iir",
        filtfilt: bool = True,
        apply_kwargs: dict = None,
        in_nyq: bool = False,
        dim: str = None,
        **kwargs,
    ) -> xr.DataArray:
        """General frequency filter, wraps frequency_filter"""
        return frequency_filter(
            self.darray,
            f_crit,
            order,
            irtype,
            filtfilt,
            apply_kwargs,
            in_nyq,
            dim,
            **kwargs,
        )

    __call__ = freq

    def savgol(
        self,
        window_length: int,
        polyorder: int,
        deriv: int = 0,
        delta: float = None,
        dim: str = None,
        mode: str = "interp",
        cval: float = 0.0,
    ) -> xr.DataArray:
        """Savitzky-Golay filter, wraps savgol_filter"""
        return savgol_filter(self.darray, window_length, polyorder, deriv, delta, dim, mode, cval)

    def decimate(
        self, q: ArrayLike = None, target_fs: ArrayLike = None, dim: str = None, **lowpass_kwargs
    ) -> xr.DataArray:
        """Decimate signal, wraps decimate"""
        return decimate(self.darray, q, target_fs, dim, **lowpass_kwargs)
