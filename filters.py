import warnings
import xarray
import scipy.signal
import numpy as np

try:
    from scipy.signal import sosfiltfilt
except ImportError:
    sosfiltfilt = None

from .utils import get_sampling_step, get_maybe_only_dim

def _firwin_ba(*args, **kwargs):
    if not kwargs.get('pass_zero'):
        args = (args[0] + 1,) + args[1:]  # numtaps must be odd
    return scipy.signal.firwin(*args, **kwargs), np.array([1])


_BA_FUNCS = {
    'iir': scipy.signal.iirfilter,
    'fir': _firwin_ba,
    }

_ORDER_DEFAULTS = {
    'iir': 4,
    'fir': 29,
    }

class FilteringNaNWarning(Warning):
    pass
# always (not just once) show filtering NaN warnings to see the responsible signal
warnings.filterwarnings('always', category=FilteringNaNWarning)


def frequency_filter(darray, f_crit, order=None, irtype='iir', filtfilt=True,
                     apply_kwargs=None, in_nyq=False, dim=None, **kwargs):
    """ Applies given frequency filter to a darray.
    
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
        The data to be filtered.
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
        raise ValueError('Wrong argument for irtype: {}, must be one of {}'.format(
            irtype, _BA_FUNCS.keys()))
    if order is None:
        order = _ORDER_DEFAULTS[irtype]
    if apply_kwargs is None:
        apply_kwargs = {}
    dim = get_maybe_only_dim(darray, dim)
    f_crit_norm = np.asarray(f_crit, dtype=np.float)
    if not in_nyq:              # normalize by Nyquist frequency
        f_crit_norm *= 2 * get_sampling_step(darray, dim)
    if np.any(np.isnan(np.asarray(darray))): # only warn since simple forward-filter or FIR is valid
        warnings.warn('data contains NaNs, filter will propagate them',
                      FilteringNaNWarning, stacklevel=2)
    if sosfiltfilt and irtype == 'iir': # TODO merge with other if branch
        sos = scipy.signal.iirfilter(order, f_crit_norm, output='sos', **kwargs)
        if filtfilt:
            ret = xarray.apply_ufunc(sosfiltfilt, sos, darray,
                                     input_core_dims = [[],[dim]],
                                     output_core_dims = [[dim]],
                                     kwargs = apply_kwargs)
        else:
            ret = xarray.apply_ufunc(scipy.signal.sosfilt, sos, darray,
                                     input_core_dims = [[],[dim]],
                                     output_core_dims = [[dim]],
                                     kwargs = apply_kwargs)
    else:
        b, a = _BA_FUNCS[irtype](order, f_crit_norm, **kwargs)
        if filtfilt:
            ret = xarray.apply_ufunc(scipy.signal.filtfilt, b, a, darray,
                                     input_core_dims = [[],[],[dim]],
                                     output_core_dims = [[dim]],
                                     kwargs = apply_kwargs)            
        else:
            ret = xarray.apply_ufunc(scipy.signal.lfilter, b, a, darray,
                                     input_core_dims = [[],[],[dim]],
                                     output_core_dims = [[dim]],
                                     kwargs = apply_kwargs)
    return ret


def _update_ftype_kwargs(kwargs, iirvalue, firvalue):
    if kwargs.get('irtype', 'iir') == 'iir':
        kwargs.setdefault('btype', iirvalue)
    else:                       # fir
        kwargs.setdefault('pass_zero', firvalue)
    return kwargs


def lowpass(darray, f_cutoff, *args, **kwargs):
    """ Applies lowpass filter to a darray.
    
    This is a 1-d filter. If the darray is one dimensional, then the dimension
    along which the filter is applied is chosen automatically if not specified
    by an arg `dim`. If `darray` is multi dimensional then axis along which
    the filter is applied has to be specified by an additional argument `dim`
    string.
    
    Parameters
    ----------
    darray : DataArray
        The data to be filtered.
    f_cutoff: array_like
        A scalar specifying the cut-off frequency for the lowpass filter.
    arg: 
        Additional arguments passed to frequency_filter function to further
        specify the filter. The following parameters can be passed:
        (order, irtype, filtfilt, apply_kwargs, in_nyq, dim)
    kwargs:
        Arbitrary keyword arguments passed when the filter is being designed.
        See frequency_filter documentation for furhter information.
    """
    kwargs = _update_ftype_kwargs(kwargs, 'lowpass', True)
    return frequency_filter(darray, f_cutoff, *args, **kwargs)


def highpass(darray, f_cutoff, *args, **kwargs):
    kwargs = _update_ftype_kwargs(kwargs, 'highpass', False)
    return frequency_filter(darray, f_cutoff, *args, **kwargs)


def bandpass(darray, f_low, f_high, *args, **kwargs):
    kwargs = _update_ftype_kwargs(kwargs, 'bandpass', False)
    return frequency_filter(darray, [f_low, f_high], *args, **kwargs)


def bandstop(darray, f_low, f_high, *args, **kwargs):
    kwargs = _update_ftype_kwargs(kwargs, 'bandstop', True)
    return frequency_filter(darray, [f_low, f_high], *args, **kwargs)


class DecimationWarning(Warning):
    pass
# always (not just once) show decimation warnings to see the responsible signal
warnings.filterwarnings('always', category=DecimationWarning)

def decimate(darray, q=None, target_fs=None, dim=None, **lowpass_kwargs):
    '''Decimate signal by given (int) factor or to closest possible target_fs

    along the specified dimension

    Decimation: lowpass to new nyquist frequency and then downsample by factor q
    lowpass_kwargs are given to the lowpass method

    If q is not given, it is approximated as the closest integer ratio
    of fs / target_fs, so target_fs must be smaller than current sampling frequency fs

    If q < 2, decimation is skipped and a DecimationWarning is emitted
    '''
    dim = get_maybe_only_dim(darray, dim)
    if q is None:
        if target_fs is None:
            raise ValueError('either q or target_fs must be given')
        dt = get_sampling_step(darray, dim)
        q = int(np.rint(1.0 / (dt * target_fs)))
    if q < 2:                   # decimation not possible or useless
        # show warning at caller level to see which signal it is related to
        warnings.warn('q factor %i < 2, skipping decimation' % q,
                      DecimationWarning, stacklevel=2)
        return darray
    new_f_nyq = 1.0 / q
    lowpass_kwargs.setdefault('dim', dim)
    lowpass_kwargs.setdefault('in_nyq', True)
    ret = lowpass(darray, new_f_nyq, **lowpass_kwargs)
    ret = ret.isel(**{dim: slice(None, None, q)})
    return ret

def savgol_filter(darray, window_length, polyorder, deriv=0, delta=None,
                  dim=None, mode='interp', cval=0.0):
    """ Apply a Savitzky-Golay filter to an array.
    
        This is a 1-d filter.  If `darray` has dimension greater than 1, `dim`
        determines the dimension along which the filter is applied.
        Parameters
        ----------
        darray : DataArray
            The data to be filtered.  If values in `darray` are not a single or
            double precision floating point array, it will be converted to type
            ``numpy.float64`` before filtering.
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
    return xarray.apply_ufunc(scipy.signal.savgol_filter, darray,
                              input_core_dims = [[dim]],
                              output_core_dims = [[dim]],
                              kwargs=dict(window_length = window_length,
                                          polyorder = polyorder,
                                          deriv = deriv, delta = delta,
                                          mode = mode, cval = cval))

@xarray.register_dataarray_accessor('filt')
class FilterAccessor(object):
    '''Accessor exposing common frequency and other filtering methods'''

    def __init__(self, darray):
        self.darray = darray

    @property
    def dt(self):
        '''Sampling step of last axis'''
        return get_sampling_step(self.darray)

    @property
    def fs(self):
        """Sampling frequency in inverse units of self.dt"""
        return 1.0 / self.dt

    @property
    def dx(self):
        '''Sampling steps for all axes as array'''
        return np.array([get_sampling_step(self.darray, dim) for dim in self.darray.dims])

    # NOTE: the arguments are coded explicitly for tab-completion to work,
    # using a decorator wrapper with *args would not expose them
    def low(self, f_cutoff, *args, **kwargs):
        """Lowpass filter, wraps lowpass"""
        return lowpass(self.darray, f_cutoff, *args, **kwargs)

    def high(self, f_cutoff, *args, **kwargs):
        """Highpass filter, wraps highpass"""
        return highpass(self.darray, f_cutoff, *args, **kwargs)

    def bandpass(self, f_low, f_high, *args, **kwargs):
        """Bandpass filter, wraps bandpass"""
        return bandpass(self.darray, f_low, f_high, *args, **kwargs)

    def bandstop(self, f_low, f_high, *args, **kwargs):
        """Bandstop filter, wraps bandstop"""
        return bandstop(self.darray, f_low, f_high, *args, **kwargs)

    def freq(self, f_crit, order=None, irtype='iir', filtfilt=True,
             apply_kwargs=None, in_nyq=False, dim=None, **kwargs):
        """General frequency filter, wraps frequency_filter"""
        return frequency_filter(self.darray, f_crit, order, irtype, filtfilt,
                                apply_kwargs, in_nyq, dim, **kwargs)

    __call__ = freq

    def savgol(self, window_length, polyorder, deriv=0, delta=None,
                      dim=None, mode='interp', cval=0.0):
        """Savitzky-Golay filter, wraps savgol_filter"""
        return savgol_filter(self.darray, window_length, polyorder, deriv, delta,
                             dim, mode, cval)

    def decimate(self, q=None, target_fs=None, dim=None, **lowpass_kwargs):
        """Decimate signal, wraps decimate"""
        return decimate(self.darray, q, target_fs, dim, **lowpass_kwargs)
