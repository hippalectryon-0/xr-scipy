import xarray
import scipy.signal
import numpy as np
try:
    from scipy.fftpack import next_fast_len
except ImportError:
    def next_fast_len(size):
        return 2**int(np.ceil(np.log2(size)))

from .utils import get_maybe_last_dim_axis, get_sampling_step

_FREQUENCY_DIM = 'frequency'

_DOCSTRING_COMMON = """              
    Parameters
    ----------
    darray : xarray
        Series of measurement values
    other_darray : xarray
        Series of measurement values
    fs : float, optional
        Sampling frequency of the `darray` and `other_darray` time series. If not specified,
        crossspectrogram will calculate it from the sampling step.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    seglen : float, optional
        Segment length in units of the used (e.g. time) dimmension. 
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // overlap_ratio``. Defaults to `None`.
    overlap_ratio : int, optional
        Used to calculate noverlap, if this is not specified (see above). Defaults to 2.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for 
        complex data, a two-sided spectrum is always returned.
    dim : str, optional, if 1D data provided, otherwise must be specified
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis.
"""


def crossspectrogram(darray, other_darray, fs=None, seglen=None,
                     overlap_ratio=2, window='hann', nperseg=256,
                     noverlap=None, nfft=None, detrend='constant',
                     return_onesided=True, scaling='density', dim=None,
                     mode='psd'):
        r"""
    Calculate the cross spectrogram.
    Parameters
    ----------
    darray : xarray
        Series of measurement values
    other_darray : xarray
        Series of measurement values
    fs : float, optional
        Sampling frequency of the `darray` and `other_darray` time series. If not specified,
        crossspectrogram will calculate it from the sampling step.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    seglen : float, optional
        Segment length in units of the used (e.g. time) dimmension. 
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // overlap_ratio``. Defaults to `None`.
    overlap_ratio : int, optional
        Used to calculate noverlap, if this is not specified (see above). Defaults to 2.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for 
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `darray` and `other_darray` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'
    dim : str, optional, if 1D data provided, otherwise must be specified
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis.
    mode : str
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively.   

    Returns
    -------
    Pxy : xarray.DataArray
        Cross spectrogram of 'darray' and 'other_darray' returned with two new dimmensions: frequency and "time".
    --------
    By convention, Pxy is computed with the conjugate FFT of `darray` 
    multiplied by the FFT of `other_darray`.
    If the input series differ in length, the shorter series will be
    zero-padded to match.
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.
    
    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of
           Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975
    """
    
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    if fs is None:
        dt = get_sampling_step(darray, dim)
        fs = 1.0 / dt
    else:
        dt = 1.0 / fs
    if seglen is not None:
        nperseg = int(np.rint(seglen / dt))
        nfft = next_fast_len(nperseg)
    if noverlap is None:
        noverlap = nperseg // overlap_ratio
    # outer join align to ensure proper sampling
    darray, other_darray = xarray.align(darray, other_darray, join='outer',
                                        copy=False)
    together = (darray, other_darray)
    if set(darray.dims) != set(other_darray.dims):
        together = xarray.broadcast(*together)
    d_val, od_val = (d.values for d in together)

    f, t, Pxy = scipy.signal.spectral._spectral_helper(d_val,
                                                       od_val,
                                                       fs, window, nperseg,
                                                       noverlap, nfft, detrend,
                                                       return_onesided,
                                                       scaling, axis, mode)
    t_0 = float(darray.coords[dim][0])
    t_axis = t + t_0
    # new dimensions and coordinates construction
    coord_darr = darray if darray.ndim >= other_darray.ndim else other_darray
    new_dims = list(coord_darr.dims)
    # frequency replaces data dim
    new_dims[new_dims.index(dim)] = _FREQUENCY_DIM
    new_dims.append(dim)   # make data dim last
    # select nearest times on other possible coordinates
    coords_ds = coord_darr.coords.to_dataset()
    coords_ds = coords_ds.sel(**{dim:t_axis, 'method':'nearest'})
    coords_ds[dim] = t_axis
    coords_ds[_FREQUENCY_DIM] = f
    new_name = 'crossspectrogram_{}_{}'.format(darray.name, other_darray.name)
    return xarray.DataArray(Pxy, name=new_name,
                            dims=new_dims, coords=coords_ds.coords)

def csd(darray, other_darray, fs=None, seglen=None, overlap_ratio=2,
        window='hann', nperseg=256, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density', dim=None,
        mode='psd'):
    r"""
    Estimate the cross power spectral density, Pxy, using Welch's method.
    Parameters
    ----------
    darray : xarray
        Series of measurement values
    other_darray : xarray
        Series of measurement values
    fs : float, optional
        Sampling frequency of the `darray` and `other_darray` time series. If not specified,
        crossspectrogram will calculate it from the sampling step.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    seglen : float, optional
        Segment length in units of the used (e.g. time) dimmension. 
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // overlap_ratio``. Defaults to `None`.
    overlap_ratio : int, optional
        Used to calculate noverlap, if this is not specified (see above). Defaults to 2.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for 
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `darray` and `other_darray` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'
    dim : str, optional, if 1D data provided, otherwise must be specified
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis.
    mode : str
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively.    

    Returns
    -------   
    Pxy : xarray.DataArray
        Cross spectral density or cross power spectrum with dimmension of an array of sample frequencies.
    --------
    By convention, Pxy is computed with the conjugate FFT of `darray` 
    multiplied by the FFT of `other_darray`.
    If the input series differ in length, the shorter series will be
    zero-padded to match.
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. For the default Hann window an overlap of
    50% is a reasonable trade off between accurately estimating the
    signal power, while not over counting any of the data. Narrower
    windows may require a larger overlap.
    
    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of
           Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975
    """
    
    Pxy = crossspectrogram(darray, other_darray, fs, seglen,
                           overlap_ratio, window, nperseg, noverlap, nfft, detrend,
                           return_onesided, scaling, dim, mode)
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    Pxy = Pxy.mean(dim=dim)
    Pxy.name = 'csd_{}_{}'.format(darray.name, other_darray.name)
    return Pxy


def freq2lag(darray, is_onesided=False, f_dim=_FREQUENCY_DIM):
    """
    Calculate the lag corresponding to .
    Parameters
    ----------
    darray : xarray.DataArray
        The result of crossspectral density serves as an input.
    is_onesided : boolean
        Indicated whether frequency dimmension is one sided or full. Defaults to 'False'.
    f_dim : string
        Defaults to 'frequency'.

    Returns
    -------
    ret : xarray
        Array of 'ret' returned with the main dimmension switched to the lag.
    """    
    
    axis = darray.get_axis_num(f_dim)
    if is_onesided:
        ret = np.fft.irfft(darray, axis=axis).real
    else:
        ret = np.fft.ifft(darray, axis=axis).real
    ret = darray.__array_wrap__(ret)
    ret.name = 'ifft_' + darray.name
    f = ret.coords[f_dim]
    df = f[1] - f[0]
    dt = 1.0 / (df * darray.shape[axis])
    lag =  f /df * dt
    ret.coords['lag'] = lag
    return ret.swap_dims({f_dim: 'lag'}).isel(lag=lag.argsort().values)


def xcorrelation(darray, other_darray, normalize=True, fs=None, seglen=None,
                 overlap_ratio=2, window='hann', nperseg=256, noverlap=None,
                 nfft=None, detrend='constant', dim=None):
    """
    Calculate the crosscorrelation.
    Parameters
    ----------
    darray : xarray
        Series of measurement values
    other_darray : xarray
        Series of measurement values
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    seglen : float, optional
        Segment length in units of the used (e.g. time) dimmension. 
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // overlap_ratio``. Defaults to `None`.
    overlap_ratio : int, optional
        Used to calculate noverlap, if this is not specified (see above). Defaults to 2.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    dim : str, optional, if 1D data provided, otherwise must be specified
        Axis along which the CSD is computed for both inputs; the
        default is over the last axis.  

    Returns
    -------
    xcorr : xarray
        Crosscorrelation of 'darray' and 'other_darray' returned with the main dimmension switched to the lag.
    """
    
    csd_d = csd(darray, other_darray, fs, seglen, overlap_ratio, window,
                nperseg, noverlap, nfft, detrend, return_onesided=False,
                scaling='spectrum', dim=dim, mode='psd')
    xcorr = freq2lag(csd_d)
    if normalize:
        norm = 1
        for sig in (darray, other_darray):
            sig_std = psd(sig, fs, seglen, overlap_ratio, window, nperseg,
                          noverlap, nfft, detrend, return_onesided=False,
                          scaling='spectrum', dim=dim, mode='psd').mean(dim=_FREQUENCY_DIM)**0.5
            norm = norm * sig_std
        xcorr /= norm
    return xcorr


def spectrogram(darray, fs=None, seglen=None, overlap_ratio=2, window='hann',
                nperseg=256, noverlap=None, nfft=None, detrend='constant',
                return_onesided=True, scaling='density', dim=None, mode='psd'):
    """
    Calculate the spectrogram using crossspectrogram applied to the same xarray (darray = other_darray).
    {common}
    mode : str
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively.  
    
    Returns
    -------
    Pxx : xarray.DataArray
        Spectrogram of 'darray'.
    """.format(common=_DOCSTRING_COMMON)    
    Pxx = crossspectrogram(darray, darray, fs, seglen, overlap_ratio, window,
                           nperseg, noverlap, nfft, detrend, return_onesided,
                           scaling, dim, mode)
    Pxx.name = 'spectrogram_{}'.format(darray.name)
    return Pxx


def psd(darray, fs=None, seglen=None, overlap_ratio=2, window='hann',
        nperseg=256, noverlap=None, nfft=None, detrend='constant',
        return_onesided=True, scaling='density', dim=None, mode='psd'):
    """
    Calculate the power spectral density.
    {common}
    mode : str
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `darray` and `other_darray` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'
    
    Returns
    -------
    Pxx : xarray.DataArray
        Power spectrum density of 'darray'.
    """.format(common=_DOCSTRING_COMMON)     
    
    Pxx = spectrogram(darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft, detrend, return_onesided, scaling, dim,
                      mode)
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    Pxx = Pxx.mean(dim=dim)
    Pxx.name = 'psd_{}'.format(darray.name)
    return Pxx

# TODO f_res
def coherogram(darray, other_darray, fs=None, seglen=None, overlap_ratio=2,
               nrolling=8, window='hann', nperseg=256, noverlap=None,
               nfft=None, detrend='constant', return_onesided=True, dim=None):
    """
    Calculate the coherogram.
    {common}
    nrolling : int, optional
            Number of running windows to provide the mean of coherence.
    
    Returns
    -------
    coh : xarray.DataArray
        Coherogram of 'darray' and 'other_darray'.
    """.format(common=_DOCSTRING_COMMON)
    
    Pxx = spectrogram(darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft, detrend, return_onesided, dim=dim)
    Pyy = spectrogram(other_darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft, detrend, return_onesided, dim=dim)
    Pxy = crossspectrogram(darray, other_darray, fs, seglen, overlap_ratio,
                           window, nperseg, noverlap, nfft, detrend,
                           return_onesided, dim=dim)
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    rol_kw = {dim: nrolling, 'center': True}
    coh = (Pxy.rolling(**rol_kw).mean() /
           (Pxx.rolling(**rol_kw).mean() * Pyy.rolling(**rol_kw).mean())**0.5)
    coh.dropna(dim=dim)         # drop nan from averaging edges
    coh.name = 'coherogram_{}_{}'.format(darray.name, other_darray.name)
    return coh


def coherence(darray, other_darray, fs=None, seglen=None, overlap_ratio=2,
              window='hann', nperseg=256, noverlap=None, nfft=None,
              detrend='constant', dim=None):
    """
    Calculate the coherence.
    {common}
    
    Returns
    -------
    coh : xarray.DataArray
        Coherence of 'darray' and 'other_darray'.
    """.format(common=_DOCSTRING_COMMON)
    
    Pxx = psd(darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft, detrend, dim=dim)
    Pyy = psd(other_darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft, detrend, dim=dim)
    Pxy = csd(darray, other_darray, fs, seglen, overlap_ratio,
                           window, nperseg, noverlap, nfft, detrend, dim=dim)
    coh = Pxy / np.sqrt(Pxx * Pyy)  # magnitude squared coherence
    coh.name = 'coherence_{}_{}'.format(darray.name, other_darray.name)
    return coh


def hilbert(darray, N=None, dim=None):
    """
    Compute the analytic signal, using the Hilbert transform.
    The transformation is done along the selected axis.

    Parameters
    darray : xarray
    Signal data. Must be real.

    N : int, optional
    Number of Fourier components. Default: x.shape[axis]

    dim : string, optional
    Axis along which to do the transformation.

    Returns
    darray : xarray
    Analytic signal of the Hilbert transform of 'darray' along selected axis.
    """
    
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    n_orig = darray.shape[axis]
    N_unspecified = N is None
    if N_unspecified:
        N = next_fast_len(n_orig)
    out = scipy.signal.hilbert(np.asarray(darray), N, axis=axis)
    if n_orig != N and N_unspecified:
        sl = [slice(None)] * out.ndim
        sl[axis] = slice(None, n_orig)
        out = out[sl]
    if not N_unspecified and N != n_orig:
        return out
    else:
        return darray.__array_wrap__(out)
