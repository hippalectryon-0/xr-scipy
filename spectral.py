import xarray
import scipy.signal
import numpy as np
try:
    from scipy.fftpack import next_fast_len
except ImportError:
    def next_fast_len(size):
        return 2**int(np.ceil(np.log2(size)))

from .utils import get_sampling_step, get_maybe_only_dim

_FREQUENCY_DIM = 'frequency'

_DOCSTRING_COMMON_PARAMS = """    fs : float, optional
        Sampling frequency of the `darray` and `other_darray` (time) series.
        If not specified, will be calculated it from the sampling step
        of the specified (or only) dimension.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    seglen : float, optional
        Segment length (i.e. nperseg) in units of the used (e.g. time) dimension.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // overlap_ratio``. Defaults to `None`.
    overlap_ratio : int, optional
        Used to calculate noverlap, if this is not specified (see above).
        Defaults to 2.
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
    dim : str, optional
        Dimension along which the FFT is computed and sampling step calculated.
        If the signal is 1D, uses the only dimension, otherwise must be specified."""

_DOCSTRING_MODE_PARAM = """mode : str
        Defines what kind of return values are expected. Options are
        ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
        equivalent to the output of `stft` with no padding or boundary
        extension. 'magnitude' returns the absolute magnitude of the
        STFT. 'angle' and 'phase' return the complex angle of the STFT,
        with and without unwrapping, respectively."""

_DOCSTRING_SCALING_PARAM = """scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross spectrum
        ('spectrum') where `Pxy` has units of V**2, if `darray` and `other_darray` are
        measured in V and `fs` is measured in Hz. Defaults to 'density'.\
"""

def _add2docstring_common_params(func):
    if hasattr(func, '__doc__'):
        func.__doc__ = func.__doc__.format(
            common_params=_DOCSTRING_COMMON_PARAMS,
            mode_param=_DOCSTRING_MODE_PARAM,
            scaling_param=_DOCSTRING_SCALING_PARAM,
        )
    return func


@_add2docstring_common_params
def crossspectrogram(darray, other_darray, fs=None, seglen=None,
                     overlap_ratio=2, window='hann', nperseg=256,
                     noverlap=None, nfft=None, detrend='constant',
                     return_onesided=True, dim=None, scaling='density',
                     mode='psd'):
    """Calculate the cross spectrogram.

    Parameters
    ----------
    darray : xarray
        Series of measurement values
    other_darray : xarray
        Series of measurement values
    {common_params}
    {scaling_param}
    {mode_param}

    Returns
    -------
    Pxy : xarray.DataArray
        Cross spectrogram of 'darray' and 'other_darray'
        with one new dimmension frequency and new coords for the specified dim.

    Notes
    -----
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
    if darray is not other_darray:
        # outer join align to ensure proper sampling
        darray, other_darray = xarray.align(darray, other_darray, join='outer',
                                            copy=False)
    kwargs = dict(fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                  nfft=nfft, detrend=detrend, return_onesided=return_onesided,
                  scaling=scaling, axis=-1, mode=mode
    )
    f, t, Pxy = xarray.apply_ufunc(scipy.signal.spectral._spectral_helper,
                                  darray, other_darray,
                                  input_core_dims=[[dim]]*2,
                                  output_core_dims=[[_FREQUENCY_DIM], ['t'], [_FREQUENCY_DIM, 't']],
                                  kwargs=kwargs)
    t_0 = float(darray.coords[dim][0])
    Pxy.coords['t'] = t + t_0
    Pxy.coords[_FREQUENCY_DIM] = f
    Pxy = Pxy.rename(t=dim)
    # new dimensions and coordinates construction
    coord_darr = darray if darray.ndim >= other_darray.ndim else other_darray
    # select nearest times on other possible coordinates
    coords_ds = coord_darr.coords.to_dataset()
    coords_ds = coords_ds.sel(**{dim: Pxy.time, 'method':'nearest'})
    Pxy = Pxy.assign_coords(**coords_ds.coords)
    Pxy.name = 'crossspectrogram_{}_{}'.format(darray.name, other_darray.name)
    return Pxy


@_add2docstring_common_params
def csd(darray, other_darray, fs=None, seglen=None, overlap_ratio=2,
        window='hann', nperseg=256, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, dim=None, scaling='density',
        mode='psd'):
    """
    Estimate the cross power spectral density, Pxy, using Welch's method.

    Parameters
    ----------
    darray : xarray
        Series of measurement values
    other_darray : xarray
        Series of measurement values
    {common_params}
    {scaling_param}
    {mode_param}

    Returns
    -------
    Pxy : xarray.DataArray
        Cross spectral density or cross power spectrum with frequency dimension.

    Notes
    -----
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
                           return_onesided, dim, scaling, mode)
    dim = get_maybe_only_dim(darray, dim)
    Pxy = Pxy.mean(dim=dim)
    Pxy.name = 'csd_{}_{}'.format(darray.name, other_darray.name)
    return Pxy


def freq2lag(darray, is_onesided=False, f_dim=_FREQUENCY_DIM):
    """
    Calculate the inverse FFT along the frequency dimension into lag-space

    Parameters
    ----------
    darray : xarray.DataArray
        The result of crossspectral density serves as an input.
    is_onesided : boolean
        Indicated whether frequency dimmension is one sided or full.
        Defaults to 'False'.
    f_dim : string
        Defaults to 'frequency'.

    Returns
    -------
    ret : xarray
        Array of 'ret' returned with the main dimmension switched to the time lag.
    """
    axis = darray.get_axis_num(f_dim)
    if is_onesided:
        ret = xarray.apply_ufunc(fft.irfft,darray,
                              input_core_dims = [[f_dim]],
                              output_core_dims = [[f_dim]])
        ret = ret.real
    else:
        ret = xarray.apply_ufunc(fft.ifft,darray,
                              input_core_dims = [[f_dim]],
                              output_core_dims = [[f_dim]])
        ret = ret.real    
    ret.name = 'ifft_' + darray.name
    f = ret.coords[f_dim]
    df = f[1] - f[0]
    dt = 1.0 / (df * darray.shape[axis])
    lag =  f /df * dt
    ret.coords['lag'] = lag
    return ret.swap_dims({f_dim: 'lag'}).isel(lag=lag.argsort().values)


@_add2docstring_common_params
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
    {common_params}

    Returns
    -------
    xcorr : xarray
        Crosscorrelation of 'darray' and 'other_darray'
        with the given dimension switched to the lag.
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


@_add2docstring_common_params
def spectrogram(darray, fs=None, seglen=None, overlap_ratio=2, window='hann',
                nperseg=256, noverlap=None, nfft=None, detrend='constant',
                return_onesided=True, dim=None, scaling='density', mode='psd'):
    """
    Calculate the spectrogram using crossspectrogram applied to the same data

    Parameters
    ----------
    darray : xarray
        Series of measurement values
    {common_params}
    {scaling_param}
    {mode_param}

    Returns
    -------
    Pxx : xarray.DataArray
        Spectrogram of 'darray'.
    """
    Pxx = crossspectrogram(darray, darray, fs, seglen, overlap_ratio, window,
                           nperseg, noverlap, nfft, detrend, return_onesided,
                           dim, scaling, mode)
    Pxx.name = 'spectrogram_{}'.format(darray.name)
    return Pxx


@_add2docstring_common_params
def psd(darray, fs=None, seglen=None, overlap_ratio=2, window='hann',
        nperseg=256, noverlap=None, nfft=None, detrend='constant',
        return_onesided=True, scaling='density', dim=None, mode='psd'):
    """
    Calculate the power spectral density.

    Parameters
    ----------
    darray : xarray
        Series of measurement values
    {common_params}
    {scaling_param}
    {mode_param}

    Returns
    -------
    Pxx : xarray.DataArray
        Power spectrum density of 'darray'.
    """
    Pxx = spectrogram(darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft, detrend, return_onesided, dim, scaling,
                      mode)
    dim = get_maybe_only_dim(darray, dim)
    Pxx = Pxx.mean(dim=dim)
    Pxx.name = 'psd_{}'.format(darray.name)
    return Pxx

# TODO f_res
@_add2docstring_common_params
def coherogram(darray, other_darray, fs=None, seglen=None, overlap_ratio=2,
               nrolling=8, window='hann', nperseg=256, noverlap=None,
               nfft=None, detrend='constant', return_onesided=True, dim=None):
    """
    Calculate the coherogram

    The coherence (i.e. averaging of complex phasors) is done
    using a rolling average <...> of given size along the FFT windows
    and then coherogram = <crossspectrogram> / sqrt(<spectrogram1> * <spectrogram2>)

    Parameters
    ----------
    darray : xarray
        Series of measurement values
    other_darray : xarray
        Series of measurement values
    nrolling : int, optional
            Number of FFT windows used in the rolling average.
    {common_params}

    Returns
    -------
    coh : xarray.DataArray, complex
        Coherogram of 'darray' and 'other_darray'.
        It is complex and abs(coh)**2 is the squared magnitude coherohram.
    """
    Pxx = spectrogram(darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft, detrend, return_onesided, dim=dim)
    Pyy = spectrogram(other_darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft, detrend, return_onesided, dim=dim)
    Pxy = crossspectrogram(darray, other_darray, fs, seglen, overlap_ratio,
                           window, nperseg, noverlap, nfft, detrend,
                           return_onesided, dim=dim)
    dim = get_maybe_only_dim(darray, dim)
    rol_kw = {dim: nrolling, 'center': True}
    coh = (Pxy.rolling(**rol_kw).mean() /
           (Pxx.rolling(**rol_kw).mean() * Pyy.rolling(**rol_kw).mean())**0.5)
    coh.dropna(dim=dim)         # drop nan from averaging edges
    coh.name = 'coherogram_{}_{}'.format(darray.name, other_darray.name)
    return coh


@_add2docstring_common_params
def coherence(darray, other_darray, fs=None, seglen=None, overlap_ratio=2,
              window='hann', nperseg=256, noverlap=None, nfft=None,
              detrend='constant', dim=None):
    """
    Calculate the coherence as <CSD> / sqrt(<PSD1> * <PSD2>)

    Parameters
    ----------
    darray : xarray
        Series of measurement values
    other_darray : xarray
        Series of measurement values
    {common_params}

    Returns
    -------
    coh : xarray.DataArray, complex
        Coherence of 'darray' and 'other_darray'.
        It is complex and abs(coh)**2 is the squared magnitude coherohram.
    """
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
    The transformation is done along the selected dimension.

    Parameters
    ----------
    darray : xarray
        Signal data. Must be real.
    N : int, optional
        Number of Fourier components. Defaults to size along dim.
    dim : string, optional
        Axis along which to do the transformation.
        Uses the only dimension of darray is 1D.

    Returns
    -------
    darray : xarray
        Analytic signal of the Hilbert transform of 'darray' along selected axis.
    """
    dim = get_maybe_only_dim(darray, dim)
    n_orig = darray.shape[axis]
    N_unspecified = N is None
    if N_unspecified:
        N = next_fast_len(n_orig)
    out = xarray.apply_ufunc(_hilbert_wraper, darray,
                              input_core_dims = [[dim]],
                              output_core_dims = [[dim]],
                              kwargs=dict(N = N, n_orig = n_orig, N_unspecified = N_unspecified))

    return out
    

def _hilbert_wraper(darray, N, n_orig, N_unspecified, axis = -1):
    """
    Hilbert wraper used to keep the signal dimension length constant
    """
    out = scipy.signal.hilbert(np.asarray(darray), N, axis = axis)
    
    if n_orig != N and N_unspecified:
        sl = [slice(None)] * out.ndim
        sl[axis] = slice(None, n_orig)
        out = out[sl]
    return out

