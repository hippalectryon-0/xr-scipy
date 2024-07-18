r""".. _spectral:

Spectral (FFT) analysis
-----------------------

.. ipython:: python
   :suppress:

    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import xrscipy.signal as dsp
    import xrscipy.signal.extra as dsp_extra


xr-scipy wraps some of scipy spectral analysis functions such as :py:func:`scipy.signal.spectrogram`, :py:func:`scipy.signal.csd` etc. For convenience, the ``xrscipy.signal`` namespace will be imported under the alias ``dsp``

.. ipython:: python

    import xrscipy.signal as dsp
    import xrscipy.signal.extra as dsp_extra

To demonstrate the basic functionality, let's create two simple example DataArray at a similar frequency but one with a frequency drift and some noise:

.. ipython:: python

    time_ax = np.arange(0,100,0.01)
    sig_1 = xr.DataArray(np.sin(100 * time_ax) + np.random.rand(len(time_ax))*3,
                         coords=[("time", time_ax)], name='sig_1')
    sig_2 = xr.DataArray((np.cos(100 * time_ax) + np.random.rand(len(time_ax))*3 +
                          3*np.sin(30 * time_ax**1.3)),
                         coords=[("time", time_ax)], name='sig_2')


Power spectra
^^^^^^^^^^^^^

The :py:func:`~xrscipy.signal.spectrogram` function can be used directly on an xarray
DataArray object. The returned object is again an ``xarray.DataArray`` object.

.. ipython:: python

    spec_1 = dsp.spectrogram(sig_1)
    spec_2 = dsp.spectrogram(sig_2)
    spec_2

The ``frequency`` dimension coords are based on the transformed dimension (``time`` in this case) coords sampling (i.e. inverse units). When the signal is 1D, the dimension does not have to be provided.


.. ipython:: python
    :okwarning:

    norm = plt.matplotlib.colors.LogNorm()
    plt.subplot(211)
    spec_1.plot(norm=norm)
    plt.subplot(212)
    spec_2.plot(norm=norm)
    @savefig spectrograms.png width=4in
    plt.show()


These routines calculate the FFT on  segments of the signal of a length controlled by ``nperseg`` and ``nfft`` parameters. The routines here offer a convenience parameter ``seglen`` which makes it possible to specify the segment length in the units of the transformed dimension's coords. If ``seglen`` is specified, ``nperseg`` is then calculated from it and ``nfft`` is set using ``scipy.fftpack.next_fast_len`` (or to closest higher power of 2). A desired frequency resolution spacing ``df`` can be achieved by specifying ``seglen=1/df``.

Another convenience parameter is ``overlap_ratio`` which calculates the ``noverlap`` parameter (by how many points the segments overlap) as ``noverlap = np.rint(overlap_ratio * nperseg)``

For example, these parameters calculate the spectrogram with a higher frequency resolution and try to make for the longer segments by overlapping them by 75%.

.. ipython:: python

    dsp.spectrogram(sig_1, seglen=1, overlap_ratio=0.75)

All the functions can be calculated on N-dimensional signals if the dimension is provided. Here the power spectral density (PSD) :math:`P_{xx}` is calculated using Welch's method (i.e. time average of the spectrogram) is shown


.. ipython:: python

    sig_2D = xr.concat([sig_1,sig_2], dim="sigs")
    psd_2D = dsp_extra.psd(sig_2D, dim="time")

.. ipython:: python
    :okwarning:

    psd_2D.plot.line(x='frequency')
    plt.loglog()
    plt.grid(which='both')
    @savefig psd.png width=4in
    plt.show()

Cross-coherence and correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same windowed FFT approach is also used to calculate the cross-spectral density :math:`P_{xy}` (using :py:func:`xrscipy.signal.csd`) and from it coherency :math:`\gamma` as

.. math::

    \gamma = \frac{\langle P_{xy}\rangle}{\sqrt{\langle P_{xx} \rangle \langle P_{yy} \rangle}}

where :math:`\langle \dots \rangle` is an average over the FFT windows, i.e. ergodicity is assumed.

.. ipython:: python

    coher_12 = dsp.coherence(sig_1, sig_2)
    coher_12[:10]

The returned :math:`\gamma` :py:class:`~xarray.DataArray` is complex (because so is :math:`P_{xy}`) and the modulus is what is more commonly called coherence and the angle is the phase shift.


.. ipython:: python
    :okwarning:

    coh = np.abs(coher_12)
    xphase = xr.apply_ufunc(np.angle, coher_12) / np.pi
    fig, axs = plt.subplots(2, 1, sharex=True)
    coh.plot(ax=axs[0])
    xphase.where(coh > 0.6).plot.line('o--', ax=axs[1])
    axs[1].set(yticks=[-1, -0.5, 0, 0.5, 1]);
    @savefig coher.png width=4in
    plt.show()


In the future more convenient wrappers returning the coherence magnitude and cross-phase might be developed.

The cross-correlation is calculated similarly as :math:`\gamma`, but with :math:`\mathcal{F}^{-1} [\langle P_*\rangle ]`, i.e. in the inverse-FFT domain. The ``lag`` coordinates are the inverse of the ``frequency`` coordinates.


.. ipython:: python
    :okwarning:

    xcorr_12 = dsp_extra.xcorrelation(sig_1, sig_2)
    xcorr_12.loc[-0.1:0.1].plot()
    plt.grid()
    @savefig xcorr.png width=4in
    plt.show()


A partially averaged counterpart to :py:func:`~xrscipy.signal.coherence` is :py:func:`~xrscipy.signal.coherogram` which uses a running average over ``nrolling`` FFT windows.
"""

from __future__ import annotations

from typing import Callable, Literal, TypeVar

import numpy as np
import scipy.signal
import scipy.signal.spectral
import xarray as xr

# noinspection PyProtectedMember
from numpy._typing import ArrayLike
from scipy.fftpack import next_fast_len
from scipy.signal._spectral_py import _spectral_helper

from xrscipy.signal.utils import get_maybe_only_dim, get_sampling_step

_FREQUENCY_DIM = "frequency"

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
        ``noverlap = np.rint(nperseg * overlap_ratio)``. Defaults to `None`.
    overlap_ratio : float, optional
        Used to calculate noverlap, if it is not specified (see above).
        Defaults to 0.5.
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
        measured in V and `fs` is measured in Hz. Defaults to 'density'.
"""

_F = TypeVar("_F", bound=Callable)


def _add2docstring_common_params(func: _F) -> _F:
    """fill-in modified docstring"""
    if hasattr(func, "__doc__"):
        func.__doc__ = func.__doc__.format(
            common_params=_DOCSTRING_COMMON_PARAMS,
            mode_param=_DOCSTRING_MODE_PARAM,
            scaling_param=_DOCSTRING_SCALING_PARAM,
        )
    return func


# noinspection PyIncorrectDocstring
@_add2docstring_common_params
def crossspectrogram(
    darray: xr.DataArray,
    other_darray: xr.DataArray,
    fs: float = None,
    seglen: float = None,
    overlap_ratio: float = 0.5,
    window: str | tuple | ArrayLike = "hann",
    nperseg: int = 256,
    noverlap: int = None,
    nfft: int = None,
    detrend: str | Callable | bool = "constant",
    return_onesided: bool = True,
    dim: str = None,
    scaling: Literal["density", "spectrum"] = "density",
    mode: str = "psd",
) -> xr.DataArray:
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
    dim = get_maybe_only_dim(darray, dim)
    if fs is None:
        dt = get_sampling_step(darray, dim)
        fs = 1.0 / dt
    else:
        dt = 1.0 / fs
    if seglen is not None:
        nperseg = int(np.rint(seglen / dt))
        nfft = next_fast_len(nperseg)
    if noverlap is None:
        noverlap = np.rint(nperseg * overlap_ratio)
    if darray is other_darray:
        d_val = od_val = darray.values
    else:
        # outer join align to ensure proper sampling
        darray, other_darray = xr.align(darray, other_darray, join="outer", copy=False)
        together = (darray, other_darray)
        if set(darray.dims) != set(other_darray.dims):
            together = xr.broadcast(*together)
        d_val, od_val = (d.values for d in together)

    # should be the same for other_darray after align
    axis = darray.get_axis_num(dim)
    # noinspection PyProtectedMember
    f, t, Pxy = _spectral_helper(
        d_val,
        od_val,
        fs,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        scaling,
        axis,
        mode,
    )
    t_0 = float(darray.coords[dim][0])
    t_axis = t + t_0
    # new dimensions and coordinates construction
    coord_darr = darray if darray.ndim >= other_darray.ndim else other_darray
    new_dims = list(coord_darr.dims)
    # frequency replaces data dim
    new_dims[new_dims.index(dim)] = _FREQUENCY_DIM
    new_dims.append(dim)  # make data dim last
    # select nearest times on other possible coordinates
    coords_ds = coord_darr.coords.to_dataset()
    coords_ds = coords_ds.sel(**{dim: t_axis, "method": "nearest"})
    coords_ds[dim] = t_axis
    coords_ds[_FREQUENCY_DIM] = f
    new_name = f"crossspectrogram_{darray.name}_{other_darray.name}"
    return xr.DataArray(Pxy, name=new_name, dims=new_dims, coords=coords_ds.coords)


# noinspection PyIncorrectDocstring
@_add2docstring_common_params
def csd(
    darray: xr.DataArray,
    other_darray: xr.DataArray,
    fs: float = None,
    seglen: float = None,
    overlap_ratio: float = 0.5,
    window: str | tuple | ArrayLike = "hann",
    nperseg: int = 256,
    noverlap: int = None,
    nfft: int = None,
    detrend: str | Callable | bool = "constant",
    return_onesided: bool = True,
    dim: str = None,
    scaling: Literal["density", "spectrum"] = "density",
    mode: str = "psd",
) -> xr.DataArray:
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
    Pxy = crossspectrogram(
        darray,
        other_darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        dim,
        scaling,
        mode,
    )
    dim = get_maybe_only_dim(darray, dim)
    Pxy = Pxy.mean(dim=dim)
    Pxy.name = f"csd_{darray.name}_{other_darray.name}"
    return Pxy


def freq2lag(darray: xr.DataArray, is_onesided: bool = False, f_dim: str = _FREQUENCY_DIM) -> xr.DataArray:
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
        ret = xr.apply_ufunc(np.fft.irfft, darray, input_core_dims=[[f_dim]], output_core_dims=[[f_dim]])
    else:
        ret = xr.apply_ufunc(np.fft.ifft, darray, input_core_dims=[[f_dim]], output_core_dims=[[f_dim]])
    ret = ret.real
    ret.name = f"ifft_{darray.name}"
    f = ret.coords[f_dim]
    df = f[1] - f[0]
    dt = 1.0 / (df * darray.shape[axis])
    lag = f / df * dt
    ret.coords["lag"] = lag
    return ret.swap_dims({f_dim: "lag"}).isel(lag=lag.argsort().values)


# noinspection PyIncorrectDocstring
@_add2docstring_common_params
def xcorrelation(
    darray: xr.DataArray,
    other_darray: xr.DataArray,
    normalize: bool = True,
    fs: float = None,
    seglen: float = None,
    overlap_ratio: float = 0.5,
    window: str | tuple | ArrayLike = "hann",
    nperseg: int = 256,
    noverlap: int = None,
    nfft: int = None,
    detrend: str | Callable | bool = "constant",
    dim: str = None,
) -> xr.DataArray:
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
    csd_d = csd(
        darray,
        other_darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided=False,
        scaling="spectrum",
        dim=dim,
        mode="psd",
    )
    xcorr = freq2lag(csd_d)
    if normalize:
        norm = 1
        for sig in (darray, other_darray):
            sig_std = (
                psd(
                    sig,
                    fs,
                    seglen,
                    overlap_ratio,
                    window,
                    nperseg,
                    noverlap,
                    nfft,
                    detrend,
                    return_onesided=False,
                    scaling="spectrum",
                    dim=dim,
                    mode="psd",
                ).mean(dim=_FREQUENCY_DIM)
                ** 0.5
            )
            norm = norm * sig_std
        xcorr /= norm
    return xcorr


# noinspection PyIncorrectDocstring
@_add2docstring_common_params
def spectrogram(
    darray: xr.DataArray,
    fs: float = None,
    seglen: float = None,
    overlap_ratio: float = 0.5,
    window: str | tuple | ArrayLike = "hann",
    nperseg: int = 256,
    noverlap: int = None,
    nfft: int = None,
    detrend: str | Callable | bool = "constant",
    return_onesided: bool = True,
    dim: str = None,
    scaling: Literal["density", "spectrum"] = "density",
    mode: str = "psd",
) -> xr.DataArray:
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
    Pxx = crossspectrogram(
        darray,
        darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        dim,
        scaling,
        mode,
    )
    Pxx.name = f"spectrogram_{darray.name}"
    return Pxx


# noinspection PyIncorrectDocstring
@_add2docstring_common_params
def psd(
    darray: xr.DataArray,
    fs: float = None,
    seglen: float = None,
    overlap_ratio: float = 0.5,
    window: str | tuple | ArrayLike = "hann",
    nperseg: int = 256,
    noverlap: int = None,
    nfft: int = None,
    detrend: str | Callable | bool = "constant",
    return_onesided: bool = True,
    dim: str = None,
    scaling: Literal["density", "spectrum"] = "density",
    mode: str = "psd",
) -> xr.DataArray:
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
    Pxx = spectrogram(
        darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        dim,
        scaling,
        mode,
    )
    dim = get_maybe_only_dim(darray, dim)
    Pxx = Pxx.mean(dim=dim)
    Pxx.name = f"psd_{darray.name}"
    return Pxx


# TODO f_res
# noinspection PyIncorrectDocstring
@_add2docstring_common_params
def coherogram(
    darray: xr.DataArray,
    other_darray: xr.DataArray,
    fs: float = None,
    seglen: float = None,
    overlap_ratio: float = 0.5,
    nrolling=8,
    window: str | tuple | ArrayLike = "hann",
    nperseg: int = 256,
    noverlap: int = None,
    nfft: int = None,
    detrend: str | Callable | bool = "constant",
    return_onesided: bool = True,
    dim: str = None,
) -> xr.DataArray:
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
    Pxx = spectrogram(
        darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        dim=dim,
    )
    Pyy = spectrogram(
        other_darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        dim=dim,
    )
    Pxy = crossspectrogram(
        darray,
        other_darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        dim=dim,
    )
    dim = get_maybe_only_dim(darray, dim)
    rol_kw = {dim: nrolling, "center": True}
    coh = Pxy.rolling(**rol_kw).mean() / (Pxx.rolling(**rol_kw).mean() * Pyy.rolling(**rol_kw).mean()) ** 0.5
    coh.dropna(dim=dim)  # drop nan from averaging edges
    coh.name = f"coherogram_{darray.name}_{other_darray.name}"
    return coh


# noinspection PyIncorrectDocstring
@_add2docstring_common_params
def coherence(
    darray: xr.DataArray,
    other_darray: xr.DataArray,
    fs: float = None,
    seglen: float = None,
    overlap_ratio: float = 0.5,
    window: str | tuple | ArrayLike = "hann",
    nperseg: int = 256,
    noverlap: int = None,
    nfft: int = None,
    detrend: str | Callable | bool = "constant",
    dim: str = None,
) -> xr.DataArray:
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
    Pxx = psd(
        darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        dim=dim,
    )
    Pyy = psd(
        other_darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        dim=dim,
    )
    Pxy = csd(
        darray,
        other_darray,
        fs,
        seglen,
        overlap_ratio,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        dim=dim,
    )
    coh = Pxy / np.sqrt(Pxx * Pyy)  # magnitude squared coherence
    coh.name = f"coherence_{darray.name}_{other_darray.name}"
    return coh


def hilbert(darray: xr.DataArray, N: int = None, dim: str = None) -> xr.DataArray:
    """
    Compute the analytic signal, using the Hilbert transform.
    The transformation is done along the selected dimension.

    Parameters
    ----------
    darray : xarray
        Signal data. Must be real.
    N : int, optional
        Number of Fourier components. Defaults to size along dim.
    dim : int, optional
        Axis along which to do the transformation.
        Uses the only dimension if darray is 1D.

    Returns
    -------
    darray : xarray
        Analytic signal of the Hilbert transform of 'darray' along selected axis.
    """
    dim = get_maybe_only_dim(darray, dim)  # TODO wrong ! this isn't a dimension, this is an axis.....
    n_orig = darray.shape[dim]
    N_unspecified = N is None
    if N_unspecified:
        N = next_fast_len(n_orig)
    return xr.apply_ufunc(
        _hilbert_wraper,
        darray,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=dict(N=N, n_orig=n_orig, N_unspecified=N_unspecified),
    )


def _hilbert_wraper(darray: xr.DataArray, N: int, n_orig: int, N_unspecified: int, axis: int = -1) -> xr.DataArray:
    """
    Hilbert wraper used to keep the signal dimension length constant
    """
    out = scipy.signal.hilbert(np.asarray(darray), N, axis=axis)

    if n_orig != N and N_unspecified:
        sl = [slice(None)] * out.ndim
        sl[axis] = slice(None, n_orig)
        out = out[sl]
    return out
