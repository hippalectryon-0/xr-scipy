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

def crossspectrogram(darray, other_darray, fs=None, seglen=None,
                     overlap_ratio=2, window='hann', nperseg=256,
                     noverlap=None, nfft=None, detrend='constant',
                     return_onesided=True, scaling='density', dim=None,
                     mode='psd'):
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
    f, t, Pxy = scipy.signal.spectral._spectral_helper(np.asarray(darray),
                                                       np.asarray(other_darray),
                                                       fs, window, nperseg,
                                                       noverlap, nfft, detrend,
                                                       return_onesided,
                                                       scaling, axis, mode)
    t_0 = float(darray.coords[dim][0])
    t_axis = t + t_0
    new_dims = darray.dims[:axis] + (_FREQUENCY_DIM, darray.dims[axis])
    new_coords = {dim: t_axis, _FREQUENCY_DIM: f}
    # add additional coordinates, TODO: align and add  also extra coordinates
    new_coords.update({d:darray.coords[d] for d in new_dims[:-2]})
    new_name = 'crossspectrogram_{}_{}'.format(darray.name, other_darray.name)
    return xarray.DataArray(Pxy, name=new_name,
                            dims=new_dims, coords=new_coords)

def csd(darray, other_darray, fs=None, seglen=None, overlap_ratio=2,
        window='hann', nperseg=256, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density', dim=None,
        mode='psd'):
    Pxy = crossspectrogram(darray, other_darray, fs, window, seglen,
                           overlap_ratio, nperseg, noverlap, nfft, detrend,
                           return_onesided, scaling, dim, mode)
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    Pxy = Pxy.mean(dim=dim)
    Pxy.name = 'csd_{}_{}'.format(darray.name, other_darray.name)
    return Pxy


def spectrogram(darray, fs=None, seglen=None, overlap_ratio=2, window='hann',
                nperseg=256, noverlap=None, nfft=None, detrend='constant',
                return_onesided=True, scaling='density', dim=None, mode='psd'):
    Pxx = crossspectrogram(darray, darray, fs, seglen, overlap_ratio, window,
                           nperseg, noverlap, nfft, detrend, return_onesided,
                           scaling, dim, mode)
    Pxx.name = 'spectrogram_{}'.format(darray.name)
    return Pxx


def psd(darray, fs=None, seglen=None, overlap_ratio=2, window='hann',
        nperseg=256, noverlap=None, nfft=None, detrend='constant',
        return_onesided=True, scaling='density', dim=None, mode='psd'):
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
    Pxx = spectrogram(darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft)
    Pyy = spectrogram(other_darray, fs, seglen, overlap_ratio, window, nperseg,
                      noverlap, nfft)
    Pxy = crossspectrogram(darray, other_darray, fs, seglen, overlap_ratio,
                           window, nperseg, noverlap, nfft)
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    coh = np.abs(Pxy)**2 / (Pxx * Pyy)  # magnitude squared coherence
    coh = coh.mean(dim=dim)
    coh.name = 'coherence_{}_{}'.format(darray.name, other_darray.name)
    return coh

