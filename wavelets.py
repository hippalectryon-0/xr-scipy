import pycwt
import numpy as np
import xarray
from .utils import get_maybe_last_dim_axis, get_sampling_step

def cwt(darray, **kwargs):
    if darray.ndim != 1:
        raise ValueError('Only 1D signals are supported')
    dim, axis = get_maybe_last_dim_axis(darray, None)
    dt = get_sampling_step(darray, dim)
    W, sj, freqs, coi, sig_ft, ft_freqs = pycwt.cwt(
        np.asarray(darray), dt, **kwargs)
    ds = xarray.Dataset(data_vars={'cwt': (['frequency', dim], W),
                                   'coi': ([dim], coi),
                                   'fft': (['fft_freqquency'], sig_ft)},
                        coords={dim: darray.coords[dim],
                                'frequency': freqs,
                                'scale': (['frequency'], sj),
                                'fft_freqquency': ft_freqs}
    )
    return ds


def wct(darray, other_darray, **kwargs):
    if darray.ndim != 1 or other_darray.ndim != 1:
        raise ValueError('Only 1D signals are supported')
    darray, other_darray = xarray.align(darray, other_darray)
    dim, axis = get_maybe_last_dim_axis(darray, None)
    dt = get_sampling_step(darray, dim)
    # significance estimate calculation is costly
    kwargs.setdefault('sig', False)
    WCT, aWCT, coi, freqs, sig = pycwt.wct(darray, other_darray, dt, **kwargs)
    ds = xarray.Dataset(data_vars={'coherence': (['frequency', dim], WCT),
                                   'phase': (['frequency', dim], aWCT),
                                   'coi': ([dim], coi)

                               },
                        coords={dim: darray.coords[dim],
                                'frequency': freqs,
                            },
                        attrs={'significance': sig}
    )
    return ds

