from __future__ import absolute_import, division, print_function
from functools import partial

import numpy as np
from scipy import fftpack
from . import errors
from .fft import _wrap1d, _wrapnd
from .docs import DocParser


def _get_spacing(x):
    if x.ndim != 1:
        raise ValueError(
            "Coordinate for FFT should be one dimensional. "
            "Axis {} is {}-dimensional.".format(x.name, x.ndim)
        )
    dx = np.diff(x)
    mean = dx.mean()
    jitter = dx.std()

    if np.abs(jitter / mean) > 1e-4:  # heuristic value
        raise ValueError("Coordinate for FFT should be evenly spaced.")

    return mean


def _wrapfftpack(func, freq_func, y, *coords, **kwargs):
    kwargs["s"] = kwargs.pop("shape", None)

    def new_func(y, **kwargs):
        kwargs["shape"] = kwargs.pop("s", None)
        return func(y, **kwargs)

    return _wrapnd(new_func, freq_func, y, *coords, **kwargs)


def _inject_docs(func, func_name, description=None, nd=False):
    try:
        doc = DocParser(getattr(fftpack, func_name).__doc__)
    except errors.NoDocstringError:
        return

    if not nd:
        doc.replace_params(
            x="obj : xarray object\n" + doc.parameters["x"][1],
            axis="coord : string\n"
            + doc.parameters["axis"][1].split(";")[0].replace("Axis", "Coordinate")
            + ".\n    The coordinate must be evenly spaced.\n",
        )
    else:
        doc.replace_params(
            x="obj : xarray object\n" + "    Object which the transform is applied.\n",
            axes="coords : string\n"
            + "    Coordinates along which the transform is applied.\n"
            "    The coordinate must be evenly spaced.\n",
            shape="shape : mapping from coords to size, optional\n"
            "    The shape of the result.",
        )

    doc.remove_params("overwrite_x")
    doc.reorder_params("obj", "coord")

    doc.remove_sections("Notes", "Examples")

    # update return statement
    returns = doc.returns.copy()
    for key, item in doc.returns.items():
        returns[key] = [it.replace("ndarray", "xarray object") for it in item]
    doc.returns = returns

    if description is not None:
        doc.insert_description(description)

    doc.insert_see_also(
        **{
            "scipy.fftpack."
            + func_name: "scipy.fftpack."
            + func_name
            + " : Original scipy implementation\n"
        }
    )

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


fft = partial(_wrap1d, fftpack.fft, fftpack.fftfreq)
_inject_docs(fft, "fft", description="fft(obj, coord, n=None)")

ifft = partial(_wrap1d, fftpack.ifft, fftpack.fftfreq)
_inject_docs(ifft, "ifft", description="ifft(obj, coord, n=None)")

rfft = partial(_wrap1d, fftpack.rfft, fftpack.rfftfreq)
_inject_docs(rfft, "rfft", description="rfft(obj, coord, n=None)")

irfft = partial(_wrap1d, fftpack.irfft, fftpack.rfftfreq)
_inject_docs(irfft, "irfft", description="irfft(obj, coord, n=None)")

dct = partial(_wrap1d, fftpack.dct, fftpack.rfftfreq)
_inject_docs(dct, "dct", description="dct(obj, coord, type=2, n=None, norm=None)")

dst = partial(_wrap1d, fftpack.dst, fftpack.rfftfreq)
_inject_docs(dst, "dst", description="dst(obj, coord, type=2, n=None, norm=None)")

idct = partial(_wrap1d, fftpack.idct, fftpack.rfftfreq)
_inject_docs(idct, "idct", description="idct(obj, coord, type=2, n=None, norm=None)")

idst = partial(_wrap1d, fftpack.idst, fftpack.rfftfreq)
_inject_docs(idst, "idst", description="idst(obj, coord, type=2, n=None, norm=None)")

fftn = partial(_wrapfftpack, fftpack.fftn, fftpack.fftfreq)
_inject_docs(fftn, "fftn", nd=True, description="fftn(obj, *coords, shape=None)")

ifftn = partial(_wrapfftpack, fftpack.ifftn, fftpack.fftfreq)
_inject_docs(ifftn, "ifftn", nd=True, description="ifftn(obj, *coords, shape=None)")
