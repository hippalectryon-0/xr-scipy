"""mirrors scipy.fftpack"""
from typing import Callable

from scipy import fftpack

from .docs import CDParam, DocParser
from .fft import _wrap1d, _wrapnd
from .utils import partial


def _wrapfftpack(func, freq_func, y, *coords, **kwargs):
    kwargs["s"] = kwargs.pop("shape", None)

    def new_func(y, **kwargs):
        kwargs["shape"] = kwargs.pop("s", None)
        return func(y, **kwargs)

    return _wrapnd(new_func, freq_func, y, *coords, **kwargs)


def _inject_docs(func: Callable, description: str = None, nd: bool = False) -> None:
    """inject xr docs into fft docs

    Parameters
    ----------
    func : object
        The function to modify
    description : object
    nd : object
        Whether the function acts on n-dimentional arrays
    """
    func_name = func.__name__
    doc = DocParser(fun=getattr(fftpack, func_name))

    if not nd:
        doc.replace_params(
            x=CDParam("obj", doc.get_parameter("x").description, "xarray object"),
            axis=CDParam(
                "coord",
                f"{doc.get_parameter('axis').description.split(';')[0].replace('Axis', 'Coordinate')}.\n    The coordinate must be evenly spaced.",
                "string",
            ),
        )
    else:
        doc.replace_params(
            a=CDParam("a", "Object which the transform is applied.", "xarray object"),
            axes=CDParam(
                "coord",
                "The axis along which the transform is applied.\n    The coordinate must be evenly spaced.",
                "string",
            ),
            s=CDParam("s", "the shape of the result.", "mapping from coords to size", optional=True),
        )

    doc.remove_params("overwrite_x")
    doc.reorder_params("obj", "coord")
    doc.remove_sections("Notes", "Examples")
    doc.replace_strings_returns(("ndarray", "xarray object"))

    doc.insert_description(description)
    doc.insert_see_also(f"numpy.fftpack.{func_name} : scipy.fft.{func_name} : Original scipy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = func_name


fft = partial(_wrap1d, fftpack.fft, fftpack.fftfreq)
_inject_docs(fft, description="fft(obj, coord, n=None)")

ifft = partial(_wrap1d, fftpack.ifft, fftpack.fftfreq)
_inject_docs(ifft, description="ifft(obj, coord, n=None)")

rfft = partial(_wrap1d, fftpack.rfft, fftpack.rfftfreq)
_inject_docs(rfft, description="rfft(obj, coord, n=None)")

irfft = partial(_wrap1d, fftpack.irfft, fftpack.rfftfreq)
_inject_docs(irfft, description="irfft(obj, coord, n=None)")

dct = partial(_wrap1d, fftpack.dct, fftpack.rfftfreq)
_inject_docs(dct, description="dct(obj, coord, type=2, n=None, norm=None)")

dst = partial(_wrap1d, fftpack.dst, fftpack.rfftfreq)
_inject_docs(dst, description="dst(obj, coord, type=2, n=None, norm=None)")

idct = partial(_wrap1d, fftpack.idct, fftpack.rfftfreq)
_inject_docs(idct, description="idct(obj, coord, type=2, n=None, norm=None)")

idst = partial(_wrap1d, fftpack.idst, fftpack.rfftfreq)
_inject_docs(idst, description="idst(obj, coord, type=2, n=None, norm=None)")

fftn = partial(_wrapfftpack, fftpack.fftn, fftpack.fftfreq)
_inject_docs(fftn, nd=True, description="fftn(obj, *coords, shape=None)")

ifftn = partial(_wrapfftpack, fftpack.ifftn, fftpack.fftfreq)
_inject_docs(ifftn, nd=True, description="ifftn(obj, *coords, shape=None)")
