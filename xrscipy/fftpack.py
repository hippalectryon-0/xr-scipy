r"""mirrors scipy.fftpack"""

from typing import Callable

from scipy import fftpack

from xrscipy.docs import CDParam, DocParser
from xrscipy.fft import _wrap
from xrscipy.utils import _DAS, partial


def _wrapfftpack(func: Callable, freq_func: Callable, x: _DAS, *coords: str, **kwargs) -> _DAS:
    """wrapper around _wrap that changes s<->shape"""
    kwargs["s"] = kwargs.pop("shape", None)

    # noinspection PyMissingOrEmptyDocstring
    def s_to_shape(xx: _DAS, **kwargs) -> _DAS:
        kwargs["shape"] = kwargs.pop("s", None)
        return func(xx, **kwargs)

    return _wrap(s_to_shape, freq_func, x, _nd=True, *coords, **kwargs)


def _inject_docs(func: Callable, description: str = None, _nd: bool = False) -> None:
    """inject xr docs into fftpack docs

    Parameters
    ----------
    func : callable
        The function to modify
    description : str
    _nd : bool
        Whether the function acts on n-dimentional arrays
    """
    func_name = func.__name__
    doc = DocParser(fun=getattr(fftpack, func_name))

    if not _nd:
        doc.replace_params(
            x=CDParam("obj", doc.get_parameter("x").description, "xarray object"),
            axis=CDParam(
                "coord",
                f"{doc.get_parameter('axis').description.split(';')[0].replace('Axis', 'Coordinate')}. The coordinate must be evenly spaced.",
                "string",
            ),
        )
    else:
        doc.replace_params(
            x=CDParam("a", "Object on which the transform is applied.", "xarray object"),
            axes=CDParam(
                "coord",
                "The axis along which the transform is applied. The coordinate must be evenly spaced.",
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


def _partial_and_doc(
    f_orig: Callable, *args, description: str = "(x, coord, n=None)", wrap: Callable = _wrap, **kwargs
) -> Callable:
    """apply partial and docs"""
    f = partial(wrap, f_orig, *args, **kwargs)

    _inject_docs(f, description=f"{f_orig.__name__}{description}", _nd=wrap == _wrapfftpack)
    return f


fft = _partial_and_doc(fftpack.fft, fftpack.fftfreq)
ifft = _partial_and_doc(fftpack.ifft, fftpack.fftfreq)
rfft = _partial_and_doc(fftpack.rfft, fftpack.rfftfreq)
irfft = _partial_and_doc(fftpack.irfft, fftpack.rfftfreq)

dct = _partial_and_doc(fftpack.dct, fftpack.rfftfreq, description="(x, coord, type=2, n=None, norm=None)")
dst = _partial_and_doc(fftpack.dst, fftpack.rfftfreq, description="(x, coord, type=2, n=None, norm=None)")
idct = _partial_and_doc(fftpack.idct, fftpack.rfftfreq, description="(x, coord, type=2, n=None, norm=None)")
idst = _partial_and_doc(fftpack.idst, fftpack.rfftfreq, description="(x, coord, type=2, n=None, norm=None)")

fftn = _partial_and_doc(
    fftpack.fftn, fftpack.fftfreq, wrap=_wrapfftpack, description="(x, *coords, shape=None, axes=None)"
)
ifftn = _partial_and_doc(
    fftpack.ifftn, fftpack.fftfreq, wrap=_wrapfftpack, description="ifftn(x, *coords, shape=None, axes=None)"
)
