r""".. _filters:

Digital filters
---------------

.. ipython:: python
   :suppress:

    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import xrscipy.signal as dsp


``xr-scipy`` wraps SciPy functions for digital signal processing. Wrappers for convenient functions such as :py:func:`scipy.signal.decimate`, :py:func:`scipy.signal.savgol_filter`, and :py:func:`scipy.signal.sosfilt` are provided.
For convenience, the ``xrscipy.signal`` namespace will be imported under the alias ``dsp``:

.. ipython:: python

    import xrscipy.signal as dsp


Decimation
^^^^^^^^^^


To demonstrate basic functionality of :py:func:`~xrscipy.signal.decimate`, let's create a simple example DataArray:

.. ipython:: python

    arr = xr.DataArray(np.sin(np.linspace(0, np.pi * 4, 300)) ** 2,
                       dims=('x'), coords={'x': np.linspace(0, np.pi * 4, 300)})
    arr

Our :py:func:`~xrscipy.signal.decimate` takes an xarray object
(possibly high dimensional) and a dimension name (if not 1D)
along which the signal should be decimated.


.. ipython:: python

    arr_decimated = dsp.decimate(arr, q=20)
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

    arr_savgol2 = dsp.savgol_filter(arr_noisy, 0.5, 2)
    arr_savgol5 = dsp.savgol_filter(arr_noisy, 1.0, 2)
    arr_savgol2
    arr_savgol5

The return type is also a DataArray with coordinates.

.. ipython:: python
    :okwarning:

    arr.plot(label='arr', color='r')
    arr_noisy.plot.line('s', label='noisy', color='b')
    arr_savgol2.plot(label='quadratic fit on 1 unit of x', color='k', linewidth=2)
    arr_savgol5.plot.line('--',label='quadratic fit on 2 units of x', linewidth=2, color='lime')
    plt.legend()
    @savefig savgol_signal.png width=4in
    plt.show()

The other options (polynomial and derivative order) are the same as for :py:func:`scipy.signal.savgol_filter`, see :py:func:`~xrscipy.signal.savgol_filter` for details.


Second-order sections (SOS) filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:func:`~xrscipy.signal.sosfilt` function provides a wrapper for :py:func:`scipy.signal.sosfilt` that applies a digital IIR filter in second-order sections format. This format is designed to minimize numerical precision errors for high-order filters by cascading second-order filter sections. The filter is defined by an array of second-order filter coefficients in the form (n_sections, 6), where each row corresponds to a second-order section with the first three columns providing the numerator coefficients and the last three providing the denominator coefficients.

For convenience, SOS filters can be easily created using :py:func:`scipy.signal.butter`, :py:func:`scipy.signal.cheby1`, :py:func:`scipy.signal.cheby2`, :py:func:`scipy.signal.ellip`, or :py:func:`scipy.signal.bessel` with ``output='sos'``.

To demonstrate basic functionality of :py:func:`~xrscipy.signal.sosfilt` and :py:func:`~xrscipy.signal.sosfiltfilt`, let's create a simple example with a 4th-order Butterworth low-pass filter:

.. ipython:: python
    :okwarning:

    t = np.linspace(0, 1, 1000)
    sig = xr.DataArray(np.sin(16*t) + np.random.normal(0, 0.1, t.size),
                       coords=[('time', t)], name='signal')

    from scipy.signal import butter
    # Create a 8th-order Butterworth low-pass filter with cuttoff 20Hz
    sos = butter(8, 20, btype='low', fs=1/np.mean(np.diff(t)), output='sos')

    # Apply the SOS filter along the 'time' dimension
    filtered = dsp.sosfilt(sos, sig, dim='time')
    # Apply the zero-phase SOS filter along the 'time' dimension
    filtered_zero_phase = dsp.sosfiltfilt(sos, sig, dim='time')

    sig.plot(label='noisy', alpha=0.7)
    filtered.plot(label='sosfilt', linewidth=2)
    filtered_zero_phase.plot(label='sosfiltfilt', linewidth=2, color="red")
    plt.legend()
    @savefig sosfilt_signal.png width=4in
    plt.show()
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import decimate as sp_decimate
from scipy.signal import savgol_filter as sp_savgol_filter
from scipy.signal import sosfilt as sp_sosfilt
from scipy.signal import sosfiltfilt as sp_sosfiltfilt

import xrscipy.docs as docs
from xrscipy.docs import CDParam
from xrscipy.signal.utils import get_maybe_only_dim, get_sampling_step


def _savgol_filter(
    darray: xr.DataArray,
    window_length: float,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    dim: str = None,
    mode: str = "interp",
    cval: float = 0.0,
) -> xr.DataArray:
    """Apply a Savitzky-Golay filter to an array."""
    dim = get_maybe_only_dim(darray, dim)

    # Convert window_length from coordinate units to samples
    dt = get_sampling_step(darray, dim)
    window_length_samples = int(np.rint(window_length / dt))

    # Ensure window_length is odd and positive
    if window_length_samples % 2 == 0:
        window_length_samples += 1
    window_length_samples = max(window_length_samples, 3)

    # Ensure polyorder is less than window_length
    polyorder = min(polyorder, window_length_samples - 1)

    # Apply the filter
    result = xr.apply_ufunc(
        sp_savgol_filter,
        darray,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=dict(
            window_length=window_length_samples,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
            axis=-1,
            mode=mode,
            cval=cval,
        ),
        exclude_dims={dim},
    )

    # Reorder dimensions to match input order
    result = result.transpose(*darray.dims)

    # Preserve the original coordinates along the filtered dimension
    if dim in darray.coords:
        result = result.assign_coords({dim: darray.coords[dim]})

    result.name = f"savgol_filtered_{darray.name}" if darray.name else "savgol_filtered"
    return result


def _decimate(
    darray: xr.DataArray,
    q: int = None,
    target_fs: float = None,
    n: int = None,
    ftype: str = "iir",
    zero_phase: bool = True,
    dim: str = None,
) -> xr.DataArray:
    """Downsample the signal after applying an anti-aliasing filter."""
    dim = get_maybe_only_dim(darray, dim)

    # Calculate q from target_fs if needed
    if q is None and target_fs is not None:
        dt = get_sampling_step(darray, dim)
        current_fs = 1.0 / dt
        q = int(np.rint(current_fs / target_fs))
    elif q is None:
        raise ValueError("Either 'q' or 'target_fs' must be specified")

    # For short signals, use a lower order filter or switch to FIR
    if darray.sizes[dim] < 30:  # Empirical threshold
        if n is None:
            n = min(4, q * 2)  # Use a smaller order for short signals
        if ftype == "iir":
            ftype = "fir"  # FIR filters are more stable for short signals

    # Apply the decimation
    result = xr.apply_ufunc(
        sp_decimate,
        darray,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=dict(
            q=q,
            n=n,
            ftype=ftype,
            axis=-1,
            zero_phase=zero_phase,
        ),
        exclude_dims={dim},
    )

    # Update coordinates to reflect decimation
    coord = darray.coords[dim]
    new_coord = coord[::q]
    result = result.assign_coords({dim: new_coord})

    result.name = f"decimated_{darray.name}" if darray.name else "decimated"
    return result


def _sosfilt(
    sos: np.ndarray,
    darray: xr.DataArray,
    dim: str = None,
    zi: np.ndarray = None,
) -> xr.DataArray | tuple[xr.DataArray, np.ndarray]:
    """Apply a digital IIR filter in cascaded second-order sections."""
    dim = get_maybe_only_dim(darray, dim)
    axis = darray.get_axis_num(dim)

    if zi is not None:
        # When zi is provided, the result is a tuple (output, final conditions)
        result_data, zf = sp_sosfilt(sos, darray.values, axis=axis, zi=zi)

        # Create result DataArray
        result = xr.DataArray(
            result_data,
            dims=darray.dims,
            coords=darray.coords,
            name=f"sosfilt_{darray.name}" if darray.name else "sosfilt",
        )

        # Preserve the original coordinates along the filtered dimension
        if dim in darray.coords:
            result = result.assign_coords({dim: darray.coords[dim]})

        return result, zf
    else:
        # Apply sosfilt without initial conditions
        result = xr.apply_ufunc(
            sp_sosfilt,
            sos,
            darray,
            input_core_dims=[[], [dim]],  # sos: no dimensions, darray: [dim]
            output_core_dims=[[dim]],  # Output: [dim]
            kwargs=dict(
                axis=-1,  # Filter along the last axis
            ),
            exclude_dims={dim},
        )

        # Reorder dimensions to match input order (apply_ufunc should maintain this)
        result = result.transpose(*darray.dims)

        # Preserve the original coordinates along the filtered dimension
        if dim in darray.coords:
            result = result.assign_coords({dim: darray.coords[dim]})

        result.name = f"sosfilt_{darray.name}" if darray.name else "sosfilt"
        return result


def _sosfiltfilt(
    sos: np.ndarray,
    darray: xr.DataArray,
    dim: str = None,
    padtype: str = "odd",
    padlen: int = None,
) -> xr.DataArray:
    """Apply a forward-backward digital filter using cascaded second-order sections."""
    dim = get_maybe_only_dim(darray, dim)

    # Apply sosfiltfilt using apply_ufunc
    result = xr.apply_ufunc(
        sp_sosfiltfilt,
        sos,
        darray,
        input_core_dims=[[], [dim]],  # sos: no dimensions, darray: [dim]
        output_core_dims=[[dim]],  # Output: [dim]
        kwargs=dict(
            axis=-1,  # Filter along the last axis
            padtype=padtype,
            padlen=padlen,
        ),
        exclude_dims={dim},
    )

    # Reorder dimensions to match input order
    result = result.transpose(*darray.dims)

    # Preserve the original coordinates along the filtered dimension
    if dim in darray.coords:
        result = result.assign_coords({dim: darray.coords[dim]})

    result.name = f"sosfiltfilt_{darray.name}" if darray.name else "sosfiltfilt"
    return result


def _inject_docs_savgol(func) -> None:
    """Inject xr docs into savgol_filter docs."""
    doc = docs.DocParser(fun=sp_savgol_filter)

    doc.replace_params(
        x=CDParam("darray", "The data to be filtered.", "xarray.DataArray"),
        window_length=CDParam(
            "window_length",
            "The length of the filter window in the units of the specified dimension. This will be converted to the number of samples based on the coordinate spacing.",
            "float",
        ),
        axis=CDParam(
            "dim",
            "The dimension of the array `darray` along which the filter is to be applied. Default is the only dimension if 1D, otherwise must be specified.",
            "str, optional",
        ),
    )

    doc.replace_strings_returns(("ndarray", "xarray.DataArray"))
    doc.replace_strings_description(("axis", "dim"))

    doc.insert_see_also("scipy.signal.savgol_filter : Original scipy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = "savgol_filter"


def _inject_docs_decimate(func) -> None:
    """Inject xr docs into decimate docs."""
    doc = docs.DocParser(fun=sp_decimate)

    doc.replace_params(
        x=CDParam("darray", "The input signal made up of equidistant samples.", "xarray.DataArray"),
        q=CDParam(
            "q",
            "The downsampling factor, which is a postive integer. If not provided, will be calculated from target_fs.",
            "int, optional",
        ),
        axis=CDParam("dim", "The dimension along which to decimate. Uses the only dimension if 1D.", "str, optional"),
    )

    # Add target_fs parameter which is xrscipy-specific
    target_fs_param = CDParam(
        "target_fs",
        "The target sampling frequency. If provided, q will be calculated as q = np.rint(current_fs / target_fs).",
        "float, optional",
    )

    # Insert the target_fs parameter
    doc.parsed_doc.params.append(target_fs_param)

    doc.replace_strings_returns(("ndarray", "xarray.DataArray"))
    doc.replace_strings_description(("axis", "dim"))

    doc.insert_see_also("scipy.signal.decimate : Original scipy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = "decimate"


def _inject_docs_sosfilt(func) -> None:
    """Inject xr docs into sosfilt docs."""
    doc = docs.DocParser(fun=sp_sosfilt)

    doc.replace_params(
        x=CDParam("darray", "The data to be filtered.", "xarray.DataArray"),
        axis=CDParam(
            "dim",
            "The dimension of the array `darray` along which the filter is to be applied. Default is the only dimension if 1D, otherwise must be specified.",
            "str, optional",
        ),
    )

    doc.replace_strings_returns(("ndarray", "xarray.DataArray"))
    doc.replace_strings_description(("axis", "dim"))

    doc.insert_see_also("scipy.signal.sosfilt : Original scipy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = "sosfilt"


def _inject_docs_sosfiltfilt(func) -> None:
    """Inject xr docs into sosfiltfilt docs."""
    doc = docs.DocParser(fun=sp_sosfiltfilt)

    doc.replace_params(
        x=CDParam("darray", "The data to be filtered.", "xarray.DataArray"),
        axis=CDParam(
            "dim",
            "The dimension of the array `darray` along which the filter is to be applied. Default is the only dimension if 1D, otherwise must be specified.",
            "str, optional",
        ),
    )

    doc.replace_strings_returns(("ndarray", "xarray.DataArray"))
    doc.replace_strings_description(("axis", "dim"))

    doc.insert_see_also("scipy.signal.sosfiltfilt : Original scipy implementation")

    # inject
    func.__doc__ = str(doc)
    func.__name__ = "sosfiltfilt"


savgol_filter = _savgol_filter
_inject_docs_savgol(savgol_filter)
decimate = _decimate
_inject_docs_decimate(decimate)
sosfilt = _sosfilt
_inject_docs_sosfilt(sosfilt)
sosfiltfilt = _sosfiltfilt
_inject_docs_sosfiltfilt(sosfiltfilt)
