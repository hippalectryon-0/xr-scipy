.. _spectral:

Spectral (FFT) analysis
-----------------------

.. ipython:: python
   :suppress:

    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    import xrscipy.signal as dsp


xr-scipy wraps some of scipy spectral analysis functions such as :py:func:`scipy.signal.spectrogram`, :py:func:`scipy.signal.csd` etc. For convenience, the ``xrscipy.signal`` namespace will be imported under the alias ``dsp``

.. ipython:: python

    import xrscipy.signal as dsp

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
    psd_2D = dsp.psd(sig_2D, dim="time")

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

    xcorr_12 = dsp.xcorrelation(sig_1, sig_2)
    xcorr_12.loc[-0.1:0.1].plot()
    plt.grid()
    @savefig xcorr.png width=4in
    plt.show()


A partially averaged counterpart to :py:func:`~xrscipy.signal.coherence` is :py:func:`~xrscipy.signal.coherogram` which uses a running average over ``nrolling`` FFT windows.



