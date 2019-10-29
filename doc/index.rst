
Scipy for Xarray
================

xr-scipy is a thin wrapper of scipy for `xarray <http://xarray.pydata.org>`_ eco-system.

Many scipy functions, such as `scipy.integrate.trapz`, requires coordinate
array as an argument.
Since xarray objects possess their coordinate values in it,
there must be simpler api for such functions.

xr-scipy wraps them to use the native coordinate objects of xarray
and returns an xarray object with the computed data.
This enables more xarray-oriented data analysis.


Documentation
-------------

**Examples**

* :doc:`grad_integ`
* :doc:`fft`
* :doc:`interpolate`
* :doc:`filters`
* :doc:`spectral`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   grad_integ
   fft
   interpolate
   filters
   spectral

**Help & reference**

* :doc:`whats-new`
* :doc:`api`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & reference

   whats-new
   api

License
-------

xr-scipy is available under the open source `Apache License`__.

__ http://www.apache.org/licenses/LICENSE-2.0.html
