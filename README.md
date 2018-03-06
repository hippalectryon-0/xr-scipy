# scipy for xarray
xr-scipy is a thin wrapper of scipy for the
[xarray](xarray.pydata.org) eco-system.

Many scipy functions, such as `scipy.integrate.trapz` requires coordinate
array as an argument.
xr-scipy wraps these functions to use native coordinate objects of xarray
and returns an xarray object with the computed data.
We keep other usage/options to almost the same to the original function.

This enables more xarray-oriented data analysis with scipy.

# Example

```python
In [1]: da = xr.DataArray([0, 3, 2, 4, 6], dims='x',
   ...:                       coords={'x': np.linspace(0, 1, 5)})
In [2]: da
Out[2]:
<xarray.DataArray (x: 5)>
array([0, 3, 2, 4, 6])
Coordinates:
  * x        (x) float64 0.0 0.25 0.5 0.75 1.0

In [3]: xrscipy.integrate.cumtrapz(da, coord='x')
Out[3]:
<xarray.DataArray (x: 5)>
array([0.   , 0.375, 1.   , 1.75 , 3.   ])
Coordinates:
  * x        (x) float64 0.0 0.25 0.5 0.75 1.0
```
