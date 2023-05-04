[![Build Status](https://travis-ci.org/fujiisoup/xr-scipy.svg?branch=master)](https://travis-ci.org/fujiisoup/xr-scipy)
[![Documentation Status](https://readthedocs.org/projects/xr-scipy/badge/?version=latest)](http://xr-scipy.readthedocs.io/en/latest/?badge=latest)

# scipy for xarray

xr-scipy is a thin wrapper of scipy for the
[xarray](xarray.pydata.org) eco-system.

Many scipy functions, such as `scipy.integrate.trapz` requires coordinate
array as an argument.
xr-scipy wraps these functions to use native coordinate objects of xarray
and returns an xarray object with the computed data.
This enables more xarray-oriented data analysis with scipy.

Other usage/options are kept almost the same as the original scipy function.

# Example

```python
In[1]: da = xr.DataArray([0, 3, 2, 4, 6], dims='x',
                         ...:                       coords = {'x': np.linspace(0, 1, 5)})
In[2]: da
Out[2]:
< xarray.DataArray(x: 5) >
array([0, 3, 2, 4, 6])
Coordinates:
*x(x)
float64
0.0
0.25
0.5
0.75
1.0

In[3]: xrscipy.integrate.cumtrapz(da, coord='x')
Out[3]:
< xarray.DataArray(x: 5) >
array([0., 0.375, 1., 1.75, 3.])
Coordinates:
*x(x)
float64
0.0
0.25
0.5
0.75
1.0
```
