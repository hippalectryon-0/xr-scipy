import warnings


class UnevenSamplingWarning(Warning):
    pass
# always (not just once) show decimation warnings to see the responsible signal
warnings.filterwarnings('always', category=UnevenSamplingWarning)


def get_maybe_last_dim_axis(darray, dim=None):
    if dim is None:
        axis = darray.ndim-1
        dim = darray.dims[axis]
    else:
        axis = darray.get_axis_num(dim)
    return dim, axis


def get_sampling_step(darray, dim=None, rtol=1e-5):
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    coord = darray.coords[dim]
    dt_avg = float(coord[-1] - coord[0]) / len(coord)
    dt_first = float(coord[1] - coord[0])
    if abs(dt_avg - dt_first) > rtol * min(dt_first, dt_avg):
        # show warning at caller level to see which signal it is related to
        warnings.warn('Average sampling {} != first sampling step {}'.format(
            dt_avg, dt_first), UnevenSamplingWarning, stacklevel=2)
    return dt_avg               # should be more precise
