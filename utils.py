
def get_maybe_last_dim_axis(darray, dim=None):
    if dim is None:
        axis = darray.ndim-1
        dim = darray.dims[axis]
    else:
        axis = darray.get_axis_num(dim)
    return dim, axis


def get_sampling_step(darray, dim=None):
    dim, axis = get_maybe_last_dim_axis(darray, dim)
    coord = darray.coords[dim]
    return float(coord[1] - coord[0])
