'''
PIV operations

Note that functions for loading PIV data are in loadpiv.py, and this 
module deals with operations on data loaded by loadpiv.py. PIV data
loaded that way is just an xarray.Dataset with:
    - variables all with dimensions (lag, T, Y, X):
        - required: u, v
        - optional: fu, fv, snr, pkh
    - coordinates:
        - required: lag, T, Y, X
        - optional: X_um, Y_um, T_s, lag_s
'''
from __future__ import annotations

import functools

import numpy as np
import xarray as xr

from abcdcs import imageop

# --------------------------------
# decorators
# --------------------------------

def _keep_attrs(f):
    """
    Keep `attrs` of the first argument (xr.Dataset) of a function
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        out = f(*args, **kwargs)
        return out.assign_attrs(args[0].attrs)
    return wrapper

def _rename_as_uv(f):
    """
    Rename the components of a PIV dataset to 'u' and 'v'.

    `components` must be a keyword argument of the function
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        out = f(*args, **kwargs)
        return (out.rename_vars(
            {c: n for c, n in zip(kwargs['components'], ['u', 'v'])})
        )
    return wrapper


# --------------------------------
# PIV operations
# --------------------------------

@_keep_attrs
@_rename_as_uv
def to_fluctuation(
    piv: xr.Dataset, *,
    components: list[str, str] = ['u', 'v'],
    d_com: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Convert a PIV dataset to a fluctuation dataset.

    If it is desired to calculate the center of mass displacement using 
    only a subset of vectors (instead of all present in input `piv`),
    pass it as `d_com` (you can also use `compute_d_com`). This is 
    useful if the piv has not been filtered.

    Resulting fluctuation components are always named as 'u' and 'v',
    regardless of specified names in the original PIV dataset.

    Parameters
    ----------
    piv : xr.Dataset, coords=('lag', 'T', 'Y', 'X')
        PIV dataset to convert to fluctuation. Must have variables
        same as those specified in `components`.
    components : list[str, str], optional
        Names of the two components in the PIV dataset, by default ['u', 'v']
    d_com : xr.Dataset, optional
        Displacement of the center of mass of the PIV dataset. If not 
        provided, it is computed by taking the mean of each PIV field.
        If provided, it must have the same components as `piv`.

    Returns
    -------
    xr.Dataset : var=('u', 'v'), coords=('lag', 'T', 'Y', 'X')
    """

    # if d_com is not provided, compute it by using all vectors in the
    # piv dataset 
    if d_com is None:
        d_com = compute_d_com(piv, components=components)

    return (piv - d_com)

@_keep_attrs
def compute_d_com(piv: xr.Dataset, *, components: list[str, str],
) -> xr.Dataset:
    return piv[components].mean(dim=('Y', 'X'))


# --------------------------------
# Filters
# --------------------------------

def _compute_r(piv, *, components: list[str, str] = ['u', 'v']) -> xr.DataArray:
    u, v = components
    return np.sqrt(piv[u]**2 + piv[v]**2)

def global_filter(
    piv: xr.Dataset,
    *,
    components: list[str, str],
    sigma: float = 3
) -> xr.Dataset:
    d = _compute_r(piv, components=components)
    return piv.where(d <= d.mean() + sigma * d.std())

def mask_filter(
    piv: xr.Dataset,
    *,
    mask: xr.DataArray,
) -> xr.Dataset:
    return imageop.Image.mask_to_keep(piv, mask=mask)

