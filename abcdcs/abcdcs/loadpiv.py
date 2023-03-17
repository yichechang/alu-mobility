# loadpiv.py
"""
Converts PIV results stored in .mat format into xarray dataset.
"""

import numpy as np
import xarray as xr
import scipy


def read_pivresult(matfpath, method) -> xr.Dataset:
    """Convert PIV results saved in mat file to xr.Dataset

    Resulting fields are all represented in pixels and not pixels per
    lag time. Conversions to physical units and conversions from
    distance to velocity should be calculated separately, but the
    information required (pixel size, lag frames, and dt per frame)
    can be found in the dataset attributes. 

    Parameters
    ----------
    matfpath : str
        file path to `.mat` file
    method : str
        'matpiv' or 'pivlab'

    Returns
    -------
    xr.Dataset
    """

    if method == 'matpiv':
        return read_matpiv(matfpath)
    elif method == 'pivlab':
        pass
    else:
        raise ValueError(f"Parsing {method} is not implemented.")


def read_matpiv(matfpath) -> xr.Dataset:
    """Convert matpiv output from mat file to xr.Dataset

    Parameters
    ----------
    matfpath : str
        Path to the .mat file containing all matpiv output
        for a single movie in a single struct `pivresult`

    Returns
    -------
    xr.Dataset
    """
    # Load matfile content and convert to dict
    pivresult = convert_pivresult_to_dict(matfpath)

    # Assemble into xrarry Dataset for all lag times 
    ds = (
        xr.concat(
            [convert_single_lag(d) 
             for d in pivresult["data"] if d["piv"] is not None],
            dim='lag'
        )
        .sortby('lag')
        .assign_attrs(pivresult["meta"])
    )


    # Calculate coordinates with physical units
    ds = ds.assign_coords(
        {
            "X_um": 
                ("X", ds["X"].data*ds.attrs["pixelsize"]),
            "Y_um": 
                ("Y", ds["Y"].data*ds.attrs["pixelsize"]),
            "T_s": 
                ("T", ds["T"].data*ds.attrs["dt"]),
            "lag_s": 
                ("lag", ds["lag"].data*ds.attrs["dt"]),
        }
    )

    # Remove duplicated lag times if exist
    _, idx_unique = np.unique(ds['lag'], return_index=True)
    ds = ds.isel(lag=idx_unique)

    return ds


def convert_pivresult_to_dict(matfpath) -> dict:
    # Load data and retrieve the single struct,
    # then convert to dict
    mat = scipy.io.loadmat(
        matfpath,
        struct_as_record=True
    )
    pivresult = mat["pivresult"][0,0]
    pivresult = struct_to_dict(pivresult, vec1d=True)

    return pivresult


def convert_single_lag(data) -> xr.Dataset:
    ds = (xr.concat([convert_single_piv(d) for d in data["piv"]],
                    dim='T')
            .sortby('T')
            .assign_coords(lag = np.array(data["lag"], dtype='int_'))
    )
    return ds


def convert_single_piv(data) -> xr.Dataset:
    """Core function converting single PIV fields to xr.Dataset.

    Indexing for pixel location and time have all been converted to 
    0-based.

    Parameters
    ----------
    data : dict
        Dict converted from matlab struct containing PIV 
        results.
        Keys: 
            'piv' - dict with keys 'x', 'y', 'u', 'v', ... etc
            'frames' - list of frame numbers for the 1st and 
                       2nd frames

    Returns
    -------
    xr.Dataset
        PIV results saved without metadata.
    """
    #
    coords = {
        "X": (data["data"]["x"][0,:] - 1).astype('int_'),
        "Y": (data["data"]["y"][:,0] - 1).astype('int_'),
        "T": (data["frames"][0] - 1).astype('int_'),
    }

    coord_var_names = ["Y", "X"]
    data_var_names = ["u", "v", "fu", "fv", "snr", "pkh"]
    data_vars = {
        data_var_name: (coord_var_names, data["data"][data_var_name].astype('float_'))
        for data_var_name in data_var_names
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return ds

# ----------------------------------------------------------------------
#   helper functions
# ----------------------------------------------------------------------

def struct_to_dict(struct, vec1d=False) -> dict:
    """Convert matlab struct into a python dict.
    
    Matlab struct represented as a numpy structured array will be
    converted to a python dict appropriately, with scalars and
    strings (as values) represented as numbers and strings instead of
    numpy array as origianlly are. Vectors can also be optionally
    represented as 1D numpy array instead of 2D.
    
    Parameters
    ----------
    struct : numpy.ndarray structured array
        A single matlab struct read in by scipy.io.loadmat.
        If multiple structs are to be processed, make separate
        calls of this funtion for each of the structs.
        
    vec1d : bool (default to True)
        If vectors should be represented as 1D numpy array.
    
    
    Return
    ------
    dict representation of the input matlab struct.
    """
    
    # Input cannot be a struct array with more than one structs
    if _is_struct_array(struct) and not _has_only_one_struct(struct):
        raise ValueError(
            "Multiple structs instead of one are passed.")
    

    out = dict()
    keys = struct.dtype.names
    for k in keys:
        v = struct[k]
        
        # Value can be structs, a single struct, not struct
        # - Turn structs to a list of structs
        # - Turn struct to a dict
        # - Unwrap non-struct values
        if _is_struct_array(v) and not _has_only_one_struct(v):
            out[k] = [struct_to_dict(s, vec1d) for s in v.squeeze()]
        elif _is_struct_array(v) and _has_only_one_struct(v):
            out[k] = struct_to_dict(v[0,0], vec1d)
        else:
            out[k] = _unwrap_value(v, vec1d)
        
    return out

def _unwrap_value(val, tovec=False):
    """Remove redundant dimensions when appropriate.
    
    Scalars and strings stored in a numpy array will be unwrapped 
    to simply a python int/float or a string. Vectors can optionally
    be downgraded to 1-D numpy array. Others will be kept the same.
    """
    # scalars and strings
    if val.size == 1:
        return val.item()

    # remainders are only vectors, matrices, or higher-dim ndarray
    # vectors or vector-likes
    elif tovec and np.any(np.array(val.shape) == 1):
        return val.squeeze()
    
    else:
        return val

def _is_struct_array(val):
    return val.dtype.names is not None

def _has_only_one_struct(structarr):
    return structarr.size == 1