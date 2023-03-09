"""
Prepare dataset from a list of images and metadata files
"""

import pathlib
import re
from typing import List, Union
import warnings

import pandas as pd
import numpy as np
import typer

_MIN_IMAGE_META = ["ExpID", "Date", "PlateID", "Condition_Plate", "WellID", "FovID"]
_MIN_SAMPLE_META = ["ExpID", "Date", "PlateID", "Condition_Plate", "WellID"]

_REQUIRED_METADATA_AND_DEFAULTS = {
    "ExpID": 'empty', 
    "Date": '920215',
    "PlateID": 1,
    "Condition_Plate": 'empty', 
    "WellID": 'empty',
    "FovID": 1
}

def list_matching_files(
        rootdir: Union[str, pathlib.Path], 
        parentdirs: Union[str, List[str]],
        ext: str,
        verbose: bool = True,
    ) -> List[str]:
    """List all files with matching directory and ext names.    

    This searches recursively a root directory for files with
    a specified extension name and under any subfolders whose
    names are in the provided list of names. 

    Parameters
    ----------
    rootdir : str
    parentdirs : str or List[str]
        Name(s) for subdirectory. Needs to be exact; partial name won't
        match.
    ext : str
        Extension name (excluding `.`)
    """

    if isinstance(rootdir, str):
        rootdir = pathlib.Path(rootdir)

    # From here, parentdirs is of type List
    if isinstance(parentdirs, str):
        parentdirs = [parentdirs]

    # TODO:
    #   - Nested for each construct in a single comprehension might 
    #     be not easy to read. Consider re-writing this part.
    fpaths = [ str(x)
        for parentdir in parentdirs
        for x in rootdir.rglob(parentdir + "/**/*." + ext)
        if x.is_file()
    ]

    if len(fpaths) == 0:
        raise ValueError('No matching files found under specified dir.')
    
    if verbose:
        print(f"{len(fpaths)} matching files found, now parsing...")

    return fpaths

def parse_filepaths(
        fpaths: List[str],
        pat: str,
        nafilter: str = 'strict',
        # patch: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
    """Parse metadata from filepath for a given regexp pattern.

    Caller can specify the behavior when some files cannot be fully 
    successfully parsed. See optional argument `nafilter`.

    Parameters
    ----------
    fpaths : file paths to be parsed
    pat : pattern to be matched
    nafilter : str, optional default to 'strict'
        'strict' to raise error if some files cannot be parsed; 'loose'
        to warn and remove problematic files; 'keep' will warn but keep
        all files, resulting in the returned dataframe to have some NaN.
    patch : bool, optional default to False
        If True, will patch metadata required to uniquely identify a 
        specific FOV, if they are not available or parsed; including
        `ExpID`=`NaN`, `Date`=`1992-02-15`, `PlateID`=`1`, 
        `Condition_Plate`=`NaN`, `WellID`=`NaN`, and `FovID`=`1`.
    
    Notes
    -----
    Date is parsed and converted to '%Y-%m-%d' but stored as str, so 
    everything non-numerical stays as str.

    Raise
    -----
    ValueError
        If nafilter is set to 'strict', and some files cannot be parsed.
    """
    
    fileinfo = pd.DataFrame({"ImageFilepath": fpaths})
    
    # Extract metadata with regexp and create corresponding columns.
    # This automatically creates the pattern group names in a list 
    # which is no longer a required argument. This is done by parsing
    # the pattern string itself. 
    pat_groups = re.compile(r"P<(\w+)>").findall(pat)
    fileinfo[pat_groups] = fileinfo["ImageFilepath"].str.extract(pat)

    # Filter out non-matching files
    notmatched = len(fileinfo) - len(fileinfo.dropna(axis="rows", how="any"))
    if (nafilter == 'strict') and (notmatched > 0):
        raise ValueError(
            f"There are {notmatched} files cannot be parsed.\n")
    elif (nafilter == 'loose') and (notmatched > 0):
        warnings.warn(
            f"There are {notmatched} files cannot be parsed so removed.\n")
        fileinfo = fileinfo.dropna()
    elif (nafilter == 'keep') and (notmatched > 0):
        warnings.warn(
            f"There are {notmatched} files cannot be parsed but kept. "
            f"Resulting dataframe will have NaN's!\n")


    if verbose:
        print(f"{len(fileinfo)} files parsed successfully!")


    if 'Date' in fileinfo.columns:
        fileinfo['Date'] = pd.to_datetime(fileinfo['Date'], format='%y%m%d').dt.strftime('%Y-%m-%d')
    
    return fileinfo


def read_samplesheet(fpath: str) -> pd.DataFrame:
    return pd.read_csv(fpath)


def merge_sample_metadata(
        fileinfo: pd.DataFrame, 
        samplemeta: pd.DataFrame,
        patch: bool = True,
    ) -> pd.DataFrame:
    
    joinon = list(set(fileinfo.columns) & set(_MIN_IMAGE_META) & set(_MIN_SAMPLE_META))
    merged = fileinfo.merge(samplemeta, on=joinon)

    if patch:
        cols_to_patch = set(_MIN_IMAGE_META) - set(_REQUIRED_METADATA_AND_DEFAULTS.keys())
        for key in cols_to_patch:
            merged[key] = _REQUIRED_METADATA_AND_DEFAULTS[key]
    
    
    # reorder columns
    cols_paths = ['ImageFilepath']
    cols_imagemeta = _MIN_IMAGE_META
    cols_others = list(set(merged.columns) - set(cols_paths + cols_imagemeta))

    return merged[cols_paths + cols_imagemeta + cols_others]