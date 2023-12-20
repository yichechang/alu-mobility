"""
Prepare dataset from a list of images and metadata files
"""

from typing import Dict, List, Union
import pathlib
import re
import warnings

import pandas as pd

_MINMETA = {
    'sample': ["ExpID", "Date", "PlateID", "Condition_Plate", "WellID"],
    'imageset': ["ExpID", "Date", "PlateID", "Condition_Plate", "WellID", "FovID"],
    'roi': ["ExpID", "Date", "PlateID", "Condition_Plate", "WellID", "FovID", "RoiID"],
}

_DEFAULT_MINMETA_IMAGESET = {
    "ExpID": 'UnknownExp', 
    "Date": '1947-02-28',
    "PlateID": '1',
    "Condition_Plate": 'UnknownPlateCondition', 
    "WellID": 'UnknownWell',
    "FovID": '1'
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
        datefmt: str = None,
        nafilter: str = 'strict',
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
    
    Notes
    -----
    Date is parsed and converted to '%Y-%m-%d' but stored as str, so 
    everything non-numerical stays as str.

    Raise
    -----
    ValueError
        If nafilter is set to 'strict', and some files cannot be parsed.
    """
    
    fileinfo = pd.DataFrame({"ImagesetFilepath": fpaths})
    
    # Extract metadata with regexp and create corresponding columns.
    # This automatically creates the pattern group names in a list 
    # which is no longer a required argument. This is done by parsing
    # the pattern string itself. 
    pat_groups = re.compile(r"P<(\w+)>").findall(pat)
    fileinfo[pat_groups] = fileinfo["ImagesetFilepath"].str.extract(pat)

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

    #
    # Possible non-string values should be parsed appropriately before
    # being convereted to string.
    #
    # TODO:
    # - Make a global dict for metadata key and value type
    #
    if 'Date' in fileinfo.columns:
        fileinfo['Date'] = pd.to_datetime(fileinfo['Date'], format=datefmt).dt.strftime('%Y-%m-%d')
    for field in ['PlateID', 'FovID']:
        if field in fileinfo.columns:
            fileinfo[field] = fileinfo[field].astype(int).astype(str)
    
    return fileinfo.astype(str)


def read_samplesheet(fpath: str) -> pd.DataFrame:
    return pd.read_csv(fpath, dtype=str)


def merge_sample_metadata(
        fileinfo: pd.DataFrame, 
        samplemeta: pd.DataFrame,
        patch: bool = True,
    ) -> pd.DataFrame:

    joinon = list(set(fileinfo.columns) & set(_MINMETA['imageset']) & set(_MINMETA['sample']))
    merged = fileinfo.merge(samplemeta, on=joinon)

    #
    # Required but non-existing metadata to be patched.
    #
    if patch:
        cols_to_patch = set(_MINMETA['imageset']) - set(_DEFAULT_MINMETA_IMAGESET.keys())
        for key in cols_to_patch:
            merged[key] = _DEFAULT_MINMETA_IMAGESET[key]
    
    # UID is used to uniquely identify files irrespective where
    # are actually stored, to bypass the hurdle dealing with
    # the same underlying files have different paths on local
    # vs cluster.
    merged['ImagesetUID'] = MetadataConverter('imageset').gather(merged)
    
    # reorder columns
    cols_paths = ['ImagesetFilepath']
    cols_imagesetUID = ['ImagesetUID'] + _MINMETA['imageset']
    cols_others = list(set(merged.columns) - set(cols_paths + cols_imagesetUID))

    return merged[cols_imagesetUID + cols_paths + cols_others]

class MetadataConverter(object):
    def __init__(self, obj: str) -> None:
        if obj not in _MINMETA.keys():
            raise ValueError(f"`{obj}` metadata schema not supported.")
        
        self._obj = obj
        self._minmeta = _MINMETA[self._obj]
    
    def separate(self, 
                 metadata: Union[str, pd.Series]
        ) -> Union[Dict[str, str], pd.DataFrame]:
        """
        UID to metadata dict or dataframe
        """

        if isinstance(metadata, str):
            values = metadata.split('_')
            return {k: v for (k, v) in zip(self._minmeta, values)}
        elif isinstance(metadata, pd.Series):
            df = metadata.str.split('_', expand=True)
            return df.rename(columns={k: v for (k,v) in zip(df.columns.to_list(), self._minmeta)})

    def gather(self, 
               metadata: Union[Dict[str, str], pd.DataFrame]
        ) -> Union[str, pd.Series]:
        """
        Dict of dataframe to UID
        """

        if isinstance(metadata, Dict):
            try:
                return '_'.join([metadata[field] for field in self._minmeta])
            except TypeError:
                _metadata = self._sanitize(metadata)
                return '_'.join([_metadata[field] for field in self._minmeta])
        elif isinstance(metadata, pd.DataFrame):
            try:
                return metadata[self._minmeta].agg('_'.join, axis=1)
            except TypeError:
                _metadata = self._sanitize(metadata)
                return _metadata[self._minmeta].agg('_'.join, axis=1)

    def _sanitize(self, 
                  metadata: Union[Dict[str, str], pd.DataFrame]
        ) -> Union[Dict[str, str], pd.DataFrame]:
        """
        Convert non-string values to string.
        """
        
        warnings.warn(
            "Some non-string metadata values have been converted to "
            "string, in order to convert metadata format. Original "
            "metadata is not altered, though."
        )
        if isinstance(metadata, Dict):
            return {k: str(v) for (k,v) in metadata.items()}
        elif isinstance(metadata, pd.DataFrame):
            return metadata.astype(str) 