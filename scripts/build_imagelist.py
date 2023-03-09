#!/usr/bin/env python
# -*- coding: utf-8 -*-

# build_imagelist.py

'''
List files from specified dirs and parse metadata from filename.
'''

from typing import Optional, List
import typer

from abcdcs import curate


def main(
        rootdir: str, 
        parentdirs: List[str], 
        ext: str, 
        pat: str, 
        samplesheet: str,
        outputdir: str,
        nafilter: Optional[str] = 'strict',
        patch: Optional[bool] = True,
        verbose: Optional[bool] = True,
    ) -> None:
    """List files, parse filenames, and merge with samplesheet metadata.

    Parameters
    ----------
    rootdir : str
        Where to look for files.
    parentdirs : List[str]
        Subfolder names to limit file search.
    ext : str
        extension name to limit file search. excluding `.`.
    pat : str
        regex pattern for parsing filepaths.
    samplesheet : str
        path to samplesheet.
    outputdir : str
        path to output directory.
    """
    flist = curate.list_matching_files(rootdir, parentdirs, ext, verbose)
    parsed = curate.parse_filepaths(flist, pat, nafilter, verbose)
    samplesheet = curate.read_samplesheet(samplesheet)
    merged = curate.merge_sample_metadata(parsed, samplesheet, patch)

    # save merged dataframe as csv with non-numerical values quoted by
    # double-quotes (quoting=2).
    merged.to_csv(outputdir, quoting=2, index=False, header=True)

if __name__ == '__main__':
    typer.run(main)