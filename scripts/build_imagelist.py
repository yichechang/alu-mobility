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
        outputdir: str,
        nafilter: Optional[str] = 'strict',
        patch: Optional[bool] = False,
        verbose: Optional[bool] = True,
    ) -> None:
    flist = curate.list_matching_files(rootdir, parentdirs, ext, verbose)
    df = curate.parse_filepaths(flist, pat, nafilter, patch, verbose)
    df.to_csv(outputdir, quoting=2, index=False, header=True)

if __name__ == '__main__':
    typer.run(main)