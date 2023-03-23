#!/usr/bin/env python
# -*- coding: utf-8 -*-

# build_imagelist.py

'''
List files from specified dirs and parse metadata from filename.
'''

from typing import Dict, Optional, List
import typer

from abcdcs import curate


def main(
        rootdir: str, 
        parentdirs: List[str], 
        ext: str, 
        pat: str, 
        samplesheet: str,
        outputpath: str,
        datefmt: Optional[str] = None,
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
    datefmt : str
        format for parsing date. e.g., '%y%m%d'
    """
    flist = curate.list_matching_files(rootdir, parentdirs, ext, verbose)
    parsed = curate.parse_filepaths(flist, pat, datefmt, nafilter, verbose)
    samplesheet = curate.read_samplesheet(samplesheet)
    merged = curate.merge_sample_metadata(parsed, samplesheet, patch)

    # save merged dataframe as csv with non-numerical values quoted by
    # double-quotes (quoting=2).
    merged.to_csv(outputpath, quoting=2, index=False, header=True)

def get_pepfile_dict(pepfile_path: str) -> Dict:
    import yaml
    with open(pepfile_path, 'rb') as f:  
        conf = yaml.load(f, Loader=yaml.CLoader)
    return conf

if __name__ == '__main__':
    if 'snakemake' in globals():
        exptype = snakemake.config['input']['experiment_type']
        pep = get_pepfile_dict(snakemake.config['input']['pepfile_path'])
        main(snakemake.config['input']['raw']['base_dir'], 
             snakemake.config['input']['raw']['subdir_name'],
             snakemake.config['input']['raw']['ext'],
             pep['experiments'][exptype]['path_pattern'],
             snakemake.config['input']['samplesheet_path'],
             snakemake.output[0],
             pep['experiments'][exptype]['datefmt'],
             snakemake.config['build_imagelist']['nafilter'],
             snakemake.config['build_imagelist']['patch'],
             snakemake.config['build_imagelist']['verbose'],
        )
    else:
        typer.run(main)