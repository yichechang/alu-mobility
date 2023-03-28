from pathlib import Path
from functools import partial
import re

import numpy as np
import pandas as pd

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

import typer


def main(
        roilist_path: str, 
        common_pattern: str, platform_prefix: str,
        output_dir: str
    ) -> None:
    df = pd.read_csv(roilist_path)
    crop_func = partial(crop_rois_from_a_file, 
                        common_pattern=common_pattern,
                        platform_prefix=platform_prefix,
                        output_dir=output_dir)
    df.groupby('ImagesetUID').apply(crop_func)


def crop_rois_from_a_file(
        df: pd.DataFrame, *, 
        common_pattern: str, platform_prefix: str, 
        output_dir: str
    ) -> None:
    # open the image (all rois are from the same image)
    im_path_original = df["ImagesetFilepath"].iloc[0]
    im_path = update_absolute_path(im_path_original, 
                                   common_pattern, platform_prefix)
    im = AICSImage(im_path)

    roi_records = df.to_dict('records')

    for roi in roi_records:
        # TODO
        #   - Validate that we don't rely on actual spatial coordinates
        ri, rf, ci, cf = np.array([roi['ri'], roi['rf'], roi['ci'], roi['cf']]).astype(int)

        cropped = im.xarray_data.isel(Y=range(ri,rf+1), X=range(ci,cf+1))
        outpath = (Path(output_dir) / roi['RoiUID']).with_suffix('.ome.tif')
        OmeTiffWriter.save(cropped.data, str(outpath)) 


def update_absolute_path(
        original_path: str, 
        common_pattern: str, 
        platform_prefix: str
) -> str:
    """
    Update the platform-specific parts of a path.

    This allows us to run parts of the workflow on local, and then 
    continue on clusters. 
    
    While input files are organized with a common hierarchy between 
    local and cluster, they would have platform-specific prefixes. Thus, 
    whereas most rules use wildcards and snakemake should figure out 
    where to look for files based on relative paths in the analysis 
    directory (i.e., the output directory), certain rules relying on 
    absolute paths written in a csv file from a previous step, would 
    need to first fix those prefixes.
    
    We set the correct *prefixes* and the *pattern of common parts* in 
    the snakemake config file, via `['input']['raw']['base_dir']` and 
    `['input']['raw']['common_pattern']`, repectively. 

    For example, on local machine it can be:

    `~/.../phd-data/microscopy/2023a/y999/WellC03_Fov001.nd2`;
    `^^^^^^^^^^^^^^`

    and on clusters:
    
    `/tigress/<user>/data/microscopy/2023a/y999/WellC03_Fov001.nd2`
    `^^^^^^^^^^^^^^^^^^^^`
    """

    # use r"..." and no need to escape for forward slash
    pattern = r"(?P<platform>.*/)" + r"(?P<common>" + common_pattern + r")"
    matched = re.compile(pattern).match(original_path)
    common = matched.group('common')
    return str(Path(platform_prefix) / common)


if __name__ == '__main__':
    if 'snakemake' in globals():
        main(snakemake.input[0], 
             snakemake.config['input']['raw']['common_pattern'],
             snakemake.config['input']['raw']['base_dir'],
             snakemake.params['outdir'])
    else:
        typer.run(main)
