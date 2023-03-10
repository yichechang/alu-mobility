from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.readers.bioformats_reader import BioformatsReader
from aicsimageio.writers import OmeTiffWriter
import typer

def main(roilist_path: str, output_dir: str) -> None:
    df = pd.read_csv(roilist_path)
    crop_func = partial(crop_rois_from_a_file, output_dir=output_dir)
    df.groupby('ImagesetUID').apply(crop_func)

def crop_rois_from_a_file(df: pd.DataFrame, *, output_dir: str) -> None:
    # open the image (all rois are from the same image)
    im_path = df["ImagesetFilepath"].iloc[0]
    im = AICSImage(im_path, reader=BioformatsReader)
    scale = im.physical_pixel_sizes.X

    roi_records = df.to_dict('records')

    for roi in roi_records:
        # TODO
        #   - Why do we bother using scale/.sel() instead of just .isel()?
        ri, rf, ci, cf = scale * np.array([roi['ri'], roi['rf'], roi['ci'], roi['cf']])

        cropped = im.xarray_data.sel(Y=slice(ri,rf), X=slice(ci,cf))
        outpath = (Path(output_dir) / roi['RoiUID']).with_suffix('.ome.tif')
        OmeTiffWriter.save(cropped.data, str(outpath)) 

if __name__ == '__main__':
    typer.run(main)