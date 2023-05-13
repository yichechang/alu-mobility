#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Segment the nucleus in each frame from a movie.

Usage
-----
python segment_nuclei_in_time.py <input> <output> --ch <ch>

<input> - path to tiff fille ([C]TYX)
<output - path to save segmentation result
<ch> - number of channel for segmentation (0-indexed), default 0 
'''

from pathlib import Path
import numpy as np

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from skimage.util import montage, view_as_blocks
from skimage.transform import rescale, resize
from skimage.morphology import remove_small_objects
from skimage.filters import gaussian

from cellpose import models

import typer

model = models.Cellpose(model_type='nuclei')
channels = [[0,0]]

def main(input_fpath: str, 
         output_fpath: str, 
         diameter: int = None, 
         downsample: int = None,
):
    
    original = AICSImage(input_fpath).xarray_data.isel(C=0).squeeze()

    # adjust bleaching
    avg = original.mean(dim=['X', 'Y'])
    original = original / avg

    # downsample if specified
    if downsample is not None:
        tstack = rescale(original, 1 / downsample,
                         order=0, preserve_range=True)
        diameter = diameter / downsample
    else:
        tstack = original

    # Segment one giant image instead of individual frames
    montaged = montage(tstack, fill=0)

    # cellpose segment on montaged 2D image
    masks, _, _, _ = model.eval(
	    [montaged], 
	    diameter=diameter,
	    channels=channels,
	    flow_threshold=None,
    )
    mask = masks[0]

    # remove small objects (neighbouring nuclei etc)
    # legal object needs to be >= 20% of the cropped image
    cutoff = np.ceil(0.2 * tstack.shape[-2] * tstack.shape[-1])
    mask = remove_small_objects(mask, min_size=cutoff)

    # convert montaged label matrix to tstack
    block_shape = tstack.shape[-2:]
    mask_tstack = view_as_blocks(mask, block_shape).reshape((-1, *block_shape))
    mask_tstack[mask_tstack > 0] = 1
    mask_tstack = mask_tstack[0:tstack.shape[0]]

    # upsample if needed
    if downsample is not None:
        mask_tstack = resize(
            mask_tstack, original.shape,
            order=1, preserve_range=True, 
            anti_aliasing=True, anti_aliasing_sigma=1, 
        )
        mask_tstack[mask_tstack > 0.5] = 1
        mask_tstack[mask_tstack < 1] = 0

    # save segmentation result to disk
    Path(output_fpath).parent.mkdir(parents=True, exist_ok=True)
    OmeTiffWriter.save(mask_tstack, output_fpath, dim_order="TYX",)
    print(f"Successfully processed and saved output for {input_fpath}.")


if __name__ == '__main__':
    if 'snakemake' in globals():
        main(snakemake.input[0], 
             snakemake.output[0], 
             snakemake.config['segment_nuclei_in_time']['diameter'],
             snakemake.config['segment_nuclei_in_time']['downsample'],
        )
    else:
        typer.run(main)