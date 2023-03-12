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
from skimage.morphology import remove_small_objects

from cellpose import models

import typer

model = models.Cellpose(model_type='nuclei')
channels = [[0,0]]
diameter = 100.

def main(input_fpath: str, output_fpath: str, ch: int = 0):
    
    # Segment one giant image instead of individual frames
    tstack = AICSImage(input_fpath).xarray_data.isel(C=ch).squeeze()
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

    # save segmentation result to disk
    Path(output_fpath).parent.mkdir(parents=True, exist_ok=True)
    OmeTiffWriter.save(mask_tstack, output_fpath, dim_order="TYX",)
    print(f"Successfully processed and saved output for {input_fpath}.")


if __name__ == '__main__':
    typer.run(main)