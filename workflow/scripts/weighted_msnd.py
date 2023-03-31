from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import xarray as xr

from skimage.filters import gaussian

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="notebook", palette="Set2", style="ticks")

from abcdcs import imageop, loadpiv, msnd


THRESH_HC = 1
COMPONENTS = ['fu', 'fv']
PIXSIZE = 'meta'
CHANNEL_HC = 'miRFP670'

def main(
    image_np_path: str,
    chnames: List[str],
    background: float,
    mask_nuc_path: str,
    piv_path: str,
    weight_chname: str,
    outdir: str,
    roiuid: str,
) -> None:
    
    image_np, mask_nuc, piv = read_data(
        image_np_path, mask_nuc_path, piv_path,
        chnames=chnames, background=background
    )
    # mask_eu = segment_euchromatin(image_np, mask_nuc)
    # piv_eu = keep_euchromatin_piv(piv, mask_eu)

    # IF YOU WANT TO USE EUCHROMATIN ONLY, UNCOMMENT THE ABOVE TWO LINES
    # AND THEN USE piv_eu INSTEAD OF piv IN THE FOLLOWING LINES

    msnd = dict()
    msnd['normal'] = calc_euchromatin_msnd(piv)
    msnd['weighted'] = (
        calc_weighted_euchromatin_msnd(
            piv, 
            # _blur_image(image_np, sigma=0.5), 
            image_np,
            weight=weight_chname,
        )
    )

    # save results
    base_fpath = str(Path(outdir) / f"{roiuid}")
    for mode, dfs in msnd.items():
        for df, dfname in zip(dfs, ['stats', 'indiv']):
            fpath = Path(outdir) / mode / f"{roiuid}_{dfname}.csv"
            df.to_csv(str(fpath), index=False)


def read_data(image_path: str, mask_path: str, piv_path: str,
              chnames: List[str], background: float = 0.0,
) -> Tuple[xr.Dataset, xr.DataArray]:
    """
    Load image and mask from a single nucleus.
    """
    image = imageop.Image.read(image_path, 'Dataset', chnames,
                bitdepth=16, rescale=True, background=background, 
                squeeze=True)
    mask = imageop.Mask.read(mask_path, 'DataArray', ['NucMask'], 
                squeeze=True, drop_single_C=True)
    piv = loadpiv.read_pivresult(piv_path, 'matpiv')
    return image, mask, piv


def segment_euchromatin(image_np: xr.Dataset, mask_nuc: xr.DataArray
) -> xr.DataArray:
    """
    Mask of euchromatin: inner nuclear pixels minus heterochromatin
    """

    # remove (1) heterochromatin
    #        (2) outter pixels close by envelop     

    # find non-heterochromatic regions
    # remove heterochromatin using blurred image
    blurred = _blur_image(image_np, sigma=0.5)
    mask_non_hc = blurred[CHANNEL_HC] <= THRESH_HC
    mask_inner_nuc = imageop.Mask.shrink_mask(mask_nuc, r=10)
    mask_eu = imageop.Image.mask_to_keep(mask_non_hc, mask_inner_nuc)

    return mask_eu.rename('EuchromatinMask')


def _blur_image(image: xr.Dataset, sigma: float = 0.5
) -> xr.Dataset:
    return xr.apply_ufunc(
        gaussian, 
        image, 
        input_core_dims=[['Y', 'X']],
        output_core_dims=[['Y', 'X']],
        vectorize=True,
        kwargs={'sigma': sigma}
    )


def keep_euchromatin_piv(piv: xr.Dataset, mask: xr.DataArray
) -> xr.Dataset:
    return imageop.Image.mask_to_keep(piv, mask)


def calc_euchromatin_msnd(piv: xr.Dataset
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return msnd.calculate_msnd(piv, 
                               components=COMPONENTS, 
                               weight=None, pixsize=PIXSIZE,
                              )


def calc_weighted_euchromatin_msnd(
    piv: xr.Dataset, 
    image: xr.Dataset,
    weight: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return msnd.calculate_msnd(piv.merge(image, join='left'), 
                               components=COMPONENTS,
                               weight=weight, pixsize=PIXSIZE,
                              )


if __name__ == '__main__':
    if 'snakemake' in globals():
        main(
            image_np_path = snakemake.input.image,
            chnames = snakemake.params.chnames,
            background = snakemake.params.background,
            mask_nuc_path = snakemake.input.mask,
            piv_path = snakemake.input.piv,
            weight_chname=snakemake.config['msnd']['weight_channel'],
            outdir = str(Path(snakemake.params.outdir)/snakemake.wildcards.protocol),
            roiuid = snakemake.wildcards.RoiUID,
        )