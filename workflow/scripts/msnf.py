from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from skimage.filters import gaussian

from abcdcs import imageop, loadpiv, pivop, msnd

def main(
    piv_path: str,
    nucmask_path: str,
    npmask_path: str,
    image_path: str,
    chnames: List[str],
    bins_args: Tuple[float, float, float],
    outdir: str,
    roiuid: str
) -> None:
    image, nucmask, npmask, piv = _read_data(
        image_path, nucmask_path, npmask_path, piv_path,
        chnames=chnames,
    )

    # I. Prepare data
    image_blurred = _blur_image(image)
    mask_inner_nucleus = imageop.Mask.erode_by_disk(nucmask, radius=15)

    # filter piv by 1. nucleus mask and 2. max 4 px in each component
    piv_filtered = (piv
        .pipe(pivop.mask_filter, mask=mask_inner_nucleus)
        .pipe(pivop.mask_filter, mask=npmask)
        .pipe(pivop.global_filter, components=['u','v'], max=4)
    )

    fluc = pivop.to_fluctuation(piv_filtered, components=['u','v'])

    # II. Calculate MSND
    msnd_results = dict()
    MSND = msnd.MSND(fluc, components=('u','v'))
    
    # i. Normal
    msnd_results['normal'] = _msnd_normal(MSND)

    # ii. Eachlevel
    start = bins_args[0]
    end = bins_args[1]
    steps = int((end - start) / bins_args[2] + 1)
    bins = np.linspace(start, end, steps)
    ss = list()
    rs = list()
    for c in image.coords['C'].values:
        s, r = _msnd_eachlevel(
            MSND, byimage=image_blurred.sel(C=c), bins=bins,
        )
        s['channel'] = c
        r['channel'] = c
        ss.append(s)
        rs.append(r)
    msnd_results[f"eachlevel"] = (pd.concat(ss, ignore_index=True),
                                  pd.concat(rs, ignore_index=True))
    
    # iii. Eachlevel 2D
    msnd_results['eachlevel2d'] = _msnd_eachlevel2d(
        MSND, byimage=image_blurred, bins=bins,
    )

    # save results
    for mode, dfs in msnd_results.items():
        for df, dfname in zip(dfs, ['stats', 'indiv']):
            df['RoiUID'] = roiuid
            fpath = Path(outdir) / mode / f"{roiuid}_{dfname}.csv"
            df.to_csv(str(fpath), index=False)

def _read_data(
    image_path: str, 
    nucmask_path: str, 
    npmask_path: str, 
    piv_path: str, 
    chnames: List[str],
) -> Tuple[xr.Dataset, xr.DataArray]:
    """
    Load image and mask from a single nucleus.
    """
    image = imageop.Image.read(
        image_path, 'DataArray', chnames, squeeze=True)
    nucmask = imageop.Mask.read(nucmask_path, 'DataArray',
                                squeeze=True, drop_single_C=True)
    npmask = imageop.Mask.read(npmask_path, 'DataArray',
                               squeeze=True, drop_single_C=True)
    piv = loadpiv.read_matpiv(piv_path)
    return image, nucmask, npmask, piv


def _blur_image(image: xr.DataArray, sigma: float = 0.5
) -> xr.DataArray:
    return xr.apply_ufunc(
        gaussian, 
        image, 
        input_core_dims=[['Y', 'X']],
        output_core_dims=[['Y', 'X']],
        vectorize=True,
        kwargs={'sigma': sigma}
    )

def _msnd_normal(
    MSNDobj: msnd.MSND
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return MSNDobj.calculate()

def _msnd_eachlevel(
    MSNDobj: msnd.MSND, 
    byimage: xr.DataArray, 
    bins: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return MSNDobj.calculate('eachlevel', byimage=byimage, bins=bins)

def _msnd_eachlevel2d(
    MSNDobj: msnd.MSND, 
    byimage: xr.DataArray, 
    bins: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return MSNDobj.calculate('eachlevel2d', byimage=byimage, bins=bins)

if __name__ == '__main__':
    if 'snakemake' in globals():
        main(
            piv_path = snakemake.input.piv,
            nucmask_path = snakemake.input.nucmask,
            npmask_path = snakemake.input.npmask,
            image_path = snakemake.input.image,
            chnames = snakemake.params.chnames,
            bins_args = snakemake.config['msnd']['bins_args'],
            outdir = str(Path(snakemake.params.outdir)/snakemake.wildcards.protocol/snakemake.wildcards.ch),
            roiuid = snakemake.wildcards.RoiUID,
        )