from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(context="paper", palette="Set2", style="ticks")

from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter

from abcdcs import imageop

# Parameters used in this script. If need to configure from the
# workflow level, these can be moved to snakemake config file.

# possible values: 'zscore', 'pclip'
NORM_METHOD_NUC = 'zscore'
# depends on NORM_METHOD_NUC
TRHESH_NP = -1.0
# possible values: 'zscore', 'pclip'
NORM_METHOD_NP = 'zscore'
# using first N frames for plotting only
NFRAMES = 5 
# number of bins in H2B signal
NLEVELS = 15
# range for binning, in normalized H2B signal
LEVEL_RANGE = (-2, 4) 

def main():
    pass


def process_one_nucleus(
    image_path: str, chnames: List[str], background: float,
    mask_path: str, 
    output_paths: List[str], outdir: str, roiuid: str,
) -> None:
    """
    main worker to remove nucleoli and then normalize the image
    within each channel at each timepoint.
    """

    image, mask = read_image_and_mask(
        image_path, mask_path, 
        chnames=chnames, background=background)
    
    normalized_nuc, normalized_np = normalize_nuc_np(
       image, mask, thresh_nucleoli=-TRHESH_NP)
    
    counts, level_edges, level_means, level_stds = (
        calc_mean_by_level(
            normalized_np.isel(T=range(NFRAMES)),
            by='miRFP670', bins=NLEVELS, range=LEVEL_RANGE)
    )
    
    figs = dict()
    figs[f"NormHist{NFRAMES}fr"] = (
        plot_normalized_image_histogram(
            normalized_np.isel(T=range(NFRAMES))
        )
    )
    figs[f"LvlMean{NFRAMES}fr"] = (
        plot_level_means(
            counts, level_means, level_stds, level='miRFP670'
        )
    )
    

    # Save the results
    output_images = [normalized_nuc, normalized_np]
    for im, outpath in zip(output_images, output_paths):
        OmeTiffWriter.save(im.to_array(dim='C', name='normalized')
                             .transpose('T','C','Y','X').data, 
                           outpath, 
                           dim_order="TCYX")
    
    for figname, fig in figs.items():
        fig.savefig(Path(outdir)/f"{roiuid}_{figname}.png" , dpi=300)


def read_image_and_mask(image_path: str, mask_path: str,
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
    return image, mask


def normalize_nuc_np(image: xr.Dataset, nucmask: xr.DataArray,
                          thresh_nucleoli: float, 
) -> xr.Dataset:
    """
    Normalize each channel for nucleoplasm pixels, plane-by-plane.
    """
    rescaled_nuc = (image
        .pipe(imageop.Image.mask_to_keep, mask=nucmask)
        .pipe(imageop.Image.normalize, method=NORM_METHOD_NUC)
    )
    mask_nucleoli = ((rescaled_nuc['miRFP670'] > thresh_nucleoli)
        .rename('NucleoliMask')
    )
    rescaled_np = (image
        .pipe(imageop.Image.mask_to_keep, mask=mask_nucleoli)
        .pipe(imageop.Image.normalize, method=NORM_METHOD_NP)
    )
    return rescaled_nuc, rescaled_np

def calc_mean_by_level(ds: xr.Dataset, by='miRFP670', **kwargs):
    """
    Create bins with equal width in intensity space, then summarize
    pixel values in each bin (level).

    Parameters
    ----------
    ds : xr.Dataset, with each data variable mapping to a channel
    ch_bin : str, channel name for binning
    ch_mean : str, channel name for calculating mean within each bin
    kwargs : dict, to pass to `np.histogram()`
    """
    counts, level_edges = np.histogram(ds[by], **kwargs)

    # these are returned as Dataset
    grouped = ds.groupby_bins(by, bins=level_edges)
    level_means = grouped.mean()
    level_stds = grouped.std()

    return counts, level_edges, level_means, level_stds


def plot_normalized_image_histogram(image: xr.Dataset) -> sns.FacetGrid:
    df = image.to_dataframe()
    chnames = df.columns.to_list()
    
    g = (df
        .reset_index()
        .melt(id_vars=['T'], value_vars=chnames,
              var_name='ch', value_name='intensity')
        .pipe(
            (sns.displot, 'data'),
                x='intensity', row='ch', height=1.5, aspect=3)
    )
    g.set_axis_labels("Rescaled Intensity", "Count")

    return g


def plot_level_means(counts, level_means, level_stds, level='miRFP670'):
    fig, ax = _make_figure(1, 1)
    
    ch_to_plot = [ch for ch in level_means.data_vars if ch != level]
    for ch in ch_to_plot:
        ax.errorbar(x=level_means[level], y=level_means[ch], 
                    xerr=level_stds[level], yerr=level_stds[ch], 
                    fmt='o', label=ch,
        )
        ax.set_xlabel(f"{level} Intensity")
        ax.set_ylabel(f"Mean Intensity")
    
    # label number of pixels in each bin (level) if less than 100 pixels
    ch = ch_to_plot[0]
    for i, txt in enumerate(counts[:]):
        if txt > 0 and txt < 100:
            ax.annotate(int(txt), 
                        (level_means[level][i], level_means[ch][i]))
    
    plt.legend()
    fig.set_tight_layout(True)
    return fig


def _make_figure(ncols, nrows, figsize=None, dpi=150):
    if figsize is None:
        figsize = (ncols*3, nrows*2)
    return plt.subplots(figsize=figsize, dpi=dpi)


if __name__ == "__main__":
    if 'snakemake' in globals():
        process_one_nucleus(
            snakemake.input.image,
            snakemake.params.chnames,
            snakemake.params.background,
            snakemake.input.mask,
            snakemake.output,
            snakemake.params.outdir,
            snakemake.wildcards.RoiUID,
        )