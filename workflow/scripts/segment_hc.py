import functools

import numpy as np
import xarray as xr
from skimage import filters, morphology

from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from abcdcs import imageop

def main(image_path, output_path):
    image = load_image(image_path)

    # segment heterochromatin
    # end result does not contain time dimension
    blurred = blur_image(image, sigma=PARAMS["hc"]["blur_sigma"])
    thresholded = threshold_image(blurred, threshold=PARAMS["hc"]["zscore_threshold"])
    opened = morphological_opening(thresholded, disk_size=PARAMS["hc"]["clean_disk_size"])
    cleaned = remove_small_objects(opened, min_size=PARAMS["hc"]["clean_min_size"])
    hc = threshold_across_time(cleaned, threshold=PARAMS["hc"]["time_threshold"])

    # save as uint8 images
    hc = hc.rename("heterochromatin").astype(np.uint8)
    OmeTiffWriter.save(hc.data, output_path, dim_order="YX")

def load_image(file_path):
    return (
        imageop.Image.read(
            file_path, 
            "DataArray", 
            channel_names=PARAMS["channel_names"]
        )
        .sel(C=PARAMS["channel"])
        .squeeze()
    )

def load_mask(file_path):
    return (
        imageop.Mask.read(
            file_path, 
            "DataArray", 
            ["nuc"],
            squeeze=True,
            drop_single_C=True
        )
    )

def apply_ufunc_simple_by_plane(func):
    """
    Wrap functions while preserving signature and docstrings.
    """
    @functools.wraps(func)
    def wrapper(data, **kwargs):
        return _apply_ufunc_simple_by_plane(func, data, **kwargs)
    return wrapper

@apply_ufunc_simple_by_plane
def blur_image(image, sigma: float = 1.0):
    return filters.gaussian(image, sigma=sigma)


@apply_ufunc_simple_by_plane
def threshold_image(image, threshold: float = 0.5):
    return image > threshold


@apply_ufunc_simple_by_plane
def morphological_opening(image, disk_size: int = 1):
    selem = morphology.disk(disk_size)
    return morphology.opening(image, selem)


@apply_ufunc_simple_by_plane
def remove_small_objects(image, min_size: int = 20):
    return morphology.remove_small_objects(image, min_size)


def threshold_across_time(image, threshold: float = 0.5):
    projected = image.mean(dim="T") > threshold
    return projected

def _apply_ufunc_simple_by_plane(
    f,
    data, 
    **kwargs
):
    """Wrapper for applying a function to each XY-plane.

    This is a wrapper for xarray.apply_ufunc with some assumptions and
    defaults that are useful for applying a function to every XY-plane.
    The function f takes a 2D array as its first argument, and does not
    require another DataArray or Dataset as its second argument. It can
    take other arguments to be passed in as kwargs. 

    Returns
    -------
    xr.DataArray
    """
    return xr.apply_ufunc(
        f,
        data,
        input_core_dims=[['Y', 'X']],
        output_core_dims=[['Y', 'X']],
        vectorize=True,
        kwargs=kwargs,
    )


if __name__ == "__main__":
    PARAMS = snakemake.params
    INPUT = snakemake.input
    OUTPUT = snakemake.output
    main(INPUT["image"][0], OUTPUT[0])