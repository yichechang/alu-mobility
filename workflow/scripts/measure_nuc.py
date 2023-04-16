import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from abcdcs import imageop

def main(image_path, mask_path, cinfo, bitdepth, outpath):
    # read image from time zero and set C to last axis for regionprops
    image = (
        imageop.Image.read(image_path, "DataArray", 
            channel_names=cinfo['fluoro'],
            background=cinfo['background'],
            rescale=True, bitdepth=bitdepth,
            squeeze=True,
        )
        .isel(T=0)
        .transpose('Y', 'X', 'C')
    )
    mask = (
        imageop.Mask.read(mask_path, "DataArray", 
            channel_names=[snakemake.wildcards.structure],
            squeeze=True, drop_single_C=True,
        )
        .isel(T=0)
        .transpose('Y', 'X')
    )


    # calculate properties
    properties = [
        "label", 
        "area",
        "intensity_mean",
    ]
    extra_properties = [intensity_std]

    feat = regionprops_table(mask.data, image.data, 
                            properties=properties,
                            extra_properties=extra_properties)


    # make a dataframe with RoiUID + the nucleus' features
    df = pd.DataFrame(feat).drop(columns=["label"])
    df["RoiUID"] = snakemake.wildcards.RoiUID
    df["object"] = mask.name
    # Move RoiUID and object to the first column ... annoyingly
    df = df[ ['object'] + [ col for col in df.columns if col != 'object' ] ]
    df = df[ ['RoiUID'] + [ col for col in df.columns if col != 'RoiUID' ] ]

    df = update_colnames_with_channels(df, cinfo)

    # save resulting dataframe
    df.to_csv(outpath, index=False)

def intensity_std(mask, image):
    """Function to calculate standard deviation"""
    return np.std(image[mask>0])

def update_colnames_with_channels(df, cinfo):
    """ 'intensity_x-n' to 'intensity_x-chname'
    
    Assume n is a single digit
    """
    cols_to_replace = df.columns.str.contains(r'intensity_.*-\d')
    colnames_to_replace = df.columns[cols_to_replace].to_list()
    dfc = pd.DataFrame({'old': colnames_to_replace})
    dfc['ch_number'] = dfc['old'].str.extract(r'.*-(\d)').astype(int)
    dfc = pd.merge(left=dfc, right=pd.DataFrame(cinfo),
                   left_on='ch_number', right_index=True)
    dfc['new'] = dfc['old'].str[0:-1] + dfc['fluoro']
    dfc_dict = dfc[['old', 'new']].to_dict('list')
    # dfc_dict = {'old': [...], 'new': [...]}
    mapper = {o: n for o, n in zip(*list(dfc_dict.values()))}
    return df.rename(columns=mapper)


if __name__ == '__main__':
    if 'snakemake' in globals():
        main(
            snakemake.input.image,
            snakemake.input.mask,
            snakemake.params.cinfo,
            snakemake.params.bitdepth,
            snakemake.output.csv,
        )