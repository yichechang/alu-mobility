from typing import Dict, List

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

import typer

# TODO:
#   Remove dependence on order or channels defined in pepfile
def main(input_path: str, output_paths: List[str], chinfo_dict=None) -> None:
    imageset = AICSImage(input_path).xarray_data

    for idx in range(len(chinfo_dict['fluoro'])):
        image = imageset.isel(C=idx).squeeze()
        OmeTiffWriter.save(image.data, output_paths[idx], dim_order="TYX")

# TODO:
#  - Don't actually need this. Just need to know the number of channels
def extract_channel_info(channel_info_list) -> Dict:
    """
    From a pepfile, convert channel info from list of {k: v}'s to {key: list of v's}
    """
    channels = channel_info_list
    keys = channels[0].keys()
    return {
        key: [ch[key] for ch in channels]
        for key in keys
    }


if __name__ == '__main__':
    if 'snakemake' in globals():
        chinfo_dict = extract_channel_info(
            snakemake.config['input']['channels'])
        main(snakemake.input[0], 
             snakemake.output, 
             chinfo_dict=chinfo_dict)
    else:
        typer.run(main)