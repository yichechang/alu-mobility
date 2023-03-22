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


def extract_channel_info(pepfile_path: str, exptype: str) -> Dict:
    """
    From a pepfile, convert channel info from list of {k: v}'s to {key: list of v's}
    """
    import yaml
    with open(pepfile_path, 'rb') as f:
        conf = yaml.load(f, Loader=yaml.CLoader)
    channels = conf['experiments'][exptype]['channels']
    keys = channels[0].keys()
    return {
        key: [ch[key] for ch in channels]
        for key in keys
    }


if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        snakemake = None
    if snakemake is not None:
        chinfo_dict = extract_channel_info(
            snakemake.config['input']['pepfile_path'], 
            snakemake.config['input']['experiment_type'])
        main(snakemake.input[0], 
             snakemake.output, 
             chinfo_dict=chinfo_dict)
    else:
        typer.run(main)