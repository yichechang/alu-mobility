from typing import Dict, List, Optional
from pathlib import Path
import shutil

import napari
from magicgui import magicgui
from magicgui.widgets import Container
from aicsimageio import AICSImage
import pandas as pd

from abcdcs import curate
import typer


# TODO:
#   Consider picking some defaults for chinfo_dict
def roi_drawer(imageset_records: List, output_dir: Path, chinfo_dict: Dict = None):
    """View and draw ROIs on each image set from the filepath list.

    Parameters
    ----------
    imageset_records: list of dict
        List of imageset records. Each record needs to contain minimally
        `ImagesetFilepath` (absolute path point to an nd2 file) and 
        `ImagesetUID` (UID associated with that nd2 file).
    output_dir: pathlib.Path
        Path to the directry where compiled results (csv) to be saved.
    chinfo_dict: Dict
        Dictionary containing channel information, in the format of
        {key: [val_Ch0, val_Ch1, val_Ch2, ...]}. Keys required include
        `fluoro` and `colormap`.

    Notes
    -----
    Need to click 'next' or 'prev' to trigger saving. This is fine for
    all but the last imageset, as it is not something natural to do at
    the end. 
    """
    # ====================
    # Start napari viewer using the first image set
    # ====================
    current_path = 0
    img = AICSImage(imageset_records[current_path]['ImagesetFilepath']).xarray_data

    if chinfo_dict is None:
        n_channels = img.coords['C'].size
        chinfo_dict = {
            'fluoro': ['channel:'+str(idx) for idx in range(n_channels)],
            'colormap': ['Greys_r' for _ in range(n_channels)]
        }

    viewer = napari.view_image(
        img, 
        channel_axis=1, 
        colormap=chinfo_dict['colormap'], 
        name=chinfo_dict['fluoro'],
    )
    # Connect these two image layers to autoscale contrast when updated
    # From: https://forum.image.sc/t/function-to-autoadjust-lut-programatically/40396/2
    for l in viewer.layers:
        l.events.data.connect(lambda e: l.reset_contrast_limits())

    # Create ShapesLayer for drawing ROIs
    shapes_layer = viewer.add_shapes(
        shape_type='rectangle',
        edge_color='coral',
        face_color='royalblue',
        name='ROIs',
    )
    shapes_layer.mode = 'add_rectangle'

    #
    # Check and prepare output file directries
    #
    output_dir.mkdir(exist_ok=True)
    


    # ====================
    # GUI functions for processing
    # ====================
    @magicgui(call_button='next')
    def next_callback(viewer: napari.Viewer):
        # nonlocal current_path
        save_current_rois(viewer)
        if current_path + 1 < len(imageset_records):
            set_up_new_file(viewer, 'next')


    @magicgui(call_button='prev')
    def prev_callback(viewer: napari.Viewer):
        # nonlocal current_path
        save_current_rois(viewer)
        if current_path - 1 >= 0:
            set_up_new_file(viewer, 'prev')


    def save_current_rois(viewer):
        outname = curate.MetadataConverter('imageset').gather(imageset_records[current_path]) + '.csv'
        outpath = output_dir / outname

        # Not sure how to access the dataframe describing ROI features.
        # But can save them asa csv and they will have the following
        # fields: index, shape-type, vertex-index, axis-0, axis-1.
        # => Since these are intermediate files, just set 'index' to be
        #    RoiID when combining the csv files. (See `compile_callback`)
        viewer.layers['ROIs'].save(str(outpath))


    def set_up_new_file(viewer, direction):
        # Keep track of state of which image file to use
        nonlocal current_path
        if direction == 'next':
            current_path += 1
        elif direction == 'prev':
            current_path -= 1
        
        # Update layers for new images without adding/removing layers
        img = AICSImage(imageset_records[current_path]["ImagesetFilepath"]).xarray_data
        # viewer.layers['miRFP670'].data = img.isel(C=0).data
        # viewer.layers['GFP'].data = img.isel(C=1).data
        # TODO:
        #   Do not rely on the order (i.e., don't use index)
        for idx,fluoro in enumerate(chinfo_dict['fluoro']):
            viewer.layers[fluoro].data = img.isel(C=idx).data

        # Reset ROI layer for the new image
        viewer.layers['ROIs'].data = []
        viewer.layers.selection.active = viewer.layers['ROIs']
        viewer.layers['ROIs'].mode = 'add_rectangle'

    # Add widgets to napari viewer
    container = Container(
        widgets=[next_callback, prev_callback]
    )
    viewer.window.add_dock_widget(container)

    napari.run()


def compile_roilist(
        tmp_dir: str, 
        output_path: str, 
        imagelist_path: str
    ) -> None:
    """
    Compile a single roilist from all tmp roilist csv files.
    """
    #
    # Gather all csv files containing ROI coordinates from each imageset
    #
    files = list(Path(tmp_dir).glob('*.csv'))
    dfs_coords = [pd.read_csv(file) for file in files]
    # Add ImagesetUID in ROI information
    for df,file in zip(dfs_coords,files):
        df['ImagesetUID'] = file.stem
    df_coords = pd.concat(dfs_coords)
    # Add ID and UID for each ROI coordinates. The Numeric ID already 
    # exists as column named 'index' (note: not as the index of the 
    # dataframe).
    df_coords = df_coords.rename(columns={'index': 'RoiID'})
    df_coords['RoiID'] = df_coords['RoiID'].astype(str)
    df_coords['RoiUID'] = df_coords['ImagesetUID'] + '_' + df_coords['RoiID']
    

    #
    # Process ROI coordinates to ROI
    #
    colnames_roiUID = ['RoiUID', 'ImagesetUID', 'RoiID']
    df_rois = pd.pivot(df_coords,
                       index=colnames_roiUID,
                       columns=['vertex-index'],
                       values=['axis-0', 'axis-1'],
                      )
    df_rois['r1'] = df_rois[[('axis-0', 0)]]
    df_rois['r2'] = df_rois[[('axis-0', 2)]]
    df_rois['c1'] = df_rois[[('axis-1', 0)]]
    df_rois['c2'] = df_rois[[('axis-1', 2)]]
    df_rois['ri'] = df_rois[['r1', 'r2']].min(axis=1)
    df_rois['rf'] = df_rois[['r1', 'r2']].max(axis=1)
    df_rois['ci'] = df_rois[['c1', 'c2']].min(axis=1)
    df_rois['cf'] = df_rois[['c1', 'c2']].max(axis=1)

    df_rois = df_rois.reset_index()
    df_rois = df_rois[colnames_roiUID + ['ri', 'rf', 'ci', 'cf']].droplevel('vertex-index', axis=1)

    #
    # Merge Imageset metadata to Roi
    #
    imagelist = pd.read_csv(imagelist_path, dtype=str)
    df_rois = df_rois.merge(imagelist, on=['ImagesetUID'])

    #
    # save output to file
    #
    df_rois.to_csv(output_path, quoting=2, index=False, header=True)

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

def main(imagesetlist_path: str, output_path: str, chinfo_dict=None
    ) -> None:
    # Don't type hint `chinfo_dict` (which should be Dict) because 
    # typer doesn't support such type. This also means currently we
    # can't specify channel information when launching via commandline,
    # and as such all channels will be in grey.
    # TODO:
    #   If refactoring roi_drawer(), pick some sensible defaults.
    """
    Combined ROI information will be saved as file <output_path>. A tmp
    directory will be created in the same dir which <output_path> is in,
    to store individual csv files for each Imageset. 
    """    
    imageset_records = pd.read_csv(imagesetlist_path, dtype=str).to_dict('records')
    
    output_pathobj = Path(output_path)
    output_pathobj.parent.mkdir(parents=True, exist_ok=True)

    # Create a new tmp dir for this specific drawing action. Note that
    # old 'tmp/roi_drawer/' will be removed first.
    tmp_dir = output_pathobj.parent / "tmp" / "roi_drawer"
    if tmp_dir.exists() and tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    roi_drawer(imageset_records, tmp_dir, chinfo_dict)
    
    compile_roilist(tmp_dir, output_path, imagesetlist_path)


if __name__ == '__main__':
    if 'snakemake' in globals():
        chinfo_dict = extract_channel_info(
            snakemake.config['input']['pepfile_path'], 
            snakemake.config['input']['experiment_type'])
        main(snakemake.input[0], 
             snakemake.output[0], 
             chinfo_dict=chinfo_dict)
    else:
        typer.run(main)

