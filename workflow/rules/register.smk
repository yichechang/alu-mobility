"""
"""

rule register_nucleus:
    input:
        multi="results/image/multi_ch/{RoiUID}.ome.tif",
    output:
        multi="results/image_registered/multi_ch/{RoiUID}.ome.tif",
    params:
        fluoros=[ch['fluoro'] for ch in config['input']['channels']],
        bitdepth=config['input']['bitdepth'],
    run:
        """
        This rule registers the first two channels of a movie to each 
        other. Any additional channels remain untouched in the 
        'registered' output, yielding a multi-channel image per the 
        config spec.
        
        Note: Only the first two channels are registered!
        """
        import xarray as xr
        from pystackreg import StackReg
        from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
        from abcdcs import imageop

        images = (
            imageop.Image.read(
                input.multi,
                fmt='DataArray',
                channel_names=params.fluoros,
                bitdepth=params.bitdepth,
            )
            .squeeze()
        )
        
        def register(by, to, mode="previous"):
            sr = StackReg(StackReg.RIGID_BODY)
            tmats = sr.register_stack(by, reference=mode)
            out = sr.transform_stack(to)
            return tmats, out

        _, registered_0 = register(images.isel(C=1).data, images.isel(C=0).data)
        _, registered_1 = register(images.isel(C=0).data, images.isel(C=1).data)

        # lazy way of preserving coordinates and only swapping pixel data
        # TODO: make it not depend on these 
        registered = images.copy()
        registered.isel(C=0).data = registered_0
        registered.isel(C=1).data = registered_1
        OmeTiffWriter.save(registered.data, output.multi, dim_order='TCYX')


all_register_input = [
    lambda w: expand(
        "results/image_registered/multi_ch/{RoiUID}.ome.tif",
        RoiUID=get_checkpoint_RoiUID(w)
    ),
]




