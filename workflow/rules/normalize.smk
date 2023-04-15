rule normalize:
    input:
        image="results/image/single_ch/{ch}/{RoiUID}.ome.tif",
        mask="results/segmentation/{structure}/{RoiUID}.ome.tif",
    output:
        image="results/image_normalized/by_{structure}/single_ch/{ch}/{RoiUID}.ome.tif"
    params:
        method=config['normalize']['method'],
        method_kwargs=config['normalize']['method_kwargs'],
    run:
        import pandas as pd
        from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
        from abcdcs import imageop
        cinfo = pd.DataFrame.from_records(config['input']['channels'])
        background = cinfo['background'][cinfo['fluoro']==wildcards.ch].item()
        image = imageop.Image.read(
            input.image,
            fmt='DataArray',
            channel_names=[wildcards.ch],
            bitdepth=config['input']['bitdepth'],
            rescale=True,
            background=background,
        )
        mask = imageop.Mask.read(
            input.mask,
            fmt='DataArray',
            channel_names=[wildcards.structure],
            squeeze=True,
            drop_single_C=True,
        )
        masked = imageop.Image.mask_to_keep(image, mask)
        if params.method_kwargs is not None:
            method_kwargs = params.method_kwargs
        else:
            method_kwargs = {}
        normalized = imageop.Image.normalize_by(
            image, masked, params.method, **method_kwargs)
        OmeTiffWriter.save(normalized.data, output.image)

all_normalize_input = (
    lambda w: expand(
        "results/image_normalized/by_{structure}/single_ch/{ch}/{RoiUID}.ome.tif",
        structure=['nucleus'], ch=ALL_CH,
        RoiUID=get_checkpoint_RoiUID(w),
    )
)
        