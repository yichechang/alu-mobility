"""
Segmenting nucleoli and nucleoplasm.


Inputs
------
- nucleus segmentation from cellpose (original)
- ilastik model for predicting nucleoli using BFP grayscale image


Outputs
-------
- nucleoli segmentation (shrunken, more conservative)
- nucleoplasm segmentation (shrunken, more conservative)
       

Notes
-----
Note that I cannot get ilastik to open `.ome.tif` files created by
aicsimageio's OmeTiffWriter, and have to use tifffile to save as 
ImageJ-compatible `.tif` files. This is rather annoying but doesn't 
add any complexity yet (maybe something to solve if relying on OME
metadata).
"""


rule all_segment_nucleoli:
    input:
        lambda w: expand("results/segmentation/{structure}/{roiuid}.ome.tif",
                         structure=['nucleoli', 'nucleoplasm'],
                         roiuid=get_checkpoint_RoiUID(w)),


rule mask_nuclear_image_for_ilastik:
    input:
        image="results/image/multi_ch/{roiuid}.ome.tif",
        mask="results/segmentation/nucleus/{roiuid}.ome.tif",
    output:
        multi="results/tmp/image_masked/nucleus/multichannel/{roiuid}.tif",
        split=expand("results/tmp/image_masked/nucleus/split/{ch}/{roiuid}.tif",
                     ch=ALL_CH, allow_missing=True)
    run:
        import numpy as np
        import tifffile
        from abcdcs import imageop
        image = imageop.Image.read(input.image, "DataArray")
        mask = imageop.Mask.read(input.mask, "DataArray", 
                                 squeeze=True, drop_single_C=True)
        masked = (image
            .pipe(imageop.Image.mask_to_keep, mask)
            .transpose('T','Z','C','Y','X')
        )
        tifffile.imwrite(output.multi, masked.astype(np.uint16).data, 
                         imagej=True)
        for c in range(masked.sizes['C']):
            tifffile.imwrite(
                output.split[c], masked.isel(C=c).astype(np.uint16).data, 
                imagej=True, metadata={'axes': 'TZYX'}
            )


rule predict_nucleoli:
    """
    ilastik model for segmenting out nucleoli regions.

    Model was trained on 2D grayscale BFP images with its original scale
    and 0's filled outside of the nucleus.

    TODO
    - [ ] Experiment if normalized images would work to handle wider 
          range of BFP expressions, or images acquired with non-16-bit.
    - [ ] Check if float (so NaN's allowed) would make it easier than
          0's outside of the nucleus (now 3 classes required)
    """
    input:
        images=expand("results/tmp/image_masked/nucleus/split/{ch}/{roiuid}.tif",
                      ch=config['predict_nucleoli']['channel'],
                      allow_missing=True),
        model="resources/ilastik/{model_name}.ilp".format(
              model_name=config['predict_nucleoli']['model_name'])
    output:
        expand("results/tmp/segmentation/nucleoli/{roiuid}.tiff",
               allow_missing=True)
    params:
        # ilp=ILASTIK_PROJ,
        outdir="results/tmp/segmentation/nucleoli",
    conda: "ilastik"
    shell:
        """
        ilastik \
--headless \
--project={input.model} \
--output_format="multipage tiff" \
--export_source="Object Predictions" \
--export_dtype=uint8 \
--output_filename_format={params.outdir}/{{nickname}}.tiff \
{input.images}
        """


rule convert_tiff_to_ometif:
    """
    Standardize output format, because ilastik provides limited options.
    """
    input:
        "results/tmp/segmentation/nucleoli/{roiuid}.tiff"
    output:
        "results/segmentation/nucleoli/{roiuid}.ome.tif"
    run:
        from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
        from abcdcs import imageop
        old = imageop.Mask.read(input[0], "DataArray")
        new = old.transpose('T','C','Z','Y','X')
        OmeTiffWriter.save(new.astype('uint8').data, output[0])


rule segment_nucleoplasm:
    input:
        nuc="results/segmentation/nucleus/{roiuid}.ome.tif",
        no="results/segmentation/nucleoli/{roiuid}.ome.tif", 
    output: 
        npl="results/segmentation/nucleoplasm/{roiuid}.ome.tif"
    run:
        import numpy as np
        from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
        from abcdcs import imageop
        nuc = imageop.Mask.read(input.nuc, "DataArray", ["nuc"], 
                                squeeze=True, drop_single_C=True)
        no = imageop.Mask.read(input.no, "DataArray", ["nucleoli"], 
                               squeeze=True, drop_single_C=True)
        nuc_eroded = imageop.Mask.erode_by_disk(nuc, 5)
        no_dilated = imageop.Mask.dilate_by_disk(no, 5)
        npl = imageop.Image.mask_to_keep(nuc_eroded, no_dilated==0)
        npl = npl.rename("nucleoplasm").astype(np.uint8)
        OmeTiffWriter.save(npl.data, output.npl, dim_order="TYX")