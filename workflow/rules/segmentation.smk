"""
Segmenting nucleus, nucleoli and nucleoplasm.

Outputs
-------
- nucleus segmentation from cellpose (original)
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

rule segment_nuclei_in_time:
    input:
        expand("results/image_registered/single_ch/{ch}/{RoiUID}.ome.tif",
               ch=config['segment_nuclei_in_time']['channel'],
               allow_missing=True)
    output:
        "results/segmentation/nucleus/{RoiUID}.ome.tif"
    resources:
        mem = 2000, 
        time = 61, 
    script:
        "../scripts/segment_nuclei_in_time.py"


# rule mask_nuclear_image_for_ilastik:
#     input:
#         image="results/image_registered/multi_ch/{RoiUID}.ome.tif",
#         mask="results/segmentation/nucleus/{RoiUID}.ome.tif",
#     output:
#         multi=temp("results/tmp/image_masked/nucleus/multichannel/{RoiUID}.tif"),
#         split=temp(
#             expand("results/tmp/image_masked/nucleus/split/{ch}/{RoiUID}.tif",
#                    ch=ALL_CH, allow_missing=True)
#         )
#     run:
#         import numpy as np
#         import tifffile
#         from abcdcs import imageop
#         image = imageop.Image.read(input.image, "DataArray")
#         mask = imageop.Mask.read(input.mask, "DataArray", 
#                                  squeeze=True, drop_single_C=True)
#         masked = (image
#             .pipe(imageop.Image.mask_to_keep, mask)
#             .transpose('T','Z','C','Y','X')
#         )
#         tifffile.imwrite(output.multi, masked.astype(np.uint16).data, 
#                          imagej=True)
#         for c in range(masked.sizes['C']):
#             tifffile.imwrite(
#                 output.split[c], masked.isel(C=c).astype(np.uint16).data, 
#                 imagej=True, metadata={'axes': 'TZYX'}
#             )


# rule predict_nucleoli:
#     """
#     ilastik model for segmenting out nucleoli regions.

#     Model was trained on 2D grayscale BFP images with its original scale
#     and 0's filled outside of the nucleus.

#     TODO
#     - [ ] Experiment if normalized images would work to handle wider 
#           range of BFP expressions, or images acquired with non-16-bit.
#     - [ ] Check if float (so NaN's allowed) would make it easier than
#           0's outside of the nucleus (now 3 classes required)
#     """
#     input:
#         images=expand("results/tmp/image_masked/nucleus/split/{ch}/{RoiUID}.tif",
#                       ch=config['predict_nucleoli']['channel'],
#                       allow_missing=True),
#         model="resources/ilastik/{model_name}.ilp".format(
#               model_name=config['predict_nucleoli']['model_name'])
#     output:
#         temp(
#             expand("results/tmp/segmentation/nucleoli/{RoiUID}.tiff",
#                    allow_missing=True)
#         )
#     params:
#         # ilp=ILASTIK_PROJ,
#         outdir="results/tmp/segmentation/nucleoli",
#     conda: "ilastik"
#     shell:
#         """
#         ilastik \
# --headless \
# --project={input.model} \
# --output_format="multipage tiff" \
# --export_source="Object Predictions" \
# --export_dtype=uint8 \
# --output_filename_format={params.outdir}/{{nickname}}.tiff \
# {input.images}
#         """


# rule convert_tiff_to_ometif:
#     """
#     Standardize output format, because ilastik provides limited options.
#     """
#     input:
#         "results/tmp/segmentation/nucleoli/{RoiUID}.tiff"
#     output:
#         "results/segmentation/nucleoli/{RoiUID}.ome.tif"
#     run:
#         from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
#         from abcdcs import imageop
#         old = imageop.Mask.read(input[0], "DataArray")
#         new = old.transpose('T','C','Z','Y','X')
#         OmeTiffWriter.save(new.astype('uint8').data, output[0])


# rule segment_nucleoplasm:
#     input:
#         nuc="results/segmentation/nucleus/{RoiUID}.ome.tif",
#         no="results/segmentation/nucleoli/{RoiUID}.ome.tif", 
#     output: 
#         npl="results/segmentation/nucleoplasm/{RoiUID}.ome.tif"
#     run:
#         import numpy as np
#         from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
#         from abcdcs import imageop
#         nuc = imageop.Mask.read(input.nuc, "DataArray", ["nuc"], 
#                                 squeeze=True, drop_single_C=True)
#         no = imageop.Mask.read(input.no, "DataArray", ["nucleoli"], 
#                                squeeze=True, drop_single_C=True)
#         nuc_eroded = imageop.Mask.erode_by_disk(nuc, 5)
#         no_dilated = imageop.Mask.dilate_by_disk(no, 5)
#         npl = imageop.Image.mask_to_keep(nuc_eroded, no_dilated==0)
#         npl = npl.rename("nucleoplasm").astype(np.uint8)
#         OmeTiffWriter.save(npl.data, output.npl, dim_order="TYX")


all_segmentation_nucleus_input = (
    lambda w: expand("results/segmentation/nucleus/{RoiUID}.ome.tif",
                     RoiUID=get_checkpoint_RoiUID(w))
)


all_segmentation_input = [
    lambda w: expand("results/segmentation/{structure}/{RoiUID}.ome.tif",
                    #  structure=['nucleus', 'nucleoli', 'nucleoplasm'],
                     structure=['nucleus'],
                     RoiUID=get_checkpoint_RoiUID(w)),
]