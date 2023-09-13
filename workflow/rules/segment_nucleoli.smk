"""
Segmenting nucleoli using snapshot taken before the movie.

Notes
-----
This is a non-ideal setup where I should have these in a self-contained
workflow that can be used as a module. For now, outputs will live in a
subfolder `results/sn` to minimize accidental deletion of output from
rules outside of this smk file.
"""

# This rule is hacky in that, it assumes the *snapshot* file is always
# the same as the *movie* file, except that its 001 (and not 002) in
# the filename.
#
# What we need           : WellF05_aAM_fov001_snap-nodelay-snap001_Deno.nd2
# What we have in roilist: WellF05_aAM_fov001_snap-nodelay-snap002_Deno.nd2
checkpoint sn_crop_roi:
    input:
        'results/roilist.csv'
    output:
        touch('results/sn/image/multi_ch/.created')
    params:
        outdir = 'results/sn/image/multi_ch'
    script:
        "../scripts/sn_crop_roi.py"


def sn_get_checkpoint_RoiUID(wildcards):
    checkpoints.sn_crop_roi.get(**wildcards)
    return glob_wildcards("results/sn/image/multi_ch/{RoiUID, [\w-]+}.ome.tif").RoiUID


rule sn_mask_tiff:
    input:
        image="results/sn/image/multi_ch/{RoiUID}.ome.tif",
        mask="results/segmentation/nucleus/{RoiUID}.ome.tif",
    output:
        single="results/sn/tmp/image_masked/nucleus/singlechannel/{RoiUID}.tif"
    params:
        c=config['sn']['predict_nucleoli']['channel_index']
    run:
        import numpy as np
        import tifffile
        from abcdcs import imageop
        image = imageop.Image.read(input.image, "DataArray")
        mask = imageop.Mask.read(input.mask, "DataArray", 
                                 squeeze=True, drop_single_C=True)
        masked = (image
            .pipe(imageop.Image.mask_to_keep, mask)
            .squeeze('T')
            .expand_dims(T=[0,1])
            .transpose('T','Z','C','Y','X')
        )
        
        tifffile.imwrite(
            output.single, masked.isel(C=params.c).astype(np.uint16).data, 
            imagej=True, metadata={'axes': 'TZYX'}
        )            


rule sn_predict_nucleoli:
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
        images="results/sn/tmp/image_masked/nucleus/singlechannel/{RoiUID}.tif",
        model="resources/ilastik/{model_name}.ilp".format(
              model_name=config['sn']['predict_nucleoli']['model_name'])
    output:
        "results/sn/tmp/segmentation/nucleoli/{RoiUID}.tiff"
    params:
        outdir="results/sn/tmp/segmentation/nucleoli",
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


rule sn_convert_to_ometif:
    """
    Standardize output format, because ilastik provides limited options.
    """
    input:
        "results/sn/tmp/segmentation/nucleoli/{RoiUID}.tiff"
    output:
        "results/sn/segmentation/nucleoli/{RoiUID}.ome.tif"
    run:
        from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
        from abcdcs import imageop
        old = imageop.Mask.read(input[0], "DataArray")
        new = old.transpose('T','C','Z','Y','X').isel(T=0)
        OmeTiffWriter.save(new.astype('uint8').data, output[0])


rule sn_segment_nucleoplasm:
    input:
        nuc="results/segmentation/nucleus/{RoiUID}.ome.tif",
        no="results/sn/segmentation/nucleoli/{RoiUID}.ome.tif", 
    output: 
        npl="results/sn/segmentation/nucleoplasm/{RoiUID}.ome.tif"
    run:
        import numpy as np
        from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
        from abcdcs import imageop
        nuc = imageop.Mask.read(input.nuc, "DataArray", ["nuc"], 
                                squeeze=True, drop_single_C=True)
        nuc = nuc.isel(T=0)
        no = imageop.Mask.read(input.no, "DataArray", ["nucleoli"], 
                               squeeze=True, drop_single_C=True)
        nuc_eroded = imageop.Mask.erode_by_disk(nuc, 5)
        no_dilated = imageop.Mask.dilate_by_disk(no, 5)
        npl = imageop.Image.mask_to_keep(nuc_eroded, no_dilated==0)
        npl = npl.rename("nucleoplasm").astype(np.uint8)
        OmeTiffWriter.save(npl.data, output.npl, dim_order="YX")


# all_segment_nucleoli_input = [
#     lambda w: expand("results/sn/segmentation/{structure}/{RoiUID}.ome.tif",
#                      structure=['nucleoli', 'nucleoplasm'],
#                      RoiUID=sn_get_checkpoint_RoiUID(w)),
# ]

all_segment_nucleoli_input = [
    lambda w: expand("results/sn/segmentation/{structure}/{RoiUID}.ome.tif",
                     structure=['nucleoli', 'nucleoplasm'],
                     RoiUID=['y505_2023-05-01_1_live_E03_1_1']),
]