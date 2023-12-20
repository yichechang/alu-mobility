"""
Segmenting nucleus

Outputs
-------
nucleus segmentation from cellpose (original)
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


all_segmentation_nucleus_input = [
    lambda w: expand("results/segmentation/nucleus/{RoiUID}.ome.tif",
                     RoiUID=get_checkpoint_RoiUID(w)),
]