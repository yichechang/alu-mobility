"""
Segment heterochromatin and euchromatin

Notes
-----
Segmentation is based on thresholding H2B channel for each 
timepoint and a union across time is produced as the final
mask.

Both HC and EC are estimated more conservatively. So EU + HC
would not be the entire nucleus.


TODO
----
Use ML-based method and labels from HP1a channel.
"""

rule segment_hc:
    input:
        image=expand(
            "results/image_normalized/by_nucleus/multi_ch/{RoiUID}.ome.tif",
            ch=config['segment_hc']['channel'],
            allow_missing=True
        )
    output:
        "results/segmentation/hc/{RoiUID}.ome.tif"
    params:
        channel_names=ALL_CH,
        channel=config["segment_hc"]["channel"],
        hc=config["segment_hc"]["hc"],
    script:
        "../scripts/segment_hc.py"

all_segment_hc_input = [
    lambda wildcards: expand(
        "results/segmentation/hc/{RoiUID}.ome.tif",
        RoiUID=get_checkpoint_RoiUID(wildcards)
    )
]