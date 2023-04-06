rule normalize_intensity:
    input:
        image="results/image/multi_ch/{RoiUID}.ome.tif",
        mask="results/segmentation/nucleus/{RoiUID}.ome.tif",
    output:
        nuc="results/image_normalized_nucleus/{RoiUID}.ome.tif",
        npl="results/image_normalized_nucleoplasm/{RoiUID}.ome.tif",
    params:
        chnames=ALL_CH,
        outdir="results/image_normalized_nucleoplasm/",
        background=config['normalize_intensity']['background'],
    script:
        "../scripts/normalize_intensity.py"


rule msnd:
    input:
        piv="results/piv/{protocol}/" + config['msnd']['channel'] + "/{RoiUID}.mat",
        image="results/image_normalized_nucleoplasm/{RoiUID}.ome.tif",
        mask="results/segmentation/nucleus/{RoiUID}.ome.tif",
    output:
        normal="results/msnd/{protocol}/normal/{RoiUID}_stats.csv",
        weighted="results/msnd/{protocol}/weighted/{RoiUID}_stats.csv",
    params:
        chnames=ALL_CH,
        background=config['normalize_intensity']['background'],
        outdir="results/msnd/",
    script:
        "../scripts/weighted_msnd.py"

all_normalize_and_msnd_input = [
    lambda w: expand("results/image_normalized_nucleus/{RoiUID}.ome.tif",
                     RoiUID=get_checkpoint_RoiUID(w)),
    lambda w: expand("results/image_normalized_nucleoplasm/{RoiUID}.ome.tif",
                     RoiUID=get_checkpoint_RoiUID(w)),
    lambda w: expand("results/msnd/{protocol}/{mode}/{RoiUID}_stats.csv",
                     protocol=ALL_PROTOCOLS,
                     mode=['normal', 'weighted'],
                     RoiUID=get_checkpoint_RoiUID(w))
]