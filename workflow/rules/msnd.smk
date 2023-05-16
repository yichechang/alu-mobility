rule msnd:
    input:
        piv="results/piv/{protocol}/{ch}/{RoiUID}.mat",
        image="results/image_normalized/by_nucleus/multi_ch/{RoiUID}.ome.tif",
        mask="results/segmentation/nucleus/{RoiUID}.ome.tif",
    output:
        expand(
            "results/msnd/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_{outtype}.csv",
            msnd_protocol=ALL_MSND_PROTOCOLS,
            outtype=['stats', 'indiv'],
            allow_missing=True,
        )
    params:
        chnames=ALL_CH,
        outdir="results/msnd/",
    script:
        "../scripts/msnd.py"


all_msnd_input = [
    lambda w: expand(
        "results/msnd/{protocol}/{ch}/{msnd_protocol}/{RoiUID}_{outtype}.csv",
        protocol=ALL_PROTOCOLS,
        ch=config['msnd']['channel'],
        msnd_protocol=ALL_MSND_PROTOCOLS,
        outtype=['stats', 'indiv'],
        RoiUID=get_checkpoint_RoiUID(w),
    )
]