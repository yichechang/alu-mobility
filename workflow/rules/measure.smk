rule measure:
    input:
        image="results/image/multi_ch/{RoiUID}.ome.tif",
        mask="results/segmentation/{structure}/{RoiUID}.ome.tif"
    output:
        csv=temp("results/measurements/individual/{RoiUID}_{structure}.csv")
    params:
        cinfo=CINFO,
        bitdepth=config['input']['bitdepth'],
    script:
        "../scripts/measure_nuc.py"

rule combine_measurements:
    input:
        lambda w: expand(
            "results/measurements/individual/{RoiUID}_{structure}.csv",
            RoiUID=get_checkpoint_RoiUID(w),
            allow_missing=True
        )
    output:
        csv="results/measurements/{structure}.csv"
    run:
        import pandas as pd
        dfs = [pd.read_csv(roi) for roi in input]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(output.csv, index=False)

all_measure_input = [
    expand("results/measurements/{structure}.csv",
           structure=config['measure']['structure'])
]