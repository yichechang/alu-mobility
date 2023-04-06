rule parse_metadata:
    input:
        get_imageset_files
    output: 
        'results/imagesetlist.csv',
    run:
        from abcdcs import curate
        import csv
        parsed = curate.parse_filepaths(
            fpaths=input, 
            pat=config['input']['image_path']['regex'], 
            datefmt=config['input']['datefmt'],
            nafilter=config['parse_metadata']['nafilter'],
            verbose=config['parse_metadata']['verbose'],
        )
        samplesheet = curate.read_samplesheet(config['input']['samplesheet_path'])
        merged = curate.merge_sample_metadata(
            parsed, samplesheet, patch=config['parse_metadata']['patch'])
        merged.to_csv(output[0], quoting=csv.QUOTE_NONNUMERIC, index=False)


rule draw_roi:
    input: 
        'results/imagesetlist.csv',
    output:
        'results/roilist.csv'
    script:
        "../scripts/draw_roi.py"


# Note:
#   Defining output by touching a file inside the desired output folder
#   (specified same as params.outdir) ensures that folder gets created.
#   This means that the script doesn't need to check/create by itself.
#   We also don't need to explicitly do mkdir before calling the script.
#   See [why this is intended](https://github.com/snakemake/snakemake/issues/774#issuecomment-1036152852)
checkpoint crop_roi:
    input:
        'results/roilist.csv'
    output:
        touch("results/image/multi_ch/.created")
    params:
        outdir = "results/image/multi_ch"
    resources:
        mem = 1000, 
        time = 30, 
        short_jobs = 1, # <61 minutes
    script:
        "../scripts/crop_roi.py"


rule split_channels:
    input:
        "results/image/multi_ch/{RoiUID}.ome.tif"
    output:
        expand("results/image/single_ch/{ch}/{RoiUID}.ome.tif", 
               ch=ALL_CH, allow_missing=True)
    resources:
        mem = 1000, 
        time = 30, 
        short_jobs = 1, # <61 minutes
    script:
        "../scripts/split_channels.py"


all_draw_roi_input = 'results/roilist.csv'

all_roi_input = [
    lambda w: expand("results/image/multi_ch/{RoiUID}.ome.tif", 
                     RoiUID=get_checkpoint_RoiUID(w)),
    lambda w: expand("results/image/single_ch/{ch}/{RoiUID}.ome.tif", 
                     ch=ALL_CH, RoiUID=get_checkpoint_RoiUID(w)),
]