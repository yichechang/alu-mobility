import pandas as pd

configfile: 'snakemake_config.yaml'
pepfile: config['input']['pepfile']

OUTPUT_DIR = config['output']['dir'] 

rule all:
    input:
        OUTPUT_DIR + "crop_roi.done"

rule build_imagelist:
    output: 
        OUTPUT_DIR + 'imagesetlist.csv',
    shell:
        # Must quote whatever that will be used as paths
        "python scripts/build_imagelist.py "
                "'{config[input][basedir]}' "
                "'{config[input][subdir]}' "
                "{config[input][ext]} "
                "'{pep.config[metadata][movie][pattern]}' "
                "'{config[input][samplesheet]}' "
                "'{output}' "
                "--nafilter strict --patch --verbose"

rule draw_roi:
    input: 
        OUTPUT_DIR + 'imagesetlist.csv'
    output:
        OUTPUT_DIR + 'roilist.csv'
    shell:
        "python scripts/draw_roi.py '{input}' '{output}'"


OUTPUT_DIR_CROP = OUTPUT_DIR + "single_nuc_movie/"
rule crop_roi:
    input:
        OUTPUT_DIR + 'roilist.csv'
    output:
        touch(OUTPUT_DIR + "crop_roi.done"),
        expand(OUTPUT_DIR_CROP + "{RoiUID}.ome.tif",
               RoiUID=pd.read_csv(OUTPUT_DIR+'roilist.csv')['RoiUID'].to_list())
    shell:
        "python scripts/crop_roi.py '{input}' '{OUTPUT_DIR_CROP}'"

