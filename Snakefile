configfile: 'snakemake_config.yaml'
pepfile: config['input']['pepfile']

rule build_imagelist:
    output: 
        config['output']['dir'] + 'imagesetlist.csv',
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
        config['output']['dir'] + 'imagesetlist.csv'
    output:
        config['output']['dir'] + 'roilist.csv'
    shell:
        "python scripts/draw_roi.py '{input}' '{output}'"