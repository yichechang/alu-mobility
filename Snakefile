configfile: 'snakemake_config.yaml'
pepfile: config['input']['pepfile']

rule build_imagelist:
    output: 
        config['output']['dir'] + 'output/imagelist.csv',
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