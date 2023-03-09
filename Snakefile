configfile: 'snakemake_config.yaml'
pepfile: config['input']['pepfile']

rule build_imagelist:
    output: 
        config['output']['dir'] + 'output/imagelist.csv',
    shell:
        "python scripts/build_imagelist.py "
                "'{config[input][basedir]}' "
                "'{config[input][subdir]}' "
                "'{config[input][ext]}' "
                "'{pep.config[metadata][movie][pattern]}' "
                "'{output}' "
                "--nafilter strict --patch --verbose"