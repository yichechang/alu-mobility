from pathlib import Path

rule gen_piv_config_json:
    output:
        expand("{protocol}_config.json", 
               protocol=config['piv']['protocol'])
    params:
        pivconfig = config['piv']['protocol_configs'][config['piv']['protocol']],
        moviemeta = config['piv']['movie_meta'],
    script:
        "../scripts/gen_piv_config_json.py"


rule piv:
    input:
        pivconfig = expand("{protocol}_config.json", 
                           protocol=config['piv']['protocol']),
        movie = "results/image_registered/single_ch/{ch}/{RoiUID}.ome.tif"
    output:
        "results/piv/{protocol}/{ch}/{RoiUID}.mat"
    params:
        protocol = config['piv']['protocol'],
        pivpkg = str(
            Path(workflow.current_basedir).parent /
            config['piv']['pkg_path']),
        mfiledir = str(Path(workflow.current_basedir).parent / "scripts"),
    threads: 1  # While our mfile can use more than one core, somehow
                # snakemake doesn't play well with it and matlab will
                # fail at starting its parpool. 
    envmodules:
        "matlab/R2019b"
    resources:
        mem = 2000, 
        time = 61, 
    shell:
        # Note that in order for {input} {output} to still be relative
        # to dir where Snakemake is invoked, we need to start matlab
        # in the same directory (hence, `-sd <dir/to/launch>` doesn't 
        # fit). 
        """
        matlab -batch "addpath('{params.mfiledir}', genpath('{params.pivpkg}')); {params.protocol}('{input.movie}', '{output}', '{input.pivconfig}', 1)"
        """


all_piv_input = [
    lambda w: expand("results/piv/{protocol}/{ch}/{RoiUID}.mat",
                     protocol=config['piv']['protocol'],
                     ch=config['piv']['channel'],
                     RoiUID=get_checkpoint_RoiUID(w))
]
