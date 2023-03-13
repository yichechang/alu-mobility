import pandas as pd

# this is set relative to where Snakefile is *stored* (but *not called*)
# thus there're two ways for configuring this workflow
#   1. modify the config file in-place, OR
#   2. duplicate the config file to the analysis directory, and then
#      specify its location as option when launching snakemake via 
#      commandline
configfile: workflow.source_path('snakemake_config.yaml')

pepfile: config['input']['pepfile_path']

# Code from reference:
# http://ivory.idyll.org/blog/2021-snakemake-checkpoints.html
class Checkpoint_MakePattern:
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, w):
        global checkpoints

        # wait for the results of 'check_crop_roi'; this will trigger an
        # exception until that rule has been run.
        checkpoints.crop_roi.get(**w)

        # the magic, such as it is, happens here: we create the
        # information used to expand the pattern, using arbitrary
        # Python code.
        names = self.get_names()
        pattern_expanded = expand(self.pattern, RoiUID=names)
        # TODO: 
        #   - why doesn't the following work? and how is w actually 
        #     even used in above line `checkpoints.check_crop_roi.get(**w)`
        #
        # pattern_expanded = expand(self.pattern, RoiUID=names, **w)
        return pattern_expanded

    def get_names(self):
        return pd.read_csv('roilist.csv')['RoiUID'].to_list()

def get_channel_names():
    channels = (pep.config['experiments']
                          [config['input']['experiment_type']]
                          ['channels'])
    return [c['fluoro'] for c in channels]

ALL_CH = get_channel_names()


rule all:
    input:
        Checkpoint_MakePattern(
            expand("piv/{ch}/{RoiUID}.mat", ch=ALL_CH, allow_missing=True)
        )


rule build_imagelist:
    output: 
        'imagesetlist.csv',
    shell:
        # Must quote whatever that will be used as paths as they *might*
        # contain spaces...
        "python scripts/build_imagelist.py "
                "'{config[input][raw][base_dir]}' "
                "'{config[input][raw][subdir_name]}' "
                "{config[input][raw][ext]} "
                "'{pep.config[experiments][movie][path_pattern]}' "
                "'{config[input][samplesheet_path]}' "
                "'{output}' "
                "--datefmt {pep.config[experiments][movie][datefmt]} "
                "--nafilter {config[build_imagelist][nafilter]} "
                "{config[build_imagelist][options]}"

rule draw_roi:
    input: 
        'imagesetlist.csv',
    output:
        'roilist.csv'
    script:
        "scripts/draw_roi.py"

# Note:
#   Defining output by touching a file inside the desired output folder
#   (specified same as params.outdir) ensures that folder gets created.
#   This means that the script doesn't need to check/create by itself.
#   We also don't need to explicitly do mkdir before calling the script.
checkpoint crop_roi:
    input:
        'roilist.csv'
    output:
        touch("single_nuc_movie/.done")
    params:
        outdir = "single_nuc_movie"
    script:
        "scripts/crop_roi.py"


rule split_channels:
    input:
        "single_nuc_movie/{RoiUID}.ome.tif"
    output:
        expand("single_nuc_movie/{ch}/{RoiUID}.ome.tif", 
               ch=ALL_CH, allow_missing=True)
    script:
        "scripts/split_channels.py"


# Dummy rule to target post-checkpoint rule (`split_channels`, in this
# case).
# 
# Use this as target rule instead of the actual rule `split_channels"
# as functions are only allowed for input but not output. This way, we
# don't have to modify the `all` rule just to test or target other 
# rules post checkpoint.    
rule all_split_channels:
    input:
        Checkpoint_MakePattern(
            expand("single_nuc_movie/{ch}/{RoiUID}.ome.tif", 
            ch=ALL_CH, allow_missing=True)
        )


rule segment_nuclei_in_time:
    input:
        "single_nuc_movie/{params.ch_to_seg}/{RoiUID}.ome.tif"
    output:
        "single_nuc_mask_movie/{RoiUID}.ome.tif"
    params:
        ch_to_seg = config['segment_nuclei_in_time']['channel']
    shell:
        "python scripts/segment_nuclei_in_time.py '{input}' '{output}'"


rule piv:
    input:
        "single_nuc_movie/{ch}/{RoiUID}.ome.tif"
    output:
        "piv/{ch}/{RoiUID}.mat"
    params:
        matlab = config['software']['matlab_path'],
        scriptdir = config['software']['mfile_dir'],
    shell:
        # Note that in order for {input} {output} to still be relative
        # to dir where Snakemake is invoked, we need to start matlab
        # in the same directory (hence, `-sd <dir/to/launch>` doesn't 
        # fit). 
        """
        {params.matlab} -batch "addpath('{params.scriptdir}'); matpiv_v2('{input}', '{output}')"
        """