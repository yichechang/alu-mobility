import pandas as pd

configfile: 'snakemake_config.yaml'
pepfile: config['input']['pepfile']

OUTPUT_DIR = config['output']['dir'] 

# Code from reference:
# http://ivory.idyll.org/blog/2021-snakemake-checkpoints.html
class Checkpoint_MakePattern:
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, w):
        global checkpoints

        # wait for the results of 'check_crop_roi'; this will trigger an
        # exception until that rule has been run.
        checkpoints.check_crop_roi.get(**w)

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
        return pd.read_csv(OUTPUT_DIR+'roilist.csv')['RoiUID'].to_list()


rule all:
    input:
        Checkpoint_MakePattern(OUTPUT_DIR + "single_nuc_mask_movie/{RoiUID}.ome.tif")

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
        touch(OUTPUT_DIR + ".crop_roi.done")
    shell:
        """
        mkdir -p '{OUTPUT_DIR_CROP}'
        python scripts/crop_roi.py '{input}' '{OUTPUT_DIR_CROP}'
        """


checkpoint check_crop_roi:
    input: 
        OUTPUT_DIR + ".crop_roi.done"
    output: 
        touch(OUTPUT_DIR + '.check_crop_roi.done')


rule segment_nuclei_in_time:
    input:
        OUTPUT_DIR + "single_nuc_movie/{RoiUID}.ome.tif"
    output:
        OUTPUT_DIR + "single_nuc_mask_movie/{RoiUID}.ome.tif"
    shell:
        "python scripts/segment_nuclei_in_time.py '{input}' '{output}' --ch 1"
