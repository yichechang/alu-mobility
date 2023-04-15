# ======================================================================
# wildcards
# ======================================================================

# Exclude forward slash in wildcards matching.
# (0) via setting allowed classes of characters
# (1) it's critical to set these regex constrains, to reduce chance of 
#     bad parsing for "ch1/UID1.ome.tif" with pattern 
#     "{ch}/{RoiUID}.ome.tif" to yield wildcard RoiUID = "ch1/UID1"
# (2) this should work globally, EXCEPT FOR inside an glob_wildcards() 
#     call where you need to specify manually! See [related issue](https://github.com/snakemake/snakemake/issues/1726) 
wildcard_constraints:
    RoiUID = "[\w-]+",
    ch = "[\w-]+",
    protocol = "[\w-]+",
    structure = "[\w-]+",


def get_channel_names():
    return [c['fluoro'] for c in config['input']['channels']]
ALL_CH = get_channel_names()

# TODO: Move to config
ALL_PROTOCOLS = ['matpiv_v2']

def get_imageset_files(wildcards):
    from abcdcs import curate
    files = []
    image_path = config['input']['image_path']
    globbed = Path(image_path['dir']).rglob(image_path['glob'])
    globbed_fpaths = [p for p in globbed if p.is_file()]
    for p in globbed_fpaths:
        if re.search(image_path['regex'], str(p)):
            files.append(str(p))
        else:
            print(f"Excluding file globbed but cannot be parsed: "
                  f"{str(p)}.")
    return files

# ======================================================================
# get wildcards from checkpoint
# ======================================================================

#
# This can be used inside an input function, to get {RoiUID}.
#
def get_checkpoint_RoiUID(wildcards):
    checkpoints.crop_roi.get(**wildcards)
    return glob_wildcards("results/image/multi_ch/{RoiUID, [\w-]+}.ome.tif").RoiUID
