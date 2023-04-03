# abcpiv

PIV-based analysis for chromatin from A- or B-compartments.

---

Date started: 2023-02-02

---

## Dependencies

Dependencies are listed in [workflow/envs/abcd.yaml](workflow/envs/abcd.yaml) 
file and can be installed as a conda environment using either conda or
mamba. 

Currently, with mamba which is faster at solving, we need to first 
create an empty environment before we can install dependencies specified
in an yaml file. See the [issue and solution](https://github.com/mamba-org/mamba/issues/633#issuecomment-812272143).

```
mamba create -n abcd
mamba env update -n abcd -f workflow/envs/abcd.yaml
```  

### MATLAB

You'll need to have `matlab` on your path. This can either be done by 
manually creating a symbolic link to the matlab executable, or by using
the environment modules on a cluster (e.g. `module load matlab/R2019b`).


## Executing snakemake workflow

For now, ROI annotation in the original image files can only be done
locally and not on della. Make sure to at least run to `draw_roi` to 
obtain `imagesetlist.csv` and `roilist.csv`, copy those two files to 
HPC at `/scratch/gpfs/<user>/proj/alu/<analysis-name>/`, then run the
rest of the pipeline on della.

### On local machine
`cd` to analysis folder, and run snakemake locally, optionally specify 
target rule `draw_roi` if just want to run up to that step, saving the 
remaining to run on della. For example:

```
snakemake \
  -s ~/repository/phd-analysis/abc-mobility/workflow/Snakefile \
  -c1 \
  draw_roi
```

### On della
`cd` to analysis folder, and run

```
snakemake \
  -s ~/abc-mobility/workflow/Snakefile \
  --profile ~/abc-mobility/config/princeton_rc/
```


## Versions

### Tagging system explanation
This repository currently contains both 
1. a snakemake workflow with its config files, scripts, etc; and 
2. a python package `abcdcs` that is required for the workflow, but also
   includes modules can be used on their own for upstream preprocessing
   as well as downstream analyses.

In the future, it might make sense to keep track of them separately, 
but currently their development is closely related. Thus, we now use a 
single tagging system for version tracking. 

The format is `yyyy.MM.dd.[a-z]` where `[a-z]` is used to differentiate
versions tagged on the same date.

- `2023.03.31.a`: remove unused function in msnd
- `2023.03.30.a`: **normalize intensity for y459 and y491 on della**
- `2023.03.28.a`: **Improve raw data compatibility with tiff file without metadata**
- `2023.03.26.c`: **matpiv_v2 della (used for y459)**
- `2023.03.26.b`: snakemake on local and della up to PIV
    - Workflow runs on both local (everything to piv) and della 
      (from cropping to piv).
    - No job grouping should be used.
    - On della, if want to avoid submit many small jobs (currently some
      of the corresponding rules have time set to `61` minutes when they
      take only a few minutes, to avoid piling up in the *short-job*
      queue), `salloc` then run without cluster profile is useful. 
