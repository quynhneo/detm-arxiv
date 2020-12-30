# Instruction for running DETM/main.py on slurm, with singularity
This guide specially applicable for the Greene HPC Cluster at New York University, but should be generally applicable for clusters using slurm.
Prerequisites:
- Having access to a cluster using slurm 
- The cluster uses singularity container

## To setup conda environment with Sigularity and overlay images

in DETM directory:
Copy the proper gzipped overlay images from `/scratch/work/public/overlay-fs-ext3`, `overlay-5GB-200K.ext3.gz` is good enough for most conda enviorment, it has 5GB free space inside and is able to hold 200K files
```
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-5GB-200K.ext3.gz .
gunzip overlay-5GB-200K.ext3.gz
```
