# Instruction for running DETM/main.py on HPC cluster with slurm and singularity
This guide specially applicable for the Greene HPC Cluster at New York University, but should be generally applicable for HPC clusters.
For general users, the prerequisites are:
- Having access to a cluster using slurm 
- The cluster has singularity container installed
- Have singularity images and overlay files prebuilt 

## To setup conda environment with Sigularity and overlay images
in a log-in node:
cd to DETM clone directory

Copy the proper gzipped overlay images from `/scratch/work/public/overlay-fs-ext3`, `overlay-5GB-200K.ext3.gz` is good enough for most conda enviorment, it has 5GB free space inside and is able to hold 200K files
```
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-5GB-200K.ext3.gz .
gunzip overlay-5GB-200K.ext3.gz
```
Choose proper Singualrity images. For PyTorch, use

`/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif`

To setup conda enviorment, fist launch container interactively 

singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash

Now inside the container, install miniconda into /ext3/miniconda3
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
export PATH=/ext3/miniconda3/bin:$PATH
conda update -n base conda -y
```
create a wrapper script /ext3/env.sh: 
```
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
```
Now exit the container 
```
exit
```
Relaunch the container 
```
singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash
source /ext3/env.sh
```
Sanity checks if miniconda is installed properly
```
$ which python
/ext3/miniconda3/bin/python
$ which pip   
/ext3/miniconda3/bin/pip
$ which conda
/ext3/miniconda3/bin/conda
$ python --version
Python 3.8.5
```

Now install packages into this base enviorment either with pip or conda.
Create a virtual env:
```
conda create --name detm --file requirements.txt 
```

