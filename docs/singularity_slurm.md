# Instruction for running DETM on HPC cluster with CUDA, slurm and singularity
This guide specially applicable for the Greene HPC Cluster of New York University, but should be generally applicable for HPC clusters.
For general HPC users, the prerequisites are:
- Having access to a cluster 
- The cluster uses Singularity container 
- Having singularity images and overlay files prebuilt 

## To setup conda environment with Sigularity and overlay images
in a log-in node:
```
$cd DETM
```

Copy the proper gzipped overlay images from `/scratch/work/public/overlay-fs-ext3`. For example, `overlay-5GB-200K.ext3.gz` is good enough for most conda environments, it has 5GB free space inside and is able to hold 200K files:
```
$cp -rp /scratch/work/public/overlay-fs-ext3/overlay-5GB-200K.ext3.gz .
$gunzip overlay-5GB-200K.ext3.gz
```
Choose a proper singularity image. For PyTorch, use:

`/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif`

To setup conda enviorment, first launch container interactively: 

```
$singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
```
Inside the container, install miniconda into /ext3/miniconda3:
```
Singularity> wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
Singularity> sh Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
Singularity> export PATH=/ext3/miniconda3/bin:$PATH
Singularity> conda update -n base conda -y
```
create a wrapper script /ext3/env.sh: 
```
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
```
Run the wrapper:
```
Singularity> source /ext3/env.sh
```
Sanity checks if miniconda is installed properly:
```
Singularity> which python
/ext3/miniconda3/bin/python
Singularity> which conda
/ext3/miniconda3/bin/conda
Singularity> python --version
Python 3.8.5
```

Now install packages into this base environment either with pip or conda.
For example, using conda to create a virtual env:
```
Singularity> conda create --name detm --file requirements.txt 
```
Now everything is ready. Conda environment named `detm` has been installed ***inside** the singularity container. This ensure that your inode quota is not consumed, and the environment is exact.
To run `DETM/main.py`, there are now two options: interactive running (good for testing, debugging, short jobs), and batch job good for real and longer jobs. 
## Interactive mode
From a log in node, request a computing node with gpu:
```
$srun --cpus-per-task=20 --gres=gpu:1 --nodes 1 --mem=50GB --time=7-00:00:00 --pty /bin/bash
```
In the computing node:
```
$singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash
Singularity> source /ext3/env.sh
Singularity> conda activate detm
Singularity> python main.py
```
## Batch mode
Make a script such as `slurm.s` below and modify directory as needed 
```
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=detm
#SBATCH --mail-type=END
#SBATCH --mail-user=your_email_address
#SBATCH --output=slurm_%j.out

cd /scratch/$USER/DETM
overlay_ext3=/scratch/$USER/DETM/overlay-5GB-200K.ext3
singularity \
exec --nv --overlay $overlay_ext3 \
/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash \
-c "source /ext3/env.sh; \
conda activate detm; \
python main.py"
```
to submit, from a log in node:
```
$sbatch slurm.s
```
check the status of the job, and job ID:
```
$squeue -u yourusername
```
check the stdout result:
```
$cat slurm_jobid.out
```

