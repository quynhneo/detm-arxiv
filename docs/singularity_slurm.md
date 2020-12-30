# Instruction for running DETM/main.py on HPC cluster with slurm and singularity
This guide specially applicable for the Greene HPC Cluster at New York University, but should be generally applicable for HPC clusters.
For general users, the prerequisites are:
- Having access to a cluster using slurm 
- The cluster has singularity container installed
- Have singularity images and overlay files prebuilt 

## To setup conda environment with Sigularity and overlay images
in a log-in node:
`$cd <DETM directory>`

Copy the proper gzipped overlay images from `/scratch/work/public/overlay-fs-ext3`, `overlay-5GB-200K.ext3.gz` is good enough for most conda environment, it has 5GB free space inside and is able to hold 200K files
```
$cp -rp /scratch/work/public/overlay-fs-ext3/overlay-5GB-200K.ext3.gz .
$gunzip overlay-5GB-200K.ext3.gz
```
Choose a proper singualrity image. For PyTorch, use

`/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif`

To setup conda enviorment, fist launch container interactively 

```
$singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
```
Inside the container, install miniconda into /ext3/miniconda3
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
Now exit the container 
```
exit
```
Relaunch the container 
```
$singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash
Singularity> source /ext3/env.sh
```
Sanity checks if miniconda is installed properly
```
Singularity> which python
/ext3/miniconda3/bin/python
Singularity> which pip   
/ext3/miniconda3/bin/pip
Singularity> which conda
/ext3/miniconda3/bin/conda
Singularity> python --version
Python 3.8.5
```

Now install packages into this base enviorment either with pip or conda.
Create a virtual env:
```
Singularity> conda create --name detm --file requirements.txt 
```
Now everything is ready. Conda environment has been installed in the singularity container. This ensure that your inode quota is not consumed, and the environment is exact.
To run `DETM/main.py`, there are now two options: interactive running (for testing, debugging, short job), and batch job for real and longer jobs. 
## Interactive run
from a log in node, request a computing node with gpu 
```
$srun --cpus-per-task=20 --gres=gpu:1 --nodes 1 --mem=50GB --time=7-00:00:00 --pty /bin/bash
```
In the computing node,
```
$singularity exec --overlay overlay-5GB-200K.ext3 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash
Singularity> source /ext3/env.sh
Singularity> conda activate detm
Singularity> python main.py
```
## Submitting a job through slurm
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
overlay_ext3=/scratch/qmn203/DETM/overlay-5GB-200K.ext3
singularity \
exec --nv --overlay $overlay_ext3 \
/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash \
-c "source /ext3/env.sh; \
conda activate detm; \
python main.py"
```
to submit:
```
$sbatch slurm.s
```
check the status of the job, and job ID:
```
$squeue -u yourusername
```
check the result:
```
$cat slurm_jobid.out
```

