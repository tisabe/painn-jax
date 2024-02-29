#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./slurm/output/profile_test.%j.out
#SBATCH -e ./slurm/output/profile_test.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J painn_jax
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=1
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=00:15:00
#SBATCH --partition=gpudev

# load the environment with modules and python packages
source ~/painn-jax/slurm/activate_gpu.sh
cd ~/painn-jax

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python gpu_profile_example.py
