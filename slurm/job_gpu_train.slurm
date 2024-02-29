#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./slurm/output/painn_test.%j.out
#SBATCH -e ./slurm/output/painn_test.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J painn_jax
#
#SBATCH --nodes=1            # Request 1 or more full nodes
#SBATCH --constraint="gpu"   # Request a GPU node
#SBATCH --gres=gpu:a100:1    # Use one a100 GPU
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --mem=64000          # memory in mB
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#SBATCH --time=12:00:00

# load the environment with modules and python packages
source ~/painn-jax/slurm/activate_gpu.sh
cd ~/painn-jax

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python validate.py --profile --epochs 10 --val-freq 1
