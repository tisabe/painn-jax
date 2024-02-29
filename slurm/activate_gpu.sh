export PYTHONNOUSERSITE=True
module purge
module load anaconda/3/2023.03
module load cuda/11.6
module load cudnn/8.7.0
module list
BASE_DIR=/u/bechtelt/painn-jax
source ~/painn-jax/venv/bin/activate
