#!/bin/bash
#
#SBATCH -J fitting_data

#SBATCH -o ./SLURM_output/fitting_data_%A_%a.txt

#SBATCH --account=2016394
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=45:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=general
#SBATCH --mail-user malharris19@unm.edu 
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --array=0-123 # for 124 files

cd ../src/scripts
module load miniconda3


source activate /users/malharris/miniconda3/envs/envRunningInJupyter

PT_CACHE="/tmp/pytensor_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$PT_CACHE"
export PYTENSOR_FLAGS="base_compiledir=${PT_CACHE}"

echo "Using PyTensor cache: $PT_CACHE"

cleanup() {
    rm -rf "$PT_CACHE"
}
trap cleanup EXIT

python 04_run_fit_refine.py "$SLURM_ARRAY_TASK_ID"