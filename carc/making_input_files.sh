#!/bin/bash
#
#SBATCH -J making_data
#SBATCH -o ./SLURM_output/making_data_%A_%a.txt
#SBATCH --account=2016394
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=45:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --partition=general
#SBATCH --mail-user malharris19@unm.edu 
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --array=0-301%20 # for 301*7261 of 1242178407 - 200 files

cd ../src
module load miniconda3
source activate /users/malharris/miniconda3/envs/envRunningInJupyter

# Run your Python script with the identifier as an argument
python Executing_code_part_1_organzing_fits_files.py "$SLURM_ARRAY_TASK_ID"