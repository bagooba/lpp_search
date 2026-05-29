#!/bin/bash
#
#SBATCH -J organizing_data
#SBATCH -o ./SLURM_output/organizing_data_%j.txt

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

cd ../src/scripts
module load miniconda3
source activate /users/malharris/miniconda3/envs/envRunningInJupyter

# Run your Python script with the identifier as an argument
python 01_prepare_targets.py

