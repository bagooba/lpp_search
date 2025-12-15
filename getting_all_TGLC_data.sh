#!/bin/bash
#
#SBATCH -J get_TGLC_datafiles
#SBATCH -o ./SLURM_output/sector_TGLC_files%j.txt

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
#SBATCH --array=0-50  # for ~51 files

module load miniconda3
source activate /users/malharris/miniconda3/envs/envRunningInJupyter

# Run your Python script with the identifier as an argument
python Executing_3_getting_TGLC_public_data.py $SLURM_ARRAY_TASK_ID

