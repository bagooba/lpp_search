#!/bin/bash
#
#SBATCH -J full_search
#SBATCH -o ./SLURM_output/full_search_%A_%a.txt

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

# Run your Python script with the identifier as an argument

python 02_run_quick_singles.py $SLURM_ARRAY_TASK_ID
python 03_run_periodic_search.py $SLURM_ARRAY_TASK_ID

