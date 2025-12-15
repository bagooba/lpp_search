#!/bin/bash
#
#SBATCH -J initial_search_w_plots
#SBATCH -o ./SLURM_output/initial_search_w_plots%j.txt

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
#SBATCH --array=0-125  # for 105 files

module load miniconda3
source activate /users/malharris/miniconda3/envs/envRunningInJupyter

# Run your Python script with the identifier as an argument
python Executing_code_part_2_init_single_search.py $SLURM_ARRAY_TASK_ID

