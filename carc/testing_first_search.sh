#!/bin/bash
#
#SBATCH -J test_first_search_w_plots
#SBATCH -o ./SLURM_output/test_first_search_w_plots%j.txt

#SBATCH --account=2016394
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=45:00:00
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=general
#SBATCH --mail-user malharris19@unm.edu 
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --array=0-116 # for 116 files

# Change to the directory containing the Python script
cd ../src
module load miniconda3
source activate /users/malharris/miniconda3/envs/envRunningInJupyter

# Run your Python script with the identifier as an argument
python Executing_code_part_2_init_single_search.py $SLURM_ARRAY_TASK_ID

cd ../src/scripts
python 02_run_quick_singles.py $SLURM_ARRAY_TASK_ID
