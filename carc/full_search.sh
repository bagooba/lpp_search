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
#SBATCH --array=60-70 # for 124 files

START_TIME=$SECONDS
cd ../src/scripts
module load miniconda3
source activate /users/malharris/miniconda3/envs/envRunningInJupyter

# Run your Python script with the identifier as an argument

python 02_run_quick_singles.py $SLURM_ARRAY_TASK_ID
python 03_run_periodic_search.py $SLURM_ARRAY_TASK_ID
python 04_run_fit_refine.py $SLURM_ARRAY_TASK_ID
python 05_generate_dv_pdf.py $SLURM_ARRAY_TASK_ID

ELAPSED=$(( SECONDS - START_TIME ))

printf "Total runtime: %02d:%02d:%02d\n" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
