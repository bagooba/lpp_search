#!/bin/bash
#
#SBATCH -J dr2_dr3_conv
#SBATCH -o ./SLURM_output/dr2_dr3_conversion_%j.txt

#SBATCH --account=2016394
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=45:00:00
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=general
#SBATCH --mail-user malharris19@unm.edu 
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --array=0-24  # for 26 files

export GAIA_USER="mharris"
export GAIA_PASS='G#%$74u2En'

cd ../src
module load miniconda3
source activate /users/malharris/miniconda3/envs/envRunningInJupyter

# Run your Python script with the identifier as an argument
python dr2_to_dr3_conversion.py $SLURM_ARRAY_TASK_ID

