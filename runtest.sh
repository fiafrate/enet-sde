#!/bin/bash
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=enet.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # essential
export OMP_PROC_BIND=spread                  # recommended
export OMP_PLACES=cores                      # recommended
export OMP_SCHEDULE=static                   # recommended
export OMP_DISPLAY_ENV=verbose               # good to know

# Load necessary modules
source /sw/batch/init.sh
module switch env/2023Q4-gcc-openmpi


source local_env/bin/activate 
# Navigate to the working directory
# cd $SSD

# Execute your Python script
python3 enet-test-multi.py

# End of script