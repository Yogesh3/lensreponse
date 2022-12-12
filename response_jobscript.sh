#!/bin/bash 
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --job-name kappa_response
#SBATCH --output=/scratch/r/rbond/ymehta3/mpi_output_%j.txt
#SBATCH --mail-type=ALL
 
cd $SLURM_SUBMIT_DIR

module load NiaEnv/2019b
module load autotools
module load gcc/8.3.0
module load gsl
module load cfitsio
module load fftw
module load intelmpi

mpiexec ./response.py
# or "srun ./mpi_example"