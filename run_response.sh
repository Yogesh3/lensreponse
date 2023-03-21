#!/bin/bash
#for i in {600..700..100}
#do 
#    mpiexec -n 10 python response.py --space harmonic --fft_normalize no --nsims "$i" --sim_version release
#done
mpiexec -n 20 python response.py --space harmonic --fft_normalize no --R_def R2 --channel MV --iso --sim_version release
mpiexec -n 20 python response.py --space harmonic --fft_normalize no --R_def R2 --channel MV --aniso --sim_version release
