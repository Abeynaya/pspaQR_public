#!/bin/bash
#SBATCH --output=spaQR_%j.out
#SBATCH --time=02:00:00    
##SBATCH --part=largemem
hostname
lscpu

mpirun -n ${SLURM_NTASKS} ../build/./spaQR -m ${MAT} -t 1e-2 --skip ${SKIP} -n ${M} -d ${D} --n_threads ${NTHREADS} --verb 0
