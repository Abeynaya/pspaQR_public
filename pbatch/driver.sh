#!/bin/bash

# 3D
MAT=../mats/advdiff/advdiff_3_64_1_p1_1.mm M=64 D=3 SKIP=4 NTHREADS=2 sbatch -c 32 -n 1 run.sh
MAT=../mats/advdiff/advdiff_3_64_1_p1_1.mm M=64 D=3 SKIP=4 NTHREADS=16 sbatch -c 32 -n 1 run.sh
MAT=../mats/advdiff/advdiff_3_64_1_p1_1.mm M=64 D=3 SKIP=4 NTHREADS=16 sbatch -c 32 -n 4 run.sh