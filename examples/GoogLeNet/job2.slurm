#!/bin/bash
#SBATCH --job-name googelenet2a
#SBATCH --time 2:00:00
#SBATCH -N 1
#SBATCH -c 112 
#SBATCH --output googlenet2a.out
#SBATCH --partition skxtrb



export LATTE_PARALLEL_STRATEGY=SIMPLE_LOOP
export LATTE_NUM_THREADS=112
export LATTE_MODE=RELEASE
export LATTE_PREFETCH_MODE=ON
export LATTE_TIMER=OFF
export LATTE_VEC_CONFIG=AVX-512
srun python 3.py 






