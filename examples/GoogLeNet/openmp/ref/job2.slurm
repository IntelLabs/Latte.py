#!/bin/bash
#SBATCH --job-name googelenet2a
#SBATCH --time 2:00:00
#SBATCH -N 1
#SBATCH -c 272               
#SBATCH --output googlenet2a.out
#SBATCH --partition knltrb



export LATTE_PARALLEL_STRATEGY=OPENMP
export LATTE_NUM_THREADS=64
export LATTE_MODE=RELEASE
export LATTE_PREFETCH_MODE=DISABLE
export OMP_DYNAMIC=false
srun numactl --membind=1 python reduced.py 






