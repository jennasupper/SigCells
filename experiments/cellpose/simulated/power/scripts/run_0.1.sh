#!/bin/bash
#SBATCH -A pawsey1073
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=work
#SBATCH --time 24:00:00
#SBATCH --mem=16GB
#SBATCH --array=1-50

module load python/3.11.6
source /scratch/pawsey1073/jsupper/sivenv/bin/activate

srun -N 1 -n 1 -c 1 python /scratch/pawsey1073/jsupper/SigCells/experiments/cellpose/simulated/power/calculate.py 0.1