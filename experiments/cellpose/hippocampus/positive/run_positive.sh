#!/bin/bash
#SBATCH -A a_mar
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time 96:00:00
#SBATCH --mem=16GB

module load python/3.11
source /home/s4702415/sivenvgpu/bin/activate

srun python /scratch/user/s4702415/SigCells/experiments/cellpose/hippocampus/positive/positive_revb.py
