#!/bin/bash
#SBATCH -A a_mar
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time 48:00:00
#SBATCH --mem=16GB
#SBATCH --array=1-500

cd /scratch/user/s4702415/Honours/models/test_cellpose
module load python/3.11
source /home/s4702415/sivenvgpu/bin/activate

srun python /scratch/user/s4702415/SigCells/experiments/genesegnet/simulated/null/calculate.py