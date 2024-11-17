#!/bin/bash
#SBATCH --job-name=accediff
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --output=slurm_logs/%x-%j/main.out
#SBATCH --error=slurm_logs/%x-%j/main.err


OUTFILE="slurm_logs/%x-%j/%n.out"
ERRFILE="slurm_logs/%x-%j/%n.err"

srun -o $OUTFILE -e $ERRFILE --open-mode=append bash $@
