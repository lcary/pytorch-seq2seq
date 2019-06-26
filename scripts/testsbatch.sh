#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --array=1-1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=16

repo_root=/om2/user/lcary/pytorch-seq2seq

declare -a commands
commands[1]="${repo_root}/scripts/test_main.sh"
bash -c "${commands[${SLURM_ARRAY_TASK_ID}]}"
