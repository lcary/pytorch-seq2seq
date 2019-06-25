#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --array=1-1
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=8

repo_root=/om2/user/lcary/pytorch-seq2seq

declare -a commands
commands[1]="${repo_root}/scripts/test_main.sh"
bash -c "${commands[${SLURM_ARRAY_TASK_ID}]}"
