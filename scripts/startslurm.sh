#!/bin/bash
set -e
cd "$( dirname "${BASH_SOURCE[0]}" )"
sbatch testsbatch.sh
#cat slurm-<jobid>.out
