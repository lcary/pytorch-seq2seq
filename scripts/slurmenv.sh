#!/bin/bash
set -e
# usage: run slurmenv.sh in a srun pty to get going

module add openmind/singularity

repo_root="$( cd "$(dirname "$0")"; cd .. ; pwd -P )"

echo "path: ${repo_root}"
cd $repo_root
singularity shell --bind "${repo_root}" container.img
