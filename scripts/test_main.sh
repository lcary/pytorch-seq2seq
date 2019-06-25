#!/bin/bash
set -e
module add openmind/singularity
repo_root=/om2/user/lcary/pytorch-seq2seq
echo "path: ${repo_root}"
cd $repo_root
singularity exec --bind "${repo_root}" container.img bash run.sh
