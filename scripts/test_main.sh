#!/bin/bash
set -e

module add openmind/singularity

repo_root=/om2/user/lcary/pytorch-seq2seq
echo "path: ${repo_root}"
cd $repo_root

start_time="$(date -u +%s)"
echo "Job start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running run.sh"
$repo_root/scripts/watch_memory.sh &

singularity exec --bind "${repo_root}" container.img bash run.sh
echo "Job finished with status: ${?}"
echo "Job end: $(date '+%Y-%m-%d %H:%M:%S')"
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total elapsed time: ${elapsed} seconds"
kill %1
