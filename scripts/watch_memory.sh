#!/bin/bash
dir_path=watchmem/topdata_$(hostname)_PID$$_$(date +%Y%m%d_T%H%M)
mkdir -p $dir_path
while true; do
    top -n 1 -b -u lcary > $dir_path/top_$(date +%Y%m%d_%H%M%S).out
    sleep 5
done
