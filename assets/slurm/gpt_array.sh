#!/bin/bash

declare -a starts=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

for idx in "${!starts[@]}"; do
    sbatch assets/slurm/gpt.sh "${starts[$idx]}"
done
