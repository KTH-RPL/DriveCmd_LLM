#!/bin/bash
#SBATCH -A berzelius-2023-154
#SBATCH -J gpt
#SBATCH --gpus 0
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yiya@kth.se
#SBATCH --output /home/x_yiyan/x_yiyanj/code/llc/assets/logs/%J.out
#SBATCH --error  /home/x_yiyan/x_yiyanj/code/llc/assets/logs/%J.err


# Apptainer images can only be used outside /home. In this example the
# image is located here
cd /home/x_yiyan/x_yiyanj/code/llc
module load Anaconda/2021.05-nsc1

start="$1"

/home/x_yiyan/x_yiyanj/tools/mambaforge/envs/llvm/bin/python scripts/main_gpt.py  --step_by_step True --provide_few_shots True --debug_len=-1 \
 --start_from "$start" \
 --slurm_job_id $SLURM_JOB_ID
