#!/bin/bash
#SBATCH -A berzelius-2023-256
#SBATCH -J gpt_num
#SBATCH --gpus 0
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cil@kth.se
#SBATCH --output /home/x_cili/x_cili_yy/llc/assets/logs/%J_gpt_num.out
#SBATCH --error  /home/x_cili/x_cili_yy/llc/assets/logs/%J_gpt_num.err


# Apptainer images can only be used outside /home. In this example the
# image is located here
cd /home/x_cili/x_cili_yy/llc
module load Anaconda/2021.05-nsc1

start="0"

/home/x_cili/x_cili_lic/conda/envs/llc/bin/python scripts/main_gpt.py --debug_len=-1 --start_from "$start"  \
 --slurm_job_id $SLURM_JOB_ID \
 --provide_detailed_explain False --provide_few_shots True --step_by_step True --num_shots 2