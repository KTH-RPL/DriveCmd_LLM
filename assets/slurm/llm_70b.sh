#!/bin/bash
#SBATCH -J llm_70b
#SBATCH --gpus 8
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/workspace/llc/assets/logs/%J_70b.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/workspace/llc/assets/logs/%J_70b.err

# The '-A' SBATCH switch above is only necessary if you are member of several
# projects on Berzelius, and can otherwise be left out. 'SBATCH -A <your-project-account>'

# Apptainer images can only be used outside /home. In this example the
# image is located here
cd /proj/berzelius-2023-154/users/x_qinzh/workspace/llc
module load Anaconda/2021.05-nsc1

/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/llc/bin/torchrun --nproc_per_node 8 --rdzv-endpoint=localhost:57012 scripts/main_llama.py \
 --debug_len -1 --slurm_job_id $SLURM_JOB_ID --ckpt_dir /proj/berzelius-2023-154/users/x_qinzh/workspace/llama/llama-2-70b-chat \
 --provide_detailed_explain True --provide_few_shots True
