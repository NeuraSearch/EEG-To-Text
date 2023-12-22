#!/bin/bash

#=================================================================
#
# Job script for running a job on a single GPU (any available GPU)
#
#=================================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the gpu partition (queue) with any GPU
#SBATCH --partition=gpu --gres=gpu:A100 --mem-per-cpu=1000
#
# Specify project account (replace as required)
#SBATCH --account=moshfeghi-pmwc
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=01:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=ALL
# Job name
#SBATCH --job-name=gpu_test
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge
module load nvidia/sdk/22.3
module load anaconda/python-3.9.7/2021.11

#Uncomment the following if you are running multi-threaded
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#
#=========================================================
# Prologue script to record job details
# Do not change the line belowgit pu
#=====================
#----------------------------------------------------------

#Modify the line below to run your program. This is an example
#=========================================================
 sbatch
python /users/gxb18167/EEG-To-Text/eval_decoding_bug_fix.py --checkpoint_path /users/gxb18167/Datasets/Checkpoints/train_decoding/WGAN_Text_2.0/best/Augment_0_0_task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt --config_path /users/gxb18167/Datasets/Checkpoints/train_decoding/WGAN_Text_2.0/Augment_0_0_task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt -cuda cuda:0

# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------
