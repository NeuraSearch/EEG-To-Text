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
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

#Uncomment the following if you are running multi-threaded
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#
#=========================================================
# Prologue script to record job details
# Do not change the line belowgit pu
#=====================
#----------------------------------------------------------sbatch cd


#Modify the line below to run your program. This is an example
#=========================================================

python /users/gxb18167/EEG-To-Text/eval_decoding_bug_fix.py --checkpoint_path /users/gxb18167/Datasets/Checkpoints/train_decoding/Generation_size_Word_Level_batch_size_64_g_d_learning_rate2e-05_2e-05_word_embedding_dim_50_z_size_100_num_epochs_100_device_cuda:0_model_final/best/Augment_v2_2_TF-IDF-Low_task1_task2_taskNRv2_finetune_WGAN_v1_Text_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt --config_path /users/gxb18167/Datasets/Checkpoints/train_decoding/Generation_size_Word_Level_batch_size_64_g_d_learning_rate2e-05_2e-05_word_embedding_dim_50_z_size_100_num_epochs_100_device_cuda:0_model_final/Augment_v2_2_TF-IDF-Low_task1_task2_taskNRv2_finetune_WGAN_v1_Text_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json

# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.shsqueue
#----------------------------------------------------------
