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
#SBATCH --partition=gpu --gres=gpu:A100 --mem-per-cpu=9600
#
# Specify project account (replace as required)
#SBATCH --account=moshfeghi-pmwc
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=10:00:00
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
# Do not change the line below
#=====================
#----------------------------------------------------------

#Modify the line below to run your program. This is an example
#=========================================================

python /users/gxb18167/EEG-To-Text/train_decoding.py --model_name BrainTranslator --generator_name WGAN_Text_2.0 --augmentation_factor 200 --task_name task1_task2_taskNRv2 --one_step --pretrained --not_load_step1_checkpoint --num_epoch_step1 20 --num_epoch_step2 30 -lr1 0.00005 -lr2 0.0000005 -b 32 -s ./checkpoints/decoding -cuda cuda:0

# Do not change the line below
#=========================================================
/opt/software/scripts/job_epilogue.sh
#----------------------------------------------------------
