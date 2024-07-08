#!/bin/bash
#SBATCH --job-name=multinode-video-gen
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o video_gen_deepspeed_output.log

export GPUS_PER_NODE=4

*****************************************
# Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded
*****************************************
export OMP_NUM_THREADS=1

export NCCL_DEBUG=INFO
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens

# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate videogen
#pip install accelerate==0.31.0



srun accelerate launch \
    --config_file ds_config.yaml \
    train_stage_1.py --config configs/train/stage1.yaml