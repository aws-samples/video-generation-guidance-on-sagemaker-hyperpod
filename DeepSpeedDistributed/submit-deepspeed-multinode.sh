#!/bin/bash
#SBATCH --job-name=multinode-video-gen
#SBATCH -N 2
#SBATCH --ntasks-per-node=1

export GPUS_PER_NODE=4
export OMP_NUM_THREADS=1

# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate videogen

#pip install accelerate==0.31.0


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO

# AWS specific
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens
export NCCL_IGNORE_DISABLED_P2P=1

## EFA settings
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa  
export FI_EFA_USE_HUGE_PAGE=0


# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="video_gen_distributed_deepspeed_output.log"

# Update the accelerate launch command
export LAUNCHER="accelerate launch \
    --config_file ds_config.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \${SLURM_PROCID} \
    --deepspeed_multinode_launcher standard \
    "

export PROGRAM="\
train_stage_1.py \
    --config configs/train/stage1.yaml
"

export CMD="$LAUNCHER $PROGRAM"

truncate -s 0 $LOG_PATH

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"