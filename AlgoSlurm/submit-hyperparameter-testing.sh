#!/bin/bash
#SBATCH --job-name=video-gen
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o video-gen-stage-1_%a.out
#SBATCH --array=0-3  # Adjust the range based on the number of snr_gamma values

export OMP_NUM_THREADS=1

# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate videogen

# Define an array of snr_gamma values to test
snr_gamma_values=(0.0 1.0 2.0 3.0)

# Get the current job index from the Slurm job array
index=$SLURM_ARRAY_TASK_ID

# Get the corresponding snr_gamma value for the current job
snr_gamma=${snr_gamma_values[$index]}

# Create a temporary copy of the stage1.yaml file
cp configs/train/stage1.yaml configs/train/stage1_temp_${index}.yaml

# Modify the snr_gamma value in the temporary stage1.yaml file using sed
sed -i "s/snr_gamma:.*/snr_gamma: $snr_gamma/" configs/train/stage1_temp_${index}.yaml

srun accelerate launch train_stage_1.py --config configs/train/stage1_temp_${index}.yaml

# Clean up the temporary stage1.yaml file
rm configs/train/stage1_temp_${index}.yaml