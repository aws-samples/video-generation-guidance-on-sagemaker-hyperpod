# SageMaker Hyperpod Cluster with Slurm 

### Running Experiments

#### Single Node with Single GPUs Job
```bash
sbatch submit-animateanyone-algo.sh
```


The slurm job will allocate single node (single GPU if g5.2xlarge instance), activate the training envoniroment and run accelerate launch train_stage_1.py --config configs/train/stage1.yaml

Note: For smaller GPU instances (e.g., G5 2xlarge), adjust `train_bs: 2` `train_width: 256 train_height: 256 ` and   `use_8bit_adam: True` to avoid out-of-memory issues.


#### Hyperparameter Testing
```bash
sbatch submit-hyperparameter-testing.sh
```

Define an array of snr_gamma values to test snr_gamma_values=(0.0 1.0 2.0 3.0) and allocate 4 job to run 4 different configuration 
accelerate launch train_stage_1.py --config configs/train/stage1_temp_${index}.yaml, each job will take 1 node (single GPU if g5.2xlarge instance). 


### Monitoring Experiments

Use MLflow for visualization:

```bash
mlflow ui --backend-store-uri ./mlruns/
```
