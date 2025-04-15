#!/bin/bash
#SBATCH -J SampleBenchmark ## Job Name
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu ## Request 1 GPU   
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/slac/slurmlogs/outLog_%x_%j.txt ### Output Log File (Optional)
#SBATCH -e /scratch/slac/slurmlogs/errLog_%x_%j.txt ### Error Log File (Optional but suggest to have it)
#SBATCH -t 20:00:00 ### Job Execution Time
#SBATCH --mail-user=benjamindrabeck@ucsb.edu
#SBATCH --mail-type ALL 

LR=0.001
srun python sample_benchmark.py --method random_oversampling --num_epochs 10 --learning_rate 0.0005 --batch_size 64 --model ResNet --outdir ./results --verbose
