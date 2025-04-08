#!/bin/bash
#SBATCH -J CNN_Baseline_30Epochs_0.01LR
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o ~/capstone-SLAC/logs/outLog_%x_%j.txt
#SBATCH -e ~/capstone-SLAC/logs/errLog_%x_%j.txt
#SBATCH -t 15:00:00

LR=0.001
srun -c 2 python ./__main__.py --num_workers 4 --nepoch 20 --lr $LR



#srun -c 4 /home/reesekaro/train_wrapper.sh $LR

#cd $SLURM_SUBMIT_DIR

#/bin/hostname
#python ~/capstone-SLAC/Model_trainer.py