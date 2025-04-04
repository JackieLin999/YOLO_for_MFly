#!/bin/bash
# The interpreter used to execute the script

# SBATCH directives that convey submission options:


#SBATCH --job-name=yolo_finetune
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jaclin@umich.edu
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64g
#SBATCH --time=3:30:00
#SBATCH --account=eecs442s015w25_class
#SBATCH --partition=gpu
#SBATCH --output=yolo_run_2.log
#SBATCH --error=yolo_run_2_%j.err


module purge
module load python/3.12
source env/bin/activate
python run_7.py