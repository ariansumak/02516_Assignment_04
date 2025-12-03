#!/bin/sh
### General options
### -- specify queue --
#BSUB -q c02516
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set the job name --
#BSUB -J train_potholes
### -- ask for number of CPU cores --
#BSUB -n 4
### -- ensure all cores are on one host --
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
### -- wall time (hh:mm) --
#BSUB -W 02:00
### -- output and error files --
#BSUB -o train_potholes_%J.out
#BSUB -e train_potholes_%J.err

# === Load environment ===
module load python/3.11.9
module load cuda/12.1
source ~/visual_env/bin/activate

# === Navigate to your project ===
cd ~/visual/potholes

# === Run training ===
python train.py \
    --dataset-root ./data \
    --proposals-root ./outputs/part1/training_proposals \
    --epochs 10 \
    --batch-size 64 \
    --device cuda \
    --amp
