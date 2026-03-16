#!/bin/bash
#SBATCH --job-name=cv_a2
#SBATCH --account=pclamd
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j_%a.out
#SBATCH --error=slurm_logs/%x_%j_%a.err
#SBATCH --array=0-8

# activate conda env
source ~/.bashrc
conda activate torch-env

mkdir -p slurm_logs

# map array index to model class
MODELS=(
    "Plain-Old-CIFAR10-FC"
    "D-shuffletruffle-FC"
    "N-shuffletruffle-FC"
    "Plain-Old-CIFAR10-CNN"
    "D-shuffletruffle-CNN"
    "N-shuffletruffle-CNN"
    "Plain-Old-CIFAR10-Attention"
    "D-shuffletruffle-Attention"
    "N-shuffletruffle-Attention"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Array ID   : $SLURM_ARRAY_TASK_ID"
echo "Model      : $MODEL"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================="

cd $SLURM_SUBMIT_DIR

python main.py \
    --model_class "$MODEL" \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --l2_regularization 0.0

echo "Done: $MODEL"
