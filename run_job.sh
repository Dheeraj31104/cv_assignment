#!/bin/bash
#SBATCH -J cv_a2
#SBATCH -A pclamd
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/%j_%a.out
#SBATCH -e slurm_logs/%j_%a.err
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@iu.edu

source /l/anaconda3-2024.02/etc/profile.d/conda.sh
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
    --batch_size 256 \
    --learning_rate 1e-4 \
    --l2_regularization 0.0

echo "Done: $MODEL"
