#!/bin/bash
# Wait for job 16620 to finish, backup everything, resubmit fresh runs,
# wait again, then regenerate all plots and summarize.

source /l/anaconda3-2024.02/etc/profile.d/conda.sh
conda activate torch-env

cd /u/dhkara/cv_assignment

echo "Waiting for job 16620 to finish..."
while squeue -j 16620 2>/dev/null | grep -q 16620; do
    sleep 60
done
echo "Job 16620 done."

# ── Backup logs ──
BACKUP_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/backup_$BACKUP_TS
mv logs/*.csv logs/backup_$BACKUP_TS/ 2>/dev/null && echo "Logs backed up to logs/backup_$BACKUP_TS/"

# ── Backup plots ──
mkdir -p plots/backup_$BACKUP_TS
mv plots/*.png plots/backup_$BACKUP_TS/ 2>/dev/null && echo "Plots backed up to plots/backup_$BACKUP_TS/"

# ── Backup pca plots ──
mkdir -p pca_plots/backup_$BACKUP_TS
mv pca_plots/*.png pca_plots/backup_$BACKUP_TS/ 2>/dev/null && echo "PCA plots backed up to pca_plots/backup_$BACKUP_TS/"

# ── Submit fresh runs ──
JOB_ID=$(sbatch run_all.sh | awk '{print $4}')
echo "Submitted fresh job: $JOB_ID"

# ── Wait for fresh runs ──
echo "Waiting for job $JOB_ID to finish..."
while squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID; do
    sleep 60
done
echo "Job $JOB_ID done."

# ── Regenerate plots ──
echo "Regenerating training plots..."
python plot_training.py

echo "Regenerating PCA plots..."
python plot_pca_embeddings.py

echo "Running summarize..."
python summarize.py

echo "All done."
