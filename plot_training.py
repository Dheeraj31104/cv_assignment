"""
Plot training/validation accuracy and loss curves for all models.
Reads CSVs from ./logs/ and saves figures to ./plots/.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LOGS_DIR = './logs'
PLOTS_DIR = './plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASETS = ['Plain-Old-CIFAR10', 'D-shuffletruffle', 'N-shuffletruffle']
ARCHS    = ['FC', 'CNN', 'Attention']
COLORS   = {'FC': '#1f77b4', 'CNN': '#ff7f0e', 'Attention': '#2ca02c'}


def load_csv(path):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            train_acc.append(float(row['train_acc']))
            vl = row['val_loss']
            va = row['val_acc']
            val_loss.append(float(vl) if vl not in ('nan', '') else np.nan)
            val_acc.append(float(va) if va not in ('nan', '') else np.nan)
    return (np.array(epochs), np.array(train_loss), np.array(train_acc),
            np.array(val_loss), np.array(val_acc))


# ── 1. One figure per dataset (2 rows × 3 cols: top=accuracy, bottom=loss) ──
for dataset in DATASETS:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{dataset} — Training Curves', fontsize=14, fontweight='bold')

    for col, arch in enumerate(ARCHS):
        name = f'{dataset}-{arch}'
        csv_path = os.path.join(LOGS_DIR, f'{name}.csv')
        if not os.path.exists(csv_path):
            for ax in axes[:, col]:
                ax.set_visible(False)
            continue

        epochs, tr_loss, tr_acc, vl_loss, vl_acc = load_csv(csv_path)

        # val points only where not nan
        val_mask = ~np.isnan(vl_acc)

        ax_acc  = axes[0, col]
        ax_loss = axes[1, col]

        # --- Accuracy ---
        ax_acc.plot(epochs, tr_acc, label='Train', color=COLORS[arch], linewidth=1.5)
        ax_acc.plot(epochs[val_mask], vl_acc[val_mask],
                    label='Val', color=COLORS[arch], linewidth=1.5,
                    linestyle='--', marker='o', markersize=4)
        ax_acc.set_title(f'{arch}', fontsize=12)
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_xlabel('Epoch')
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, alpha=0.3)

        # --- Loss ---
        ax_loss.plot(epochs, tr_loss, label='Train', color=COLORS[arch], linewidth=1.5)
        ax_loss.plot(epochs[val_mask], vl_loss[val_mask],
                     label='Val', color=COLORS[arch], linewidth=1.5,
                     linestyle='--', marker='o', markersize=4)
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f'{dataset}_curves.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ── 2. One figure per architecture (comparing datasets, 2 rows × 3 cols) ──
DATASET_COLORS = {
    'Plain-Old-CIFAR10': '#1f77b4',
    'D-shuffletruffle':  '#ff7f0e',
    'N-shuffletruffle':  '#2ca02c',
}

for arch in ARCHS:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{arch} Models — Training Curves by Dataset', fontsize=14, fontweight='bold')

    col = 0
    for dataset in DATASETS:
        name = f'{dataset}-{arch}'
        csv_path = os.path.join(LOGS_DIR, f'{name}.csv')
        if not os.path.exists(csv_path):
            col += 1
            continue

        epochs, tr_loss, tr_acc, vl_loss, vl_acc = load_csv(csv_path)
        val_mask = ~np.isnan(vl_acc)
        c = DATASET_COLORS[dataset]

        ax_acc  = axes[0, col]
        ax_loss = axes[1, col]

        ax_acc.plot(epochs, tr_acc, label='Train', color=c, linewidth=1.5)
        ax_acc.plot(epochs[val_mask], vl_acc[val_mask],
                    label='Val', color=c, linestyle='--',
                    marker='o', markersize=4, linewidth=1.5)
        ax_acc.set_title(dataset, fontsize=11)
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_xlabel('Epoch')
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, alpha=0.3)

        ax_loss.plot(epochs, tr_loss, label='Train', color=c, linewidth=1.5)
        ax_loss.plot(epochs[val_mask], vl_loss[val_mask],
                     label='Val', color=c, linestyle='--',
                     marker='o', markersize=4, linewidth=1.5)
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        col += 1

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f'{arch}_curves.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ── 3. Combined overview: all 9 models on one page ──
fig, axes = plt.subplots(6, 3, figsize=(15, 22))
fig.suptitle('All Models — Training Curves (top=Acc, bottom=Loss per dataset)', fontsize=13, fontweight='bold')

for col, arch in enumerate(ARCHS):
    for row_pair, dataset in enumerate(DATASETS):
        name = f'{dataset}-{arch}'
        csv_path = os.path.join(LOGS_DIR, f'{name}.csv')

        ax_acc  = axes[row_pair * 2,     col]
        ax_loss = axes[row_pair * 2 + 1, col]

        # Column header on first row
        if row_pair == 0:
            ax_acc.set_title(arch, fontsize=12, fontweight='bold')

        if not os.path.exists(csv_path):
            ax_acc.set_visible(False)
            ax_loss.set_visible(False)
            continue

        epochs, tr_loss, tr_acc, vl_loss, vl_acc = load_csv(csv_path)
        val_mask = ~np.isnan(vl_acc)
        c = DATASET_COLORS[dataset]

        ax_acc.plot(epochs, tr_acc,  color=c, linewidth=1.2, label='Train')
        ax_acc.plot(epochs[val_mask], vl_acc[val_mask],
                    color=c, linestyle='--', marker='o', markersize=3,
                    linewidth=1.2, label='Val')
        ax_acc.set_ylabel(f'{dataset}\nAcc (%)', fontsize=8)
        ax_acc.legend(fontsize=7)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.tick_params(labelsize=7)

        ax_loss.plot(epochs, tr_loss, color=c, linewidth=1.2, label='Train')
        ax_loss.plot(epochs[val_mask], vl_loss[val_mask],
                     color=c, linestyle='--', marker='o', markersize=3,
                     linewidth=1.2, label='Val')
        ax_loss.set_ylabel('Loss', fontsize=8)
        ax_loss.set_xlabel('Epoch', fontsize=8)
        ax_loss.legend(fontsize=7)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.tick_params(labelsize=7)

plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'all_models_overview.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
