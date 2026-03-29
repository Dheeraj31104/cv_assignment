# CSCI-B 657 Assignment 2 — Deep Models on CIFAR-10

## Overview

This assignment trains and evaluates 9 deep learning models on CIFAR-10 across three architecture families (FC, CNN, Transformer/Attention) and three task variants (Plain, D-shuffletruffle, N-shuffletruffle). The core challenge is building models that are **robust to patch-shuffled test images without ever training on shuffled data** — the invariance must be baked into the architecture.

---

## Quick Start

```bash
python main.py --model_class <model> --epochs 100 --batch_size 128 --learning_rate 0.01 --l2_regularization 0.0
```

**All 9 model classes:**
```
Plain-Old-CIFAR10-FC         D-shuffletruffle-FC         N-shuffletruffle-FC
Plain-Old-CIFAR10-CNN        D-shuffletruffle-CNN        N-shuffletruffle-CNN
Plain-Old-CIFAR10-Attention  D-shuffletruffle-Attention  N-shuffletruffle-Attention
```

**Run all 9 in parallel on SLURM:**
```bash
mkdir -p slurm_logs
sbatch run_all.sh
```

---

## Assignment Requirements

- Python 3 + PyTorch
- CIFAR-10: 40k train / 10k validation / 10k test split
- Normalize using train-set mean/std: mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
- Each model family must have at least 3 layers
- Track and plot training vs validation loss and accuracy per epoch
- **Part 1** — Plain models: evaluate on standard CIFAR-10 test set
- **Part 2** — D-shuffletruffle: train on original, test on original + `test_patch_16.npz`
- **Part 3** — N-shuffletruffle: train on original, test on original + `test_patch_8.npz`
- **Part 4** — Compare all 9 models; build 75-sample analysis set (25 original + shuffled), extract pre-final embeddings, run PCA to 2D and plot
- Do **not** use patch-shuffle augmentation during training

---

## Core Concept: Shuffle Invariance via Mean-Pool Architecture

The key architectural insight: **split → encode → mean-pool**.

```
images → extract patches → encode each patch independently (shared weights) → mean-pool → classify
```

Since mean-pool (addition) is commutative, shuffling patches in any order produces the **same output**. Invariance is guaranteed by math, not learned from shuffled training data.

- **D-shuffletruffle**: 16×16 patches → invariant to 16×16 shuffling
- **N-shuffletruffle**: 8×8 patches → invariant to both 8×8 and 16×16 shuffling (coarser shuffling just rearranges groups of fine patches)

Plain models use positional information (flat index for FC, spatial conv feature maps for CNN, learned `pos_embed` for Attention) — all of which break under patch shuffling.

---

## Architecture Decisions and Rationale

### FC Models

**Plain-Old-CIFAR10-FC**
- `Flatten → Linear(3072→512) + BN + ReLU → Linear(512→256) + BN + ReLU → Linear(256→128) + BN + ReLU → Linear(128→10)`
- BatchNorm after each linear to reduce internal covariate shift and stabilize training
- 3 hidden layers for sufficient depth without overfitting
- Position-sensitive: pixel positions are encoded implicitly by their flattened index

**D-shuffletruffle-FC** (16×16 patch invariant)
- `extract_patches(16) → (B×4, 768) → Linear(768→256)+BN+ReLU → Linear(256→128)+BN+ReLU → Linear(128→64)+BN+ReLU → mean-pool → Linear(64→128)+ReLU → Linear(128→10)`
- Shared FC encoder across all 4 patches (batched together as B×4)
- 2-layer classifier head post-pool for more expressiveness
- **Not** invariant to patch-8 because pixels within each 16×16 patch are still position-sensitive by flat index

**N-shuffletruffle-FC** (8×8 patch invariant, and 16×16 by extension)
- `extract_patches(8) → (B×16, 192) → Linear(192→128)+BN+ReLU → Linear(128→64)+BN+ReLU → Linear(64→32)+BN+ReLU → mean-pool → Linear(32→64)+ReLU → Linear(64→10)`
- 16 patches instead of 4 — BN stats even more stable due to large effective batch
- Smaller per-patch feature dim (32 vs 64) since 8×8 patches hold less information than 16×16

---

### CNN Models

**ResidualBlock** used throughout: two 3×3 convolutions with BN, skip connection for gradient flow. Projection shortcut (1×1 conv) used when spatial dimensions or channels change.

**Plain-Old-CIFAR10-CNN**
- `stem Conv(3→32) → ResBlock(32→32)×2 → ResBlock(32→64, stride=2)+ResBlock(64→64) → ResBlock(64→128, stride=2)+ResBlock(128→128) → AdaptiveAvgPool → Linear(128→10)`
- stride=2 progressively downsamples, building receptive field: full spatial awareness → position-sensitive
- Why ResNet over plain conv: skip connections let each block learn residuals, enabling deeper stable training

**D-shuffletruffle-CNN** (16×16 patch invariant)
- Each 16×16 patch processed by shared ResNet encoder: `stem Conv(3→32) → ResBlock(32→32) → ResBlock(32→64, stride=2) → ResBlock(64→128, stride=2) → AdaptiveAvgPool` → mean-pool → `Linear(128→64)+ReLU → Linear(64→10)`
- Conv kernels never cross patch boundaries
- **Not** invariant to patch-8: residual blocks process 16×16 patches with spatial awareness internally

**N-shuffletruffle-CNN** (8×8 patch invariant)
- Same design as D-CNN but patch_size=8, scaled-down channels: `stem Conv(3→16) → ResBlock(16→16) → ResBlock(16→32, stride=2) → ResBlock(32→64, stride=2) → AdaptiveAvgPool` → mean-pool → `Linear(64→32)+ReLU → Linear(32→10)`
- Smaller channels (16→32→64 vs 32→64→128) because 8×8 patches hold less spatial information

---

### Attention / Transformer Models

**TransformerBlock** (Pre-LN): `LayerNorm → MultiheadAttention → residual add → LayerNorm → MLP(GELU+Dropout) → residual add`. Pre-LN chosen over Post-LN for more stable training on shorter runs.

**Plain-Old-CIFAR10-Attention** (ViT with positional embeddings)
- `extract_patches(4) → flatten → Linear(48→128) → LayerNorm → + cls_token → + pos_embed → TransformerBlock×4 → LayerNorm → CLS[0] → Linear(128→256)+GELU+Dropout → Linear(256→10)`
- patch_size=4 → 64 tokens, fine spatial resolution
- `pos_embed`: learnable positional embedding makes model position-sensitive (encodes where each patch is)
- `cls_token`: aggregates global info from all patches for classification
- `embed_norm`: LayerNorm immediately after patch embedding stabilizes token magnitudes before pos_embed is added
- `trunc_normal_` init (std=0.02) on patch_embed.weight, cls_token, pos_embed — follows original ViT for stable training
- Why patch_size=4 for plain: maximizes spatial resolution for the position-sensitive model

**D-shuffletruffle-Attention** (ViT without positional embeddings, 16×16 patches)
- Same as plain but: **no pos_embed**, **no cls_token**, **mean-pool** instead of CLS
- Self-attention is permutation-equivariant without positional info → shuffling inputs shuffles outputs in same order → mean-pool destroys order → fully invariant
- Larger patches (16×16) → only 4 tokens, lower compute; still captures enough per-patch features

**N-shuffletruffle-Attention** (ViT without positional embeddings, 8×8 patches)
- Same design as D-Attention but patch_size=8 → 16 tokens
- Invariant to both 8×8 and 16×16 shuffling by the same mean-pool argument
- Finer-grained patches allow the transformer to attend across more distinct local features

---

## Experimental History

### Initial State (commits `2a0ad00`, `531aa63`)
Starter code had 9 model classes that were all effectively the same `Flatten + Linear` placeholder. No real architecture differences, no normalization, no proper train/val split.

**Changes made:**
- Implemented all 9 real architectures from scratch
- Added fixed-seed (seed=0) reproducible train/val split (40k/10k)
- Added CIFAR-10 mean/std normalization using train-set statistics
- Added per-epoch CSV logging (loss + accuracy for train and val)
- Added best-checkpoint saving (based on validation accuracy)
- Added evaluation on patch-shuffled test sets

---

### Early Architecture Iteration (`6677948` Model Improvements, `8d03f9b` Fine tuning)

**Issues encountered:**
- Initial FC models had too few layers and no BatchNorm → unstable, plateaued early
- Shuffletruffle FC models had single-layer classifier heads post-pool → insufficient capacity
- CNN model used plain convolutions without skip connections → training instability at depth
- Attention model used Post-LN transformer → less stable than Pre-LN for short training runs
- Attention model had no `embed_norm` → token magnitudes varied widely after patch embedding

**Corrections made:**
- Added 3rd hidden layer to FC and a 2-layer head to shuffletruffle FC variants
- Added BatchNorm1d after every FC layer in all FC models
- Replaced plain CNN with ResNet-style residual blocks
- Switched Attention to Pre-LN transformer
- Added `embed_norm` (LayerNorm after patch embedding, before pos_embed)
- Added `trunc_normal_` initialization for patch_embed, cls_token, pos_embed

**Reason for ResNet:** Skip connections allow deeper training without vanishing gradients. They make each block learn a residual correction rather than a full transformation, which is much easier to optimize.

---

### SLURM Setup (`510bab6`)

All 9 models were submitted as a job array on the IU LAIR cluster (account: `pclamd`, partition: `general`, 1 GPU + 40G RAM per job, 24h wall time). A separate `run_attention.sh` was added later for re-running attention models independently with different hyperparameters.

---

### Run 1 — Regression Run (Job 16611, backup in `backup/`)

**Config:** LR=1e-4, L2=0.2, data augmentation added (RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation), val eval every epoch, best checkpoint saving

**What we tried:** Reduce overfitting by dropping LR 100× and adding L2 + augmentation simultaneously.

**What went wrong:**
- LR 0.01 → 1e-4: models failed to converge in 100 epochs — too slow for Adam on CIFAR-10 FC/CNN
- L2=0.2: over-regularized, crushed all models
- Augmentation alone is fine, but it increases effective task difficulty and needs higher LR or more epochs to compensate
- Changing three things at once made it impossible to identify which caused the regression

| Model | Test Acc | Patch-16 | Patch-8 |
|---|---:|---:|---:|
| Plain-Old-CIFAR10-FC | 42.78% | 26.99% | 21.98% |
| D-shuffletruffle-FC | 24.38% | 24.38% | 21.63% |
| N-shuffletruffle-FC | 19.29% | 19.29% | 19.29% |
| Plain-Old-CIFAR10-CNN | 55.10% | 42.80% | 28.43% |
| D-shuffletruffle-CNN | 33.10% | 33.10% | 29.98% |
| N-shuffletruffle-CNN | 21.27% | 21.27% | 21.27% |
| Plain-Old-CIFAR10-Attention | 27.17% | 27.16% | 27.16% |
| D-shuffletruffle-Attention | 21.26% | 21.26% | 18.19% |
| N-shuffletruffle-Attention | 21.67% | 21.67% | 21.67% |

**Lesson:** Change one variable at a time. LR and regularization interact strongly with augmentation.

---

### Run 2 — FC/CNN Recovered (Job 16620, backup in `backup_20260327_014149/`)

**Config:** LR=0.01, L2=0.0, data augmentation kept, best checkpoint saving

**Changes from Run 1:** Restored LR to 0.01, removed L2. Augmentation was kept because it's inherently beneficial; only the LR/L2 change caused the regression.

**Result:** FC and CNN fully recovered and improved significantly. Attention still underperforming.

| Model | Test Acc | Patch-16 | Patch-8 | Notes |
|---|---:|---:|---:|---|
| Plain-Old-CIFAR10-FC | 60.81% | 28.00% | 21.34% | Recovered |
| D-shuffletruffle-FC | 52.86% | 52.86% | 29.01% | Improved |
| N-shuffletruffle-FC | 54.13% | 54.13% | 54.13% | Improved |
| Plain-Old-CIFAR10-CNN | **89.84%** | 64.42% | 37.68% | Best CNN yet |
| D-shuffletruffle-CNN | **82.55%** | 82.55% | 57.34% | Big improvement |
| N-shuffletruffle-CNN | **68.23%** | 68.23% | 68.23% | Big improvement |
| Plain-Old-CIFAR10-Attention | 25.98% | 23.58% | 23.21% | Wrong LR |
| D-shuffletruffle-Attention | 32.69% | 32.69% | 23.86% | Wrong LR |
| N-shuffletruffle-Attention | 27.53% | 27.53% | 27.53% | Wrong LR |

**Key insight discovered:** Attention models require a **lower learning rate (1e-4)** than FC/CNN. Transformer training is inherently less stable at LR=0.01 — the large attention weight updates cause oscillations and prevent convergence. FC/CNN models, being simpler feedforward architectures, converge well at LR=0.01.

---

### Run 3 — Per-Architecture LR, Final (Jobs 16628–16637, backup in `backup_20260327_093957/`)

**Config:** LR=0.01 for FC/CNN, LR=1e-4 for Attention, L2=0.0, augmentation kept, best checkpoint saving

**Changes from Run 2:** Attention models got their own LR=1e-4. FC/CNN kept LR=0.01 (confirmed best in Run 2).

| Model | Test Acc | Patch-16 | Patch-8 | Notes |
|---|---:|---:|---:|---|
| Plain-Old-CIFAR10-FC | 60.81% | 28.00% | 21.34% | Consistent |
| D-shuffletruffle-FC | 52.86% | 52.86% | 29.01% | Consistent |
| N-shuffletruffle-FC | 54.13% | 54.13% | 54.13% | Consistent |
| Plain-Old-CIFAR10-CNN | **90.03%** | 62.18% | 35.52% | Best across all runs |
| D-shuffletruffle-CNN | 82.46% | 82.46% | 55.80% | Strong shuffle robustness |
| N-shuffletruffle-CNN | 67.39% | 67.39% | 67.39% | Fully invariant |
| Plain-Old-CIFAR10-Attention | **76.55%** | 55.41% | 55.49% | Fully recovered (was ~26%) |
| D-shuffletruffle-Attention | 54.92% | 54.92% | 27.86% | Recovered vs Run 2 |
| N-shuffletruffle-Attention | **63.95%** | 63.95% | 63.95% | Best Attention shuffle model |

**Outcome:** CNN peaked at 90.03%. Attention recovered from ~27% → 76.55% on clean data. Per-architecture LR confirmed as the correct final approach.

---

### Config Change Summary

| Change | Run | Effect |
|---|---|---|
| LR 0.01 → 1e-4 (all models) | Run 1 | Catastrophic — FC/CNN didn't converge in 100 epochs |
| L2 0.0 → 0.2 | Run 1 | Over-regularized, hurt all models severely |
| Data augmentation added | Run 1 | Good long-term but masked by LR/L2 issues |
| LR 1e-4 → 0.01, L2 removed | Run 1→2 | FC/CNN fully recovered and improved; Attention still wrong |
| Attention LR 0.01 → 1e-4 | Run 2→3 | Attention fully recovered to 76.55%; FC/CNN unchanged |
| Val eval every epoch + best checkpoint | Run 1 onward | Consistently positive: saves best model, not final model |

---

## PCA Visualization

**Script:** `plot_pca_embeddings.py`

Builds a 75-sample analysis set (25 original CIFAR-10 test images + their 16×16 and 8×8 shuffled versions), extracts pre-final layer embeddings from each model, runs PCA to 2D, and saves labeled scatter plots.

**Issue encountered:** Initial plot used filled markers with similar colors — all three variants (original, patch16, patch8) appeared as overlapping solid colored dots. When points from different variants landed near each other in PCA space, they occluded each other and the legend colors were indistinguishable.

**Fix applied:** Switched to **hollow markers** (`facecolor='none'`) with distinct edge colors:
- `original`: blue circle `o`
- `patch16`: red triangle `^`
- `patch8`: green square `s`

Hollow markers allow overlapping points to remain simultaneously visible since they don't fully occlude each other.

**Plots saved to:** `pca_plots/`
**Backups:** `pca_plots/backup_20260329_092304/` (pre-hollow-marker style)

---

## Output Files

| File/Directory | Description |
|---|---|
| `logs/{model_class}.csv` | Per-epoch: epoch, train_loss, train_acc, val_loss, val_acc |
| `logs/test_summary.csv` | Final test results across all 3 test sets per run |
| `checkpoints/{model_class}.pt` | Best validation checkpoint for each model |
| `plots/` | Training curve plots (loss + accuracy over epochs) |
| `pca_plots/` | PCA scatter plots for all 9 models |
| `slurm_logs/` | SLURM stdout/stderr logs |
| `results_summary.md` | Numerical results table across all experimental runs |
| `code.md` | Detailed architecture documentation |

---

## Key Invariance Properties

| Model | Original | Patch-16 | Patch-8 | Why |
|---|---|---|---|---|
| Plain-FC/CNN/Attention | Sensitive | Drops | Drops | Uses pixel/spatial positions |
| D-shuffletruffle-* | Good | **0% drop** | Drops | 16×16 patch mean-pool |
| N-shuffletruffle-* | Good | **0% drop** | **0% drop** | 8×8 patch mean-pool; 16×16 = rearranging 8×8 groups |

---

## SLURM Settings

| Setting | Value |
|---|---|
| Account | `pclamd` |
| Partition | `general` |
| GPU | 1 per job |
| Memory | 40G |
| Wall time | 24:00:00 |
| Array | 0-8 (one job per model) |

Monitor jobs:
```bash
squeue -u $USER          # running jobs
sacct -j <JOBID>         # status after completion
```
