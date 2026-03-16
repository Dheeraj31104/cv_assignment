# CSCI-B 657 Assignment 2 — Code Documentation

## Overview

This assignment trains 9 deep learning models on CIFAR10 across three architecture families (FC, CNN, Transformer) and three task variants (Plain, D-shuffletruffle, N-shuffletruffle). The core challenge is building models that are robust to patch-shuffled test images without ever training on shuffled data.

---

## How to Run

```bash
python main.py --model_class <model> --epochs <n> --batch_size 128 --learning_rate 0.01 --l2_regularization 0.0
```

Available model classes:
```
Plain-Old-CIFAR10-FC         D-shuffletruffle-FC         N-shuffletruffle-FC
Plain-Old-CIFAR10-CNN        D-shuffletruffle-CNN        N-shuffletruffle-CNN
Plain-Old-CIFAR10-Attention  D-shuffletruffle-Attention  N-shuffletruffle-Attention
```

Example:
```bash
python main.py --model_class Plain-Old-CIFAR10-CNN --epochs 100 --learning_rate 0.001
```

---

## Output Files

| File | Description |
|------|-------------|
| `./logs/{model_class}.csv` | Per-epoch training log: `epoch, train_loss, train_acc, val_loss, val_acc` |
| `./logs/test_summary.csv` | Final test results for all models (appended after each run) |

Val columns are `nan` on epochs where validation is not evaluated (every 10 epochs by default).

---

## Core Concept: Shuffle Invariance

The key insight of this assignment is the difference between **position-sensitive** and **position-invariant** models.

### Plain models — position-sensitive
Process the full image. Spatial layout (where things are) is used to make predictions. Shuffling patches at test time breaks the spatial structure → accuracy drops significantly.

### Shuffletruffle models — position-invariant
Built around three steps that together guarantee invariance:
1. **Split** the image into patches
2. **Encode** each patch independently with shared weights (no information crosses patch boundaries)
3. **Mean-pool** across all patch features — since addition is commutative, order never matters

```
shuffle patches → same patches in different order → same mean → same prediction
```

This invariance is baked into the **architecture**, not the training data. Models are trained only on normal CIFAR10 images.

---

## Shared Utility

### `extract_patches(x, patch_size)` — [main.py:19](main.py#L19)

```python
def extract_patches(x, patch_size):
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    B, C, nph, npw, ph, pw = x.shape
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    return x.reshape(B, nph * npw, C, ph, pw)
```

Splits a batch of images into non-overlapping patches using `unfold`.

| Input | Output |
|-------|--------|
| `(B, 3, 32, 32)` | `(B, num_patches, 3, patch_size, patch_size)` |

- `patch_size=16` → `num_patches=4` (2×2 grid)
- `patch_size=8`  → `num_patches=16` (4×4 grid)
- `patch_size=4`  → `num_patches=64` (8×8 grid)

`unfold(dim, size, step)` slides a window of `size` along `dim` with stride `step`. Two consecutive unfolds on H and W dimensions carve out a 2D grid of patches. The `permute` reorders axes so patches are grouped together before the final `reshape`.

---

## FC Models

### `Net_FC` — [main.py:30](main.py#L30)

Plain fully-connected network. Sees the entire image as a flat vector.

```
Input (3, 32, 32)
    → Flatten → (3072,)
    → Linear(3072 → 512) + BatchNorm1d(512) + ReLU
    → Linear(512 → 256)  + BatchNorm1d(256) + ReLU
    → Linear(256 → 128)  + BatchNorm1d(128) + ReLU   # 3rd hidden layer
    → Linear(128 → 10)
```

- **BatchNorm1d** after each linear normalizes activations between layers, reducing internal covariate shift and stabilizing training.
- Extra hidden layer `256→128` gives the network more capacity before the final projection.

Sensitive to patch shuffling because pixel positions are encoded implicitly by their index in the flattened vector.

---

### `Net_D_shuffletruffle_FC` — [main.py:46](main.py#L46)

Invariant to **16×16** patch shuffling. Splits image into 4 patches, encodes each with shared FC layers, then mean-pools.

```
Input (B, 3, 32, 32)
    → extract_patches(16) → (B, 4, 3, 16, 16)
    → reshape            → (B*4, 768)         # treat each patch as its own input
    → Linear(768 → 256) + BatchNorm1d(256) + ReLU
    → Linear(256 → 128) + BatchNorm1d(128) + ReLU
    → Linear(128 → 64)  + BatchNorm1d(64)  + ReLU   # each patch → 64-d vector
    → reshape + mean(dim=1) → (B, 64)               # mean-pool across 4 patches
    → Linear(64 → 128) + ReLU                        # deeper classifier head
    → Linear(128 → 10)
```

`patch_dim = 3 × 16 × 16 = 768`

- **BatchNorm1d** operates on `(B*4, features)` — the 4x patch multiplier just increases the effective batch size, so stats are well-estimated.
- Classifier is now 2-layer (`64→128→10`) instead of a single linear.

Why invariant to patch-16 but not patch-8: The FC encoder flattens the full 16×16 patch — pixel positions within the patch are encoded by index. Shuffling 8×8 sub-blocks within a 16×16 patch changes those indices → output changes.

---

### `Net_N_shuffletruffle_FC` — [main.py:74](main.py#L74)

Invariant to **8×8** patch shuffling (and automatically to 16×16 as well). Same design as D-FC but with smaller patches.

```
Input (B, 3, 32, 32)
    → extract_patches(8) → (B, 16, 3, 8, 8)
    → reshape            → (B*16, 192)
    → Linear(192 → 128) + BatchNorm1d(128) + ReLU
    → Linear(128 → 64)  + BatchNorm1d(64)  + ReLU
    → Linear(64 → 32)   + BatchNorm1d(32)  + ReLU   # each patch → 32-d vector
    → reshape + mean(dim=1) → (B, 32)               # mean-pool across 16 patches
    → Linear(32 → 64) + ReLU                         # deeper classifier head
    → Linear(64 → 10)
```

`patch_dim = 3 × 8 × 8 = 192`

- **BatchNorm1d** operates on `(B*16, features)` — 16 patches per image gives even larger effective batch size, making BN stats very stable.
- Classifier is now 2-layer (`32→64→10`) instead of a single linear.

Also invariant to patch-16 because shuffling 16×16 regions just rearranges groups of 8×8 patches, and mean-pool is agnostic to order.

---

## CNN Models

### `ResidualBlock` — [main.py:104](main.py#L104)

Standard residual block used in `Net_CNN`.

```
input x
  ├─→ Conv(3×3, stride) → BN → ReLU → Conv(3×3) → BN
  └─→ shortcut (identity or 1×1 conv if dims change)
        ↓ add
      ReLU → output
```

The shortcut connection lets gradients flow directly during backprop, enabling training of deeper networks without vanishing gradients.

---

### `Net_CNN` — [main.py:135](main.py#L135)

Plain ResNet-style CNN. Processes the full image at every layer.

```
Input (B, 3, 32, 32)
    → stem: Conv(3→32, 3×3) + BN + ReLU        → (B, 32, 32, 32)
    → layer1: ResBlock(32→32) × 2               → (B, 32, 32, 32)
    → layer2: ResBlock(32→64, stride=2) + ResBlock(64→64)  → (B, 64, 16, 16)
    → layer3: ResBlock(64→128, stride=2) + ResBlock(128→128) → (B, 128, 8, 8)
    → AdaptiveAvgPool(1,1) → Flatten            → (B, 128)
    → Linear(128 → 10)
```

`stride=2` in layer2 and layer3 progressively downsamples the feature map, expanding the receptive field so later layers see large portions of the image. This is why it's position-sensitive — spatial relationships are encoded in the feature maps.

---

### `Net_D_shuffletruffle_CNN` — [main.py:171](main.py#L171)

Invariant to **16×16** patch shuffling. Each 16×16 patch is processed by the same **ResNet-style encoder** with shared weights. Conv kernels never cross patch boundaries.

```
Input (B, 3, 32, 32)
    → extract_patches(16) → (B, 4, 3, 16, 16)
    → reshape             → (B*4, 3, 16, 16)   # 4 patches per image, all batched together
    → stem: Conv(3→32, 3×3) + BN + ReLU        # 16×16
    → ResBlock(32→32)                           # 16×16 — identity shortcut
    → ResBlock(32→64,  stride=2)                # 16×16 → 8×8 — projection shortcut
    → ResBlock(64→128, stride=2)                # 8×8   → 4×4 — projection shortcut
    → AdaptiveAvgPool(1,1) → flatten            → (B*4, 128)
    → reshape + mean(dim=1) → (B, 128)          # mean-pool across 4 patches
    → Linear(128 → 64) + ReLU
    → Linear(64 → 10)
```

Using ResNet over plain conv gives skip connections — each block learns residuals rather than full transformations, enabling richer per-patch features and more stable gradient flow.

Not invariant to patch-8 because ResBlocks process the interior of each 16×16 patch with full spatial awareness — rearranging 8×8 blocks within a patch changes what the kernels see.

---

### `Net_N_shuffletruffle_CNN` — [main.py:215](main.py#L215)

Invariant to **8×8** patch shuffling. Same **ResNet-style** design as D-CNN but with 8×8 patches (16 patches total) and a scaled-down encoder.

```
Input (B, 3, 32, 32)
    → extract_patches(8) → (B, 16, 3, 8, 8)
    → reshape            → (B*16, 3, 8, 8)
    → stem: Conv(3→16, 3×3) + BN + ReLU        # 8×8
    → ResBlock(16→16)                           # 8×8 — identity shortcut
    → ResBlock(16→32,  stride=2)                # 8×8 → 4×4 — projection shortcut
    → ResBlock(32→64,  stride=2)                # 4×4 → 2×2 — projection shortcut
    → AdaptiveAvgPool(1,1) → flatten            → (B*16, 64)
    → reshape + mean(dim=1) → (B, 64)           # mean-pool across 16 patches
    → Linear(64 → 32) + ReLU
    → Linear(32 → 10)
```

Smaller channels (16→32→64 vs 32→64→128) because 8×8 patches contain less spatial information than 16×16.

Comparison with D-CNN encoder:

| | D-shuffletruffle CNN | N-shuffletruffle CNN |
|--|---------------------|---------------------|
| Patch size | 16×16 | 8×8 |
| stem | Conv(3→32) | Conv(3→16) |
| layer1 | ResBlock(32→32) | ResBlock(16→16) |
| layer2 | ResBlock(32→64, s=2) | ResBlock(16→32, s=2) |
| layer3 | ResBlock(64→128, s=2) | ResBlock(32→64, s=2) |
| output dim | 128 | 64 |

---

## Attention / Transformer Models

### `TransformerBlock` — [main.py:234](main.py#L234)

Standard Pre-LN transformer block (layer norm applied before attention and MLP, not after).

```
input x
  ├─→ LayerNorm → MultiheadAttention → residual add
  └─→ LayerNorm → MLP (Linear → GELU → Dropout → Linear → Dropout) → residual add
```

- **MultiheadAttention**: each token attends to all other tokens. Computes similarity between every pair of tokens and uses it to mix information.
- **MLP**: per-token feedforward network with hidden size `dim × mlp_ratio` (default 4×).
- **Pre-LN**: more stable training than Post-LN for shorter training runs.

---

### `Net_Attention` — [main.py:256](main.py#L256)

Vision Transformer (ViT) WITH positional embeddings — position-sensitive.

```
Input (B, 3, 32, 32)
    → extract_patches(4) → (B, 64, 3, 4, 4)
    → flatten patches    → (B, 64, 48)
    → Linear(48 → 128)   → (B, 64, 128)    # patch embedding
    → LayerNorm          → (B, 64, 128)    # embed_norm: stabilizes token scale
    → prepend CLS token  → (B, 65, 128)
    → add pos_embed      → (B, 65, 128)    # ← encodes position
    → TransformerBlock × 4
    → LayerNorm
    → CLS token [0]      → (B, 128)
    → Linear(128 → 256) + GELU + Dropout(0.1)
    → Linear(256 → 10)
```

Key components:
- **`cls_token`**: a learnable token prepended to the sequence. After all transformer blocks, its representation aggregates information from all patches and is used for classification.
- **`pos_embed`**: learnable positional embedding added to each token. Without it, the transformer would treat patches as an unordered set. This is what makes the plain model position-sensitive.
- **`embed_norm`**: LayerNorm applied immediately after patch embedding — normalizes token magnitudes before positional embeddings are added, stabilizing early training.
- **2-layer MLP head**: replaces a single linear layer — `Linear(128→256)→GELU→Dropout→Linear(256→10)` gives the classifier more expressive power.
- **`trunc_normal_` init**: `patch_embed.weight`, `cls_token`, and `pos_embed` are all initialized from a truncated normal (std=0.02), following the original ViT paper for stable training.
- **patch_size=4** → 64 patches, which gives the model fine spatial resolution.

---

### `Net_D_shuffletruffle_Attention` — [main.py:288](main.py#L288)

ViT WITHOUT positional embeddings, using **16×16 patches**. No positional info → shuffle invariant.

```
Input (B, 3, 32, 32)
    → extract_patches(16) → (B, 4, 3, 16, 16)
    → flatten patches     → (B, 4, 768)
    → Linear(768 → 128)   → (B, 4, 128)    # patch embedding
    → LayerNorm           → (B, 4, 128)    # embed_norm
    # NO pos_embed added
    → TransformerBlock × 4
    → LayerNorm
    → mean(dim=1)         → (B, 128)       # mean-pool tokens
    → Linear(128 → 256) + GELU + Dropout(0.1)
    → Linear(256 → 10)
```

Three differences from `Net_Attention`:
1. **No `pos_embed`** — attention treats all 4 tokens as an unordered set. Shuffling them produces the same attention weights (self-attention is permutation equivariant without positional info).
2. **Mean-pool instead of CLS** — no CLS token needed; averaging all token outputs is equivalent and simpler.
3. **`trunc_normal_` init on `patch_embed.weight`** (std=0.02) — same as the plain ViT for consistent initialization.

Why these together guarantee invariance: Without positional embeddings, `TransformerBlock` is permutation-equivariant — shuffling input tokens just shuffles output tokens in the same order. Mean-pool then destroys the order entirely.

---

### `Net_N_shuffletruffle_Attention` — [main.py:313](main.py#L313)

ViT WITHOUT positional embeddings, using **8×8 patches**. Same design as D-Attention but finer grain.

```
Input (B, 3, 32, 32)
    → extract_patches(8) → (B, 16, 3, 8, 8)
    → flatten patches    → (B, 16, 192)
    → Linear(192 → 128)  → (B, 16, 128)   # patch embedding
    → LayerNorm          → (B, 16, 128)   # embed_norm
    # NO pos_embed added
    → TransformerBlock × 4
    → LayerNorm
    → mean(dim=1)        → (B, 128)       # mean-pool 16 tokens
    → Linear(128 → 256) + GELU + Dropout(0.1)
    → Linear(256 → 10)
```

`trunc_normal_` init on `patch_embed.weight` (std=0.02). Invariant to both 8×8 and 16×16 shuffling by the same mean-pool argument as D-Attention.

---

## Evaluation

### `eval_model(model, data_loader, criterion, device)` — [main.py:336](main.py#L336)

Runs the model in `eval()` mode (disables dropout, uses running BatchNorm stats) over an entire dataloader. Returns average loss and accuracy (%).

---

## Training Loop — `main()` — [main.py:356](main.py#L356)

### Dataset setup
- Downloads CIFAR10 to `./data/`
- Splits 50k train set into **40k train / 10k validation** using a fixed seed (seed=0) for reproducibility
- Test set: standard CIFAR10 test set (10k images)
- `transform`: only `ToTensor()` — no normalization or augmentation in the skeleton

### Model selection
A simple `if/elif` chain maps the `--model_class` string to the appropriate class and moves it to device (CPU or CUDA).

### Optimizer
Adam with the specified learning rate and L2 regularization (weight decay).

### Training loop
```
for each epoch:
    for each batch:
        forward pass → compute loss → backward → optimizer step

    every 10 epochs:
        evaluate on validation set, print val_loss and val_acc

    log to CSV: epoch, train_loss, train_acc, val_loss, val_acc
```

### CSV logging
- **`./logs/{model_class}.csv`**: one row per epoch. `val_loss` and `val_acc` are `nan` on non-evaluated epochs.
- **`./logs/test_summary.csv`**: one row appended per run with final test results across all 3 test sets.

### Final evaluation
After training, the model is evaluated on:
1. Standard CIFAR10 test set
2. `test_patch_16.npz` — images with shuffled 16×16 patches
3. `test_patch_8.npz`  — images with shuffled 8×8 patches

---

## Model Comparison (20 epoch results)

### Test Accuracy

| Model | CIFAR10 test | patch-16 | patch-8 | patch-16 drop |
|-------|-------------|----------|---------|---------------|
| Plain FC | 30.4% | 21.6% | 20.1% | -8.8% |
| D-shuffletruffle FC | 35.4% | 35.4% | 22.3% | 0.00% |
| N-shuffletruffle FC | 38.1% | 38.1% | 38.1% | 0.00% |
| Plain CNN | 78.8% | 40.8% | 20.1% | -38.0% |
| D-shuffletruffle CNN | 63.6% | 63.6% | 39.0% | 0.00% |
| N-shuffletruffle CNN | 59.2% | 59.2% | 59.2% | 0.00% |
| Plain Attention | 16.7% | 15.9% | 15.8% | -0.8% |
| D-shuffletruffle Attention | 31.7% | 31.7% | 24.1% | 0.00% |
| N-shuffletruffle Attention | 24.5% | 24.5% | 24.5% | 0.00% |

### Key observations

**All shuffletruffle models achieve exactly 0% accuracy drop on their target patch size** — the invariance is mathematically guaranteed by the mean-pool architecture.

**D-shuffletruffle models correctly drop on patch-8** (e.g. D-CNN: 63.6% → 39.0%) because they process 16×16 patches with spatial awareness internally.

**N-shuffletruffle models are invariant to both patch-16 and patch-8** because 16×16 shuffling is just rearranging groups of 8×8 patches, which mean-pool handles transparently.

**Attention models underperform** at 20 epochs with lr=0.01. Transformers converge slower and need a lower learning rate (0.001) with warmup.

---

## Running on a SLURM Cluster

Use `run_all.sh` to submit all 9 models as a job array — each model gets its own GPU job and runs in parallel.

```bash
mkdir -p slurm_logs
sbatch run_all.sh
```

Key SLURM settings in `run_all.sh`:

| Setting | Value | Notes |
|---------|-------|-------|
| `--account` | `pclamd` | cluster account |
| `--partition` | `general` | check with `sinfo -s` |
| `--gres` | `gpu:1` | 1 GPU per job |
| `--mem` | `40G` | RAM per job |
| `--time` | `24:00:00` | max wall time |
| `--array` | `0-8` | 9 jobs, one per model |

Monitor jobs:
```bash
squeue -u $USER          # running jobs
sacct -j <JOBID>         # status after completion
```

Logs are written to `slurm_logs/{jobname}_{jobid}_{arrayid}.out`.

---

## File Structure

```
.
├── main.py              # all 9 models + training loop
├── dataset_class.py     # PatchShuffled_CIFAR10 dataset loader
├── create_data.py       # script that created the .npz test files
├── run_all.sh           # SLURM job array script — runs all 9 models in parallel
├── test_patch_16.npz    # 10k test images with shuffled 16×16 patches
├── test_patch_8.npz     # 10k test images with shuffled 8×8 patches
├── data/                # CIFAR10 raw data (downloaded automatically, gitignored)
├── logs/                # CSV logs (created at runtime, gitignored)
│   ├── {model_class}.csv
│   └── test_summary.csv
├── slurm_logs/          # SLURM stdout/stderr (created at runtime)
└── code.md              # this file
```
