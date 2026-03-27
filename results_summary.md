# Experimental Results Summary

## Experimental Runs

### Run 1 — Regression Run (backup2/ — Job 16611)
**Config:** LR=1e-4, L2=0.2, data augmentation added (RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation), val eval every epoch, best checkpoint saving

**Changes introduced:**
- LR dropped 100× (0.01 → 1e-4) — models failed to converge in 100 epochs
- L2 regularization introduced (0.2) — over-regularized
- Data augmentation added — harder training, needs more epochs/higher LR to compensate

**Result: Massive regression across all models**

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

---

### Run 2 — FC/CNN Recovered (backup_20260327_014149/ — Job 16620)
**Config:** LR=0.01, L2=0.0, data augmentation kept, val eval every epoch, best checkpoint saving

**Changes from Run 1:**
- LR restored to 0.01 — FC and CNN models fully recovered
- L2 removed — regularization was hurting convergence
- Attention models still used LR=0.01 (incorrect for this architecture — needs 1e-4)

**Result: FC/CNN recovered and improved; Attention still underperforming**

| Model | Test Acc | Patch-16 | Patch-8 | Notes |
|---|---:|---:|---:|---|
| Plain-Old-CIFAR10-FC | 60.81% | 28.00% | 21.34% | On par with Run 1 |
| D-shuffletruffle-FC | 52.86% | 52.86% | 29.01% | Slight improvement |
| N-shuffletruffle-FC | 54.13% | 54.13% | 54.13% | Improved vs Run 1 |
| Plain-Old-CIFAR10-CNN | **89.84%** | 64.42% | 37.68% | Best CNN yet |
| D-shuffletruffle-CNN | **82.55%** | 82.55% | 57.34% | Big improvement |
| N-shuffletruffle-CNN | **68.23%** | 68.23% | 68.23% | Big improvement |
| Plain-Old-CIFAR10-Attention | 25.98% | 23.58% | 23.21% | Wrong LR |
| D-shuffletruffle-Attention | 32.69% | 32.69% | 23.86% | Wrong LR |
| N-shuffletruffle-Attention | 27.53% | 27.53% | 27.53% | Wrong LR |

**Key insight:** Attention models require a lower learning rate (1e-4) to converge stably, while FC/CNN models need the higher LR (0.01). Using LR=0.01 for Attention causes unstable/poor training.

---

### Run 3 — Per-Architecture LR (Final)
**Config:** LR=0.01 for FC/CNN, LR=1e-4 for Attention, L2=0.0, data augmentation kept, best checkpoint saving

**Changes from Run 2:**
- Attention models get LR=1e-4
- FC/CNN keep LR=0.01 (confirmed best in Run 2)

**Result: Best of both worlds — strong FC/CNN + Attention fully recovered**

| Model | Test Acc | Patch-16 | Patch-8 | Notes |
|---|---:|---:|---:|---|
| Plain-Old-CIFAR10-FC | 60.81% | 28.00% | 21.34% | Consistent with Run 2 |
| D-shuffletruffle-FC | 52.86% | 52.86% | 29.01% | Consistent with Run 2 |
| N-shuffletruffle-FC | 54.13% | 54.13% | 54.13% | Consistent with Run 2 |
| Plain-Old-CIFAR10-CNN | **90.03%** | 62.18% | 35.52% | Best CNN across all runs |
| D-shuffletruffle-CNN | 82.46% | 82.46% | 55.80% | Strong shuffle robustness |
| N-shuffletruffle-CNN | 67.39% | 67.39% | 67.39% | Fully invariant |
| Plain-Old-CIFAR10-Attention | **76.55%** | 55.41% | 55.49% | Fully recovered with correct LR |
| D-shuffletruffle-Attention | 54.92% | 54.92% | 27.86% | Recovered vs Run 2 |
| N-shuffletruffle-Attention | **63.95%** | 63.95% | 63.95% | Best Attention shuffle model |

**Key result:** Per-architecture LR confirmed as the right approach. CNN peaked at 90.03%, Attention recovered from ~27% to 76.55% on clean data.

---

## Config Change Summary

| Change | Run | Effect |
|---|---|---|
| LR 0.01 → 1e-4 (all models) | Run 1 | Catastrophic — models didn't converge in 100 epochs |
| L2 0.0 → 0.2 | Run 1 | Over-regularized, hurt all models |
| Data augmentation added | Run 1 | Good idea but needs correct LR; masked by LR/L2 issues |
| LR 1e-4 → 0.01, L2 removed | Run 1→2 | FC/CNN fully recovered and improved; Attention still wrong |
| Per-model LR (Attention=1e-4) | Run 2→3 | Attention fully recovered; FC/CNN gains maintained |
| Val eval every epoch + best checkpoint | Run 1 onward | Positive — saves best model, not final model |
