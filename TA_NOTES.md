# Assignment 2 TA Notes

This file summarizes the discussion so far and gives a practical implementation roadmap.

## Course Assignment Context

- Assignment: `Assignment 2: Deep models`
- Due date in PDF: `Sunday, March 29, 2026, 11:59 PM`
- Core requirement: implement and compare deep models on CIFAR-10 and shuffled-patch test sets.

## High-Level Requirements (from the assignment)

1. Use Python 3 and PyTorch.
2. Keep required file names / skeleton structure.
3. Use CIFAR-10 with `40,000` train and `10,000` validation split.
4. Normalize data using train-set mean/std.
5. Build 3 model families with at least 3 layers each:
- Fully connected
- CNN
- Transformer
6. Track and plot over epochs:
- Training vs validation loss
- Training vs validation accuracy
7. Part 1: Evaluate 3 plain models on CIFAR-10 test set and compare.
8. Part 2 (D-shuffletruffle): train on original data, evaluate on original + `test_patch_16.npz`.
9. Part 3 (N-shuffletruffle): train on original data, evaluate on original + `test_patch_8.npz`.
10. Save best variant weights for each architecture.
11. Part 4 analysis:
- Compare all 9 models on all 3 test sets
- Build 75-sample analysis set (25 original + shuffled versions)
- Extract pre-final embeddings
- Run PCA to 2D and plot
12. Do not use patch-shuffle augmentation during training.

## Current Starter Code Status

- The current `main.py` model classes are placeholders.
- All 9 classes are effectively the same `Flatten + Linear` model right now.
- You need to implement real architecture differences for:
- FC vs CNN vs Transformer
- Plain vs D vs N variants

## Step-by-Step Implementation Plan

1. Foundation pass
- Reproducible seed setup
- Train/val split fixed
- Train-set mean/std normalization
- Logging for loss/accuracy each epoch
- Checkpoint save for best validation model

2. Part 1 pass (plain CIFAR models)
- Implement plain FC/CNN/Transformer
- Train and tune each
- Plot train/val curves
- Evaluate on CIFAR-10 test
- Keep best checkpoint per architecture

3. Part 2 pass (D models, patch-16 robustness)
- Implement D-FC, D-CNN, D-Transformer
- Train on original CIFAR only
- Evaluate on original + patch-16
- Compare and retain best variant per architecture

4. Part 3 pass (N models, patch-8 robustness)
- Implement N-FC, N-CNN, N-Transformer
- Train on original CIFAR only
- Evaluate on original + patch-8
- Compare and retain best variant per architecture

5. Part 4 analysis pass
- Build one summary table for all 9 models across all 3 test sets
- Create 75-example analysis set
- Get pre-final embeddings for each model
- Run PCA (`n_components=2`) and plot labeled points
- Write interpretation of cluster behavior and robustness

6. Final report pass
- Architecture + hyperparameter descriptions
- Required plots/tables
- Interpretation, assumptions, limitations
- Final answers to the assignment analysis questions

## Deep Concept Notes: D vs N Architectures

Main idea:
- D and N should differ by spatial scale and order sensitivity, not just random hyperparameters.

Design axes:
1. Receptive field scale
- D: larger spatial context per feature
- N: smaller/local context per feature
2. Order sensitivity
- High dependence on absolute patch location hurts shuffled-patch robustness
- More order-agnostic aggregation usually improves robustness

Interpretation goal:
- D should be strong on original and robust to 16x16 patch shuffling.
- N should be strong on original and robust to 8x8 patch shuffling.

Useful metrics to track:
- `delta16 = acc_original - acc_patch16`
- `delta8 = acc_original - acc_patch8`
- `gap_dn = abs(acc_patch16 - acc_patch8)`

Reading these metrics:
- Good D tends to keep `delta16` low.
- Good N tends to keep `delta8` low.
- If both deltas are large, the model is too position-sensitive.
- If both deltas are tiny but original accuracy is poor, model may be too invariant.

## Working Agreement

- You will implement code yourself.
- I will act as TA/reviewer:
- help with design decisions
- debug issues with you
- review your results and suggest next improvements

## Immediate Next Checklist

1. Implement proper FC/CNN/Transformer for Part 1 first.
2. Add standardized training logs and plotting.
3. Share Part 1 results (metrics + plots), then we design D/N variants intentionally.
