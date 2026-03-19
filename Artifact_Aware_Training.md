# Artifact-Aware Training via Continuous Loss Discounting

## Overview

This document specifies the implementation of an **artifact-aware training strategy** for histopathology models.

Instead of removing patches with artifacts during preprocessing, all patches are retained, and their contribution to training is **continuously discounted** based on the fraction of artifact coverage.

This approach:
- avoids repatching,
- removes the need for threshold tuning,
- preserves real-world artifact distribution,
- and maintains a simple, deterministic training policy.

---

## Core Idea

Each patch has artifact coverage values per class:

- Fold
- Darkspot & Foreign Object
- PenMarking
- Edge & Air Bubble
- Out-of-Focus (OOF)

Let:

cov_c ∈ [0, 1]  for each artifact class c

Define:

alpha_eff = max(cov_c)

Then modify the loss:

L' = L * (1 - alpha_eff)

Where:
- L is the original loss (e.g., Dice, BCE, hybrid),
- alpha_eff is the maximum artifact coverage across classes.

---

## Why max() Instead of Sum

We intentionally use:

alpha_eff = max(cov_c)

instead of:

alpha = sum(cov_c)

Reasons:
- Avoids double-counting overlapping artifacts
- Prevents over-penalization from multiple small artifacts
- Reflects that the worst artifact dominates degradation
- Avoids expensive union-area computations

---

## Patching Stage Requirements

Modify patch extraction to:

1. Compute artifact coverage per class for each patch:
   - cov_fold
   - cov_penmarking
   - cov_oof
   - cov_darkspot_foreign
   - cov_edge_airbubble

2. Do NOT drop patches based on thresholds

3. Do NOT save artifact-only debug patches

4. Store metadata per patch in a PatchIndex file (Parquet)

### PatchIndex Schema

Required:
- img_path
- mask_path
- label
- patient_id
- cov_fold
- cov_penmarking
- cov_oof
- cov_darkspot_foreign
- cov_edge_airbubble

Optional:
- x, y, level, window_size
- slide_id
- tissue_pct

---

## Training Stage Implementation

### Dataset Output

(image, mask, artifact_cov_vector)

artifact_cov_vector = [
    cov_fold,
    cov_penmarking,
    cov_oof,
    cov_darkspot_foreign,
    cov_edge_airbubble
]

---

### Loss Modification

loss_per_sample = criterion(outputs, targets, reduction='none')

alpha_eff = artifact_cov_vector.max(dim=1).values

weight = 1.0 - alpha_eff

loss = (loss_per_sample * weight).mean()

---

## Optional Stability Clamp

weight = clamp(1.0 - alpha_eff, min=0.2)

---

## Interpretation

- Clean patches → full contribution
- Artifact-heavy patches → reduced contribution
- No thresholds required

---

## Computational Impact

Advantages:
- No repatching
- Minimal overhead
- Fully vectorized
- Use of Pyarrow library

Limitations:
- No reduction in dataset size
- Does not change batch composition

---

## Summary

This replaces threshold-based filtering with continuous weighting, preserving data while controlling artifact impact.
