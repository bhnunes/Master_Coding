# Stage 7 Upgrade Plan: Phikon-v2, Protected Retention, and Optional GIST

## Goal

Upgrade Stage 7 smart sampling to use pathology-domain embeddings from `Phikon-v2`, add protected retention for clinically important patches, and preserve GIST as an optional feature-flagged selector that works on top of the new embedding backend.

## Decisions Already Made

1. `Phikon-v2` will become the default and only Stage 7 embedding backend.
2. No backward compatibility is needed for the old CNN/U-Net embedding path.
3. Protected rows should bypass reduction entirely.
4. `m_max` should apply only to the reducible, non-protected pool.
5. GIST should remain available behind a feature flag and must work with the new `Phikon-v2` embeddings.

## Implementation Plan

### 1. Replace Stage 7 embeddings with `Phikon-v2`

- Remove the legacy CNN/U-Net embedding path from Stage 7.
- Standardize Stage 7 on Hugging Face `owkin/phikon-v2`.
- Use CLS-token embeddings from:

```python
outputs.last_hidden_state[:, 0, :]
```

- Keep the embedding output interface unchanged from the caller perspective:
  - return `float32` NumPy arrays

### 2. Simplify Stage 7 config around the new backend

- Remove old CNN-specific config fields from Stage 7:
  - `SMART_SAMPLER_ENCODER_NAME`
  - `SMART_SAMPLER_ENCODER_WEIGHTS`
- Keep relevant runtime settings:
  - `SMART_SAMPLER_DEVICE`
  - `SMART_SAMPLER_BATCH_SIZE`
  - `SMART_SAMPLER_NUM_WORKERS`
  - `SMART_SAMPLER_SEED`
- Add explicit model identifier for reproducibility:
  - `SMART_SAMPLER_MODEL_NAME=owkin/phikon-v2`

### 3. Refactor embedding extraction for Hugging Face

- Update Stage 7 embedding extraction to:
  - load raw patch images from HDF5
  - convert them into the format expected by `AutoImageProcessor`
  - run `AutoModel` on the configured device
  - extract CLS-token features
- Ensure model loading fails early with a clear error if the processor or model cannot be initialized.
- Preserve deterministic behavior as much as possible.

### 4. Update dependencies and runtime assumptions

- Confirm `transformers` and required model-loading dependencies are installed.
- Ensure the Stage 7 run config records the model identifier for auditability.
- Keep inference behavior explicit and reproducible.

### 5. Preserve the rest of Stage 7 behavior initially

Keep these parts unchanged during the first embedding migration:

- patient-wise HDF5 indexing
- stability-based candidate-pool search
- HDF5 writing
- sidecar artifact generation

This isolates the first scientific change to the embedding space only.

### 6. Implement protected-class retention

Extend Stage 7 selection to inspect:

- `labels`
- mask burden / positive-mask fraction

Split each patient’s rows into:

- protected rows
- reducible rows

Protected rows bypass reduction entirely.

### 7. Define protected retention policy explicitly

Add config for:

- protect positive-label patches
- protect mask-positive patches
- minimum positive-mask fraction threshold

Recommended default behavior:

- keep all positive-label patches
- keep all patches whose positive-mask fraction exceeds the configured threshold

### 8. Adopt the recommended `m_max` semantics

Use this rule:

- protected rows are always kept
- `m_max` limits only the reducible pool

This means patient output count may exceed `m_max` when protected rows are present.

### 9. Keep GIST as a feature flag and adapt it to `Phikon-v2`

- Preserve `SMART_SAMPLER_USE_GIST` as an optional selector flag.
- GIST should operate on the new `Phikon-v2` embeddings.
- Re-check runtime guardrails because the new embedding geometry may change candidate-pool behavior.
- Keep the current fallback behavior for overly large candidate pools if still needed.

### 10. Update per-patient selection flow

Per patient:

1. identify protected rows
2. define the reducible pool
3. build embeddings only for the reducible pool
4. run stability-based candidate-pool search on the reducible pool
5. run one final selector on the reducible candidate pool:
   - legacy selector, or
   - GIST selector when enabled
6. merge:
   - all protected rows
   - selected reducible rows

Patient-wise isolation must remain intact.

### 11. Update outputs for transparency

Extend output artifacts so they show:

- protected-kept rows
- legacy-sampled rows
- GIST-sampled rows

Update patient-level stats to report:

- protected count
- sampled reducible count
- rejected reducible count

Record in run config:

- `SMART_SAMPLER_MODEL_NAME`
- `SMART_SAMPLER_USE_GIST`

### 12. Update tests to the new scientific baseline

Replace old CNN-specific assumptions in tests.

Add tests for:

- `Phikon-v2` embedding extraction contract
- protected positive-label retention
- protected mask-threshold retention
- `m_max` applying only to the reducible pool
- GIST path working on `Phikon-v2` embeddings
- training-compatible filtered HDF5 output
- sidecar correctness and transparency fields

### 13. Update documentation

Update `.env_example` and Stage 7 documentation to reflect:

- `Phikon-v2` as the default and only embedding backend
- protected retention behavior
- `SMART_SAMPLER_USE_GIST` remaining optional
- `m_max` applying only to reducible rows

## Recommended Implementation Order

1. Replace embeddings with `Phikon-v2`
2. Remove obsolete CNN config
3. Update embedding tests
4. Add protected retention
5. Implement recommended `m_max` semantics
6. Ensure GIST still works as a flag on top of `Phikon-v2`
7. Update sidecars and run-config transparency
8. Update documentation

## Scientific Rationale

This plan addresses the most important scientific weaknesses first:

1. `Phikon-v2` improves the quality of the embedding space and makes distance-based sampling more pathology-relevant.
2. Protected retention reduces the risk of deleting rare, clinically important patches.
3. GIST remains available as an optional selector, but only after the embedding foundation is improved.

This sequence is more scientifically defensible than continuing to optimize selection behavior on top of weak generic embeddings.
