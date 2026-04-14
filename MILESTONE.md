# Migration Milestones

## --DONE Milestone 1: Freeze Stage 5 and Stage 6 Contracts

- Keep `5_crossfold.py` unchanged as the producer of singleton `TRAIN.h5`, `VALIDATION.h5`, and `TEST.h5`.
- Keep `6_sanity_checks.py` unchanged and validating only Stage 5 singleton outputs.
- Do not extend Stage 6 to validate shard outputs.

## --DONE Milestone 2: Implement Stage 6.5 Patient Sharding

- Add `6_5_patient_shards.py` between Stage 5 and Stage 7.
- Read Stage 5 singleton split files and generate patient-isolated shard directories.
- Create:
  - `TRAIN_shards/<patient_id>.h5`
  - `VALIDATION_shards/<patient_id>.h5`
  - `TEST_shards/<patient_id>.h5`
- Enforce the storage invariant that every shard contains exactly one patient and every patient is stored in exactly one shard per split.

## --DONE Milestone 3: Preserve HDF5 Contract and Provenance in Shards

- Ensure each patient shard keeps the canonical datasets:
  - `images`
  - `masks`
  - `labels`
  - `patient_ids`
  - `filenames`
- Preserve row alignment, patient membership, filename parity, and label semantics.
- Carry forward relevant provenance attributes from Stage 5 split outputs into each patient shard.

## --DONE Milestone 4: Emit Split-Level and Sample-Level Shard Manifests

- Write a manifest for each shard directory:
  - `TRAIN_shards/manifest.parquet`
  - `VALIDATION_shards/manifest.parquet`
  - `TEST_shards/manifest.parquet`
- Each split-level manifest row should include at least:
  - `patient_id`
  - `relative_hdf5_path`
  - `rows`
  - `label_0_count`
  - `label_1_count`
- Also write sample-level manifests for downstream random-access stages with at least:
  - `patient_id`
  - `relative_hdf5_path`
  - `row_in_shard`
  - `label`
  - `filename`

## --DONE Milestone 5: Validate Stage 6.5 Shard Integrity

- Add Stage 6.5-specific validation that confirms for each split:
  - total row count is preserved
  - patient set is preserved
  - per-patient row counts are preserved
  - class counts are preserved
  - image/mask/filename alignment is preserved
- Validate that every emitted shard contains exactly one unique patient ID.

## --DONE Milestone 6: Migrate Stage 7 Input to TRAIN Shards

- Replace Stage 7 singleton `TRAIN.h5` assumptions with shard-directory and manifest input.
- Keep the existing patient-wise smart-sampling behavior.
- Process one patient shard at a time from `TRAIN_shards`.

## --DONE Milestone 7: Migrate Stage 7 Output to Filtered Patient Shards

- Make Stage 7 emit `TRAIN_FILTERED_shards/<patient_id>.h5` instead of `TRAIN_FILTERED.h5`.
- Keep one-patient-per-shard isolation in filtered outputs.
- Generate a derived filtered manifest for `TRAIN_FILTERED_shards`.

## --DONE Milestone 8: Add Local SSD Patient-Shard Workspace for Stage 7

- Use Colab local SSD only as transient workspace.
- For each patient:
  - copy source shard from Drive to SSD
  - process locally
  - write filtered shard locally
  - publish filtered shard back to Drive
  - delete local temporary files after successful publish
- Prevent long-term accumulation of Stage 7 outputs on local SSD.

## --DONE Milestone 9: Add Stage 7 Resume Support via Output Shard Existence

- Before processing a patient, check whether `TRAIN_FILTERED_shards/<patient_id>.h5` already exists on Drive.
- If it exists, skip the patient.
- If it does not exist, process and publish it.
- Use temporary output paths plus atomic rename so incomplete uploads are never treated as completed shards.

## --DONE Milestone 10: Make Shard Files the Resume Source of Truth

- Do not use the filtered manifest as the live checkpoint state.
- Treat Drive shard existence as the authoritative completion signal.
- Rebuild the filtered manifest from output shards at the end of the run, or periodically in batches for convenience.

## --DONE Milestone 11: Add Shared Local Patient-Shard Cache for GPU Stages

- Implement a reusable Drive-to-SSD patient-shard cache for Stages 7 to 11.
- Cache policy must include:
  - local cache directory
  - size cap in bytes
  - LRU eviction
- Cache unit is exactly one patient shard.

## --DONE Milestone 12: Migrate Stage 11 to TEST Shards

- Replace singleton `TEST.h5` assumptions in Stage 11 with shard-directory and manifest input.
- Keep inference logic unchanged apart from the data-access layer.
- Use patient-shard loading and local cache reuse.

## --DONE Milestone 13: Migrate Stage 10 to VALIDATION Shards

- Replace singleton `VALIDATION.h5` assumptions in Stage 10 with shard-directory and manifest input.
- Keep optimization and evaluation logic unchanged apart from the data-access layer.
- Use patient-shard loading and local cache reuse.

## --DONE Milestone 14: Migrate Stage 8 to TRAIN and VALIDATION Shards

- Replace singleton training/validation HDF5 assumptions in Stage 8 with shard manifests and patient shards.
- Back dataset indexing, labels, and patient metadata from manifest data instead of monolithic HDF5 scanning.
- Preserve current LR finder sampling and reproducibility behavior.

## --DONE Milestone 15: Migrate Stage 9 to TRAIN_FILTERED and VALIDATION Shards

- Replace singleton `TRAIN.h5`, `TRAIN_FILTERED.h5`, and `VALIDATION.h5` assumptions in Stage 9 with shard manifests and patient shards.
- Keep model, optimizer, training loop, metrics, and checkpoint logic unchanged apart from the data-access layer.
- Back `__len__`, `__getitem__`, `get_labels()`, and `get_patient_ids()` from manifest-backed indexing.

## --DONE Milestone 16: Add Regression Tests for the Shard-Based Pipeline

- Add tests for Stage 6.5 shard generation.
- Add tests for Stage 7 shard input, shard output, and resume behavior.
- Add tests for shard-backed Stage 8, 9, 10, and 11 data loading.
- Verify preservation of patient isolation, row alignment, label semantics, and provenance across the migration.
