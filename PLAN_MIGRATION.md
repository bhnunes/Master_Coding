# Metadata-First Patient-Shard Migration Plan

## Goal

Move the pipeline from repeated downstream HDF5 rewrites to a metadata-first design with:

- one canonical Stage 2 patient-shard image store
- one master SQLite manifest for filtering, splitting, sampling, and lineage state
- on-the-fly stain normalization during model-facing stages

The main objective is to reduce disk duplication from Stages 3-11 while preserving:

- patient-level split isolation
- image/mask row alignment
- filename-keyed provenance
- scientific reproducibility
- current training/inference behavior
- the existing patient-shard contract across downstream stages

## Current State

Today the pipeline duplicates data multiple times:

1. Stage 2 writes the canonical HDF5 patient shards and patch-level artifact metadata.
2. Stage 3 rewrites accepted Stage 4.3 rows into a filtered source HDF5.
3. Stage 5 rewrites TRAIN/VALIDATION/TEST HDF5 files, optionally applying stain normalization.
4. Stage 6.5 / 7.1 rewrites patient-local shard HDF5s.
5. Stage 7.2 rewrites filtered TRAIN shard HDF5s.

Later stages already consume metadata heavily, but most still assume that metadata points to downstream-materialized HDF5 files instead of the original Stage 2 patient shards.

## Target Architecture

### Core Principle

Keep the Stage 2 patient shards as the only canonical image/mask store and move all downstream decisions into one centralized metadata store.

### Canonical Data Assets

1. Canonical Stage 2 patient shards
   - Stage 2 HDF5 shards remain the only mandatory pixel store for downstream stages.
   - Each shard contains `images`, `masks`, `labels`, `patient_ids`, `filenames`, and provenance fields.
   - Downstream manifests always resolve rows back to these original shards.

2. Master metadata store
   - A single `master_manifest.sqlite` file tracks one row per Stage 2 patch.
   - Rows are keyed by `source_hdf5_path` and `source_row_index`.
   - Downstream stages update columns in this database instead of creating separate row-manifest files.
   - Patch-level artifact coverage and related derived artifact fields live in this same database.

3. Normalization metadata
   - Stage 5 still fits normalization state using TRAIN only.
   - Instead of writing normalized images, it saves the normalization configuration, fitted parameters, reference templates, and lineage needed for on-the-fly execution.

### Metadata-Driven Stages

- Stage 4.3 updates accepted/rejected row decisions.
- Stage 5 updates split assignment and normalization metadata.
- Stage 7.2 updates selected/rejected sampling decisions.
- Stages 7.2-11 load canonical images by resolving rows from the master SQLite manifest back to the Stage 2 patient shards.

### Artifact Metadata Policy

- Replace the separate artifact coverage parquet with SQLite-backed patch metadata.
- Store compact per-patch derived artifact values in `master_manifest.sqlite`.
- Do not store heavy raw geometry blobs or full GeoJSON payloads in the hot patch tables.
- Keep raw artifact source files external and reference them only through provenance fields when needed.

## Non-Negotiable Invariants

The migration must preserve these contracts throughout:

1. Patient isolation
   - No patient may cross TRAIN/VALIDATION/TEST boundaries.

2. Row alignment
   - Image, mask, label, patient id, and filename must remain aligned for every source row.

3. Source identity
   - Every downstream sample must resolve back to a canonical Stage 2 `(source_hdf5_path, source_row_index)`.

4. Reproducibility
   - The master metadata store plus stage sidecars must carry sufficient lineage to reconstruct how each row state was produced.

5. Scientific semantics
   - Label meaning, contamination filtering semantics, smart sampling behavior, and normalization fitting rules must not change silently.

6. Native-only migration
   - No backward-compatibility materialization path is required.
   - Obsolete rewrite stages and code paths should be removed rather than preserved behind flags.

## Important Design Decision

Do not use one mutable parquet file as a database.

Use one mutable SQLite database as the master manifest.

Reason:

- SQLite supports repeated indexed updates cleanly
- SQLite fits the one-file manifest requirement
- SQLite avoids full-file parquet rewrites on every stage update
- SQLite is well-suited for stage orchestration queries and validation

Critical runtime rule:

- SQLite must not sit in the hot sample-loading path
- Stages 7.2-11 should query SQLite once at startup, materialize the needed row records in memory, and then perform minibatch loading from in-memory metadata plus HDF5 only

Explicit scalability rule:

- `master_manifest.sqlite` is orchestration metadata only, never a per-sample runtime dependency
- millions of rows are acceptable as long as runtime stages use indexed startup queries plus in-memory caching
- batch inserts and updates must run in transactions
- the database should hold compact patch metadata only, not pixels, masks, raw GeoJSON, or other heavy blobs

Recommended file set:

- `master_manifest.sqlite`
- stage-specific run-config and lineage JSON sidecars
- optional immutable exports for audits or debugging only

The concrete Milestone 1 contract for schema, ownership, row identity, and runtime boundaries lives in `MASTER_MANIFEST_CONTRACT.md`.

## Recommended End State by Stage

### Stage 2

Keep Stage 2 largely unchanged.

Responsibilities:

- extract canonical patches
- persist canonical patient-shard HDF5 files
- persist patch-level artifact metadata in `master_manifest.sqlite`
- persist source provenance
- create and own `master_manifest.sqlite`

Required enhancement:

- ensure all downstream-required stable columns are available in the master manifest, including:
  - `source_hdf5_path`
  - `source_row_index`
  - `filename`
  - `patient_id`
  - `label`
  - optional `slide_id`

Recommended Stage 2 ownership:

- Stage 2 should insert the initial patch rows into `master_manifest.sqlite`
- Stage 2 should attach stable artifact coverage fields to those same patch rows when artifact data is available
- Stage 2 should be the only stage that creates new patch identity rows
- downstream stages should only update status and lineage columns for existing rows

### Stage 3

Delete Stage 3.

Reason:

- it only exists to rewrite data that will now stay in the Stage 2 canonical patient shards
- the accepted row set can be represented directly as a manifest join over Stage 2 row identity
- keeping the script or helper path adds maintenance cost without providing value in the native architecture

Replacement artifacts:

- accepted-row status columns in `master_manifest.sqlite`
- source dataset provenance JSON

### Stage 4.3

Current role already fits the target architecture well.

Target adjustments:

- keep accepted/rejected decisions as the source of truth
- add stable schema and hashes for downstream joins
- write decisions into `master_manifest.sqlite`

Required row keys:

- `source_hdf5_path`
- `source_row_index`
- `filename`
- `patient_id`
- `cleaning_decision`
- `contamination_rate`

### Stage 5

Current role:

- compute split assignment
- fit TRAIN-derived stain normalization metadata
- write TRAIN/VALIDATION/TEST HDF5 files

Target role:

- compute split assignment exactly as today
- fit normalization state on TRAIN only exactly as today
- update the master manifest with split and normalization metadata
- stop writing split HDF5 files

New outputs:

- `normalization_state.json`
- `normalization_templates/`
- run config JSON with lineage and hashes

Master-manifest columns updated by Stage 5:

- `split`
- `source_hdf5_path`
- `source_row_index`
- `filename`
- `patient_id`
- `label`
- `normalization_method`
- `normalization_metadata_path`

### Stage 6.5

Delete Stage 6.5 / `7_1_patient_shards.py`.

Reason:

- Stage 2 patient shards are already the canonical storage contract
- a second patient-sharding stage is redundant in the native architecture
- downstream stages should consume manifests plus canonical Stage 2 shards directly

### Stage 7.2

Current role:

- materialize filtered TRAIN shard HDF5s

Target role:

- update sample-selection columns only
- define final training sample set by metadata
- load Stage 2 patient-shard rows on demand when it needs pixel access
- support on-the-fly stain normalization before any augmentation whenever normalization is enabled

New outputs:

- `summary.json`
- patient reduction stats CSV/JSON as today
- run config JSON with lineage and hashes

### Stages 8-11

Target role:

- load images and masks from canonical Stage 2 patient shards using rows selected from SQLite
- apply ImageNet normalization on GPU as today
- apply stain normalization on the fly before augmentations and before ImageNet normalization

This is the major implementation step because current loaders still assume data has already been materialized into downstream split or filtered shard HDF5 files.

## Loader Refactor Strategy

### New Shared Abstraction

Introduce a shared SQLite-backed sample dataset abstraction under `helpers/training/` and reuse it across:

- Stage 7.2 smart sampler
- Stage 8 LR finder
- Stage 9 training
- Stage 10 optimizer
- Stage 11 inference

Suggested responsibilities:

1. Load row records from `master_manifest.sqlite` with one stage-specific query at startup.
2. Resolve each sample to canonical Stage 2 `source_hdf5_path` and `source_row_index`.
3. Open canonical Stage 2 HDF5 patient shards lazily and reuse handles per worker.
4. Read `images[row]` and `masks[row]` on demand.
5. Apply optional on-the-fly stain normalization before any augmentation.
6. Apply existing Albumentations transforms.
7. Preserve filename/patient/label accessors for current samplers and validators.

### Hot-Path Constraint

For Stages 7.2-11:

1. Query SQLite once during stage setup.
2. Materialize the selected rows into compact in-memory records.
3. Never hit SQLite inside `Dataset.__getitem__` or batch collation.
4. Restrict the runtime hot path to in-memory metadata, HDF5 reads, normalization, and transforms.
5. Treat SQLite as a startup filter and provenance store, not as an online sample-serving layer.

### Why this is the key change

Once the loaders can dereference canonical Stage 2 rows directly, downstream HDF5 duplication is no longer required.

## On-the-Fly Normalization Strategy

### Supported methods

The on-the-fly path should support:

- `none`
- `reinhard`
- `ruifrok`
- `macenko`
- `vahadane`

### Implementation direction

Use `torch-staintools` as the primary normalization backend for GPU-capable stain normalization.

Design rules:

1. Apply stain normalization on the fly while loading patches from HDF5.
2. Run stain normalization before Albumentations or other data augmentation.
3. Keep ImageNet normalization as a later model-facing step.
4. Prefer GPU execution and batched tensor operations wherever practical.
5. Avoid TIAToolbox dependencies in the migrated path.

### Method-specific notes

1. `reinhard`
   - use the torch-staintools implementation with TRAIN-fitted target statistics

2. `macenko`
   - use the torch-staintools implementation with TRAIN-derived references

3. `vahadane`
   - use the torch-staintools implementation with TRAIN-derived references

4. `ruifrok`
   - implement as fixed-matrix stain deconvolution/reconvolution on GPU
   - do not require a separate library-level class
   - treat it as Macenko/Vahadane-style normalization with a fixed stain matrix instead of stain-matrix estimation

5. `none`
   - bypass the normalization branch with zero extra work

### Fastest-theoretical execution model

Because speed is the priority and this migration is native-only:

1. Fit any required stain references once from TRAIN rows in Stage 5.
2. Persist only compact normalization state and templates.
3. Query SQLite once and cache row metadata in memory.
4. Load patches from canonical Stage 2 shards on demand.
5. Convert to tensors once.
6. Apply stain normalization on GPU before augmentation.
7. Avoid any intermediate normalized HDF5 or PNG materialization.

## Migration Phases

### Phase 1: Stabilize canonical row identity

Deliverables:

- master SQLite schema
- stable row keys across stages
- tests covering row lineage and filename parity

Tasks:

1. Define one master SQLite schema shared across stages.
2. Ensure Stage 2 creates and populates it.
3. Ensure Stage 4.3 updates join cleanly onto it.

Exit criteria:

- every downstream sample can be addressed by canonical source row identity

### Phase 2: Stop Stage 3 image rewrites in native mode

Deliverables:

- manifest-driven accepted row selection
- Stage 3 removal

Tasks:

1. Replace Stage 3 filtered HDF5 as the source of truth with SQLite row-state updates.
2. Delete `3_pack_splits_to_hdf5.py` and obsolete packaging-only migration code.

Exit criteria:

- Stage 5 can run from manifests without requiring any Stage 3 rewritten HDF5

### Phase 3: Stop Stage 5 split HDF5 rewrites

Deliverables:

- split columns in `master_manifest.sqlite`
- normalization metadata only

Tasks:

1. Keep current splitting logic.
2. Keep current normalizer fitting logic.
3. Replace split HDF5 writing with SQLite updates only.
4. Preserve current manifest/statistics outputs.

Exit criteria:

- split definition exists without materialized TRAIN/VALIDATION/TEST HDF5 files

### Phase 4: Build shared SQLite-backed dataset loader

Deliverables:

- shared canonical-row dataset class
- stage-specific SQLite query helpers
- worker-safe HDF5 handle reuse
- optional on-the-fly stain normalization hook

Tasks:

1. Refactor Stage 7.2, Stage 8, and Stage 9 loaders first.
2. Migrate Stage 10 and Stage 11 to the same abstraction.
3. Preserve helper methods used for patient checks and weighting.
4. Make stain normalization run before augmentation in every consumer.

Exit criteria:

- Stages 8-11 can run without split-specific HDF5 files

### Phase 5: Convert Stage 7.2 to selection-manifest output

Deliverables:

- selection columns in `master_manifest.sqlite` replace filtered TRAIN shard HDF5

Tasks:

1. Emit selected rows only.
2. Teach training loaders to query selected rows from SQLite directly at startup.
3. Remove filtered shard materialization code.

Exit criteria:

- smart sampling no longer rewrites images in native mode

### Phase 6: Remove Stage 6.5 / 7.1 patient resharing

Deliverables:

- Stage 6.5 / 7.1 removed from the active pipeline
- documentation and configs updated to point directly at Stage 2 patient shards

Exit criteria:

- pipeline no longer depends on any post-Stage-2 patient-shard rewrite step

### Phase 7: Finish native on-the-fly normalization rollout

Deliverables:

- shared normalization abstraction used by Stages 7.2-11
- TIAToolbox removed from the migrated training/inference path
- Ruifrok fixed-matrix GPU path implemented

Exit criteria:

- no scientific regression

## Master Manifest Schema

Use one `master_manifest.sqlite` file with at least the following conceptual tables.

`MASTER_MANIFEST_CONTRACT.md` is the authoritative contract for the required tables, minimum columns, ownership rules, and runtime constraints. The list below remains the architectural summary.

1. `patches`
   - one immutable row per Stage 2 patch
   - stable identity, static metadata, source provenance, and HDF5 location

2. `patch_stage_state`
   - mutable per-row stage decisions such as cleaning, split, sampling, normalization assignment, and eligibility flags

3. `runs`
   - one row per stage execution with config snapshot, timestamps, code/version identifiers, and input/output hashes

4. `normalization_artifacts`
   - normalization method, fitted state paths, template paths, and reference metadata

Artifact metadata placement recommendation:

- keep one-to-one stable artifact-derived values directly on `patches`
- use a separate table only if artifact metadata becomes sparse, multi-valued, or evolves independently from the core patch rows

Good candidates for direct `patches` columns:

- `artifact_coverage_fraction`
- `artifact_overlap_flag`
- `artifact_exclusion_flag`
- artifact-processing signature or source reference fields used for provenance

Minimum indexes:

- unique `(source_hdf5_path, source_row_index)`
- `patient_id`
- `filename`
- `cleaning_decision`
- `split`
- `sampling_decision`
- composite indexes for common stage filters

## Provenance Requirements

Every stage updating the master manifest should record:

- input source paths
- input hashes
- canonical Stage 2 source HDF5 hash or source signature
- artifact source/signature hashes when artifact-derived patch metadata is present
- normalization method and stats hash when applicable
- config snapshot
- row counts by split and class

Where possible, prefer immutable JSON sidecars plus SHA-256 hashes over inferred lineage.

## Test Plan

Each behavior change should ship with tests.

### Required tests

1. Manifest join integrity
   - Stage 2 artifact metadata and Stage 4.3/5 updates target the correct Stage 2 rows in SQLite

2. Split integrity
   - TRAIN/VALIDATION/TEST patient isolation is unchanged in SQLite-backed mode

3. Sample identity preservation
   - manifest-backed loader returns the same image, mask, filename, label, and patient id as the canonical Stage 2 shard for the same source rows

4. Smart sampling equivalence
   - selected row sets match current behavior

5. Normalization equivalence
   - on-the-fly normalization output matches saved reference behavior within acceptable tolerance for `reinhard`, `ruifrok`, `macenko`, and `vahadane`

6. Provenance fail-closed behavior
   - stale or mismatched SQLite lineage metadata raises clear errors

7. Artifact metadata equivalence
   - SQLite-backed artifact coverage fields match the previous patch-level artifact derivations used by downstream consumers

### Recommended regression tests

1. multiple canonical source HDF5 files in one SQLite query result
2. empty split behavior
3. filename dataset decoding stability
4. worker process reopening of HDF5 handles
5. startup-only SQLite access for runtime datasets

## Main Risks

### Risk 1: Training slowdown from on-the-fly normalization

Mitigation:

- prefer GPU execution
- keep normalization state compact
- avoid extra materialization passes

### Risk 2: Complex lineage across many manifest layers

Mitigation:

- keep mutable row state in one SQLite file
- keep lineage/config in immutable JSON sidecars
- hash every input artifact
- validate lineage explicitly

### Risk 3: SQLite becoming a runtime bottleneck

Mitigation:

- use SQLite only during stage setup
- load selected rows into memory before DataLoader iteration starts
- never query SQLite inside `__getitem__`
- add indexes for stage filter columns
- keep all writes batched and transactional

### Risk 4: Random-access HDF5 reads becoming slower than current shard-local layout

Mitigation:

- preserve the patient-shard layout already used in Stage 2
- reuse HDF5 handles per worker
- keep local staging and shard cache patterns where useful

### Risk 5: Silent scientific drift

Mitigation:

- add regression tests before removing old paths
- validate every normalization method against fixed references

## Recommendation Summary

The migration is worth doing, and the native target should be implemented in this order:

1. Make filtering, splitting, and sampling SQLite-native.
2. Delete Stages 3 and 6.5 / 7.1 rather than keeping compatibility wrappers.
3. Build one shared SQLite-backed startup query layer plus in-memory loader for Stages 7.2-11.
4. Move stain normalization to on-the-fly GPU execution backed by `torch-staintools`.
5. Implement `ruifrok` as a fixed stain-matrix GPU path, not as a special materialized workflow.

## First Concrete Implementation Slice

The safest first slice is:

1. Define the master SQLite schema.
2. Make Stage 2 create and populate `master_manifest.sqlite`.
3. Remove Stage 3 as a source of truth and replace it with SQLite row-state updates.
4. Remove Stage 6.5 / 7.1 from the migration target entirely.
5. Make Stage 5 update split columns plus normalization state only.
6. Build one SQLite-backed loader prototype for Stage 8 or Stage 9 with on-the-fly normalization.

This establishes the permanent native architecture directly instead of spending effort on transitional compatibility paths.
