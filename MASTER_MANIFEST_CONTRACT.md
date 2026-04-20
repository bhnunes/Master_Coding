# Master Manifest Contract

This document locks the Milestone 1 contract for the metadata-first migration.

## Purpose

`master_manifest.sqlite` is the single metadata source for downstream orchestration.

It exists to answer startup-time stage questions such as:

- which Stage 2 patch rows exist
- which rows are accepted by Stage 4.3 cleaning
- which rows belong to each Stage 5 split
- which rows were selected by Stage 7.2 sampling
- which normalization artifact a stage run should use

It does not exist to serve pixels at runtime.

## Non-Negotiable Rules

1. Stage 2 patient shards are the only pixel store.
2. `master_manifest.sqlite` is the only mutable metadata database shared across stages.
3. Canonical row identity is exactly `(source_hdf5_path, source_row_index)`.
4. Stage 2 is the only stage allowed to create patch identity rows.
5. Downstream stages may update row state and lineage only for existing Stage 2 rows.
6. Stages 7.2-11 may query SQLite at startup only; `Dataset.__getitem__` and batch collation must use in-memory metadata plus HDF5 reads only.
7. SQLite stores compact orchestration metadata only. It must never store pixels, masks, raw GeoJSON, or other heavy blobs.

## Ownership

### Stage 2 ownership

Stage 2 owns:

- creating `master_manifest.sqlite`
- creating the core schema
- inserting one immutable patch row per extracted Stage 2 patch
- storing stable patch metadata and provenance needed by later stages

Stage 2 is the only stage allowed to create new canonical patch rows.

### Downstream ownership

Stage 4.3, Stage 5, and Stage 7.2+ may:

- insert stage-run lineage rows
- update mutable stage-state columns for existing patch identities
- validate provenance before applying updates

They must not:

- create new patch identities
- rewrite canonical pixel location fields
- treat SQLite as a minibatch-serving dependency

## Canonical Row Identity

Every downstream sample must resolve to one Stage 2 patch row by:

- `source_hdf5_path`: canonical path string for the Stage 2 patient shard containing the row
- `source_row_index`: zero-based row index inside that canonical Stage 2 HDF5 shard

Identity contract:

- the pair `(source_hdf5_path, source_row_index)` is globally unique in the database
- `filename`, `patient_id`, and `label` are required stable attributes of that identity row
- if a stage emits sidecars, they must reference rows by canonical identity or by a database primary key derived from that identity row
- downstream joins must never depend on mutable split-specific or rewrite-specific HDF5 paths

## Runtime Contract

Stages 7.2-11 must follow this execution model:

1. Query SQLite once during startup.
2. Materialize selected rows into in-memory records.
3. Open Stage 2 HDF5 files lazily and reuse handles per worker.
4. Read `images[row]` and `masks[row]` directly from the canonical Stage 2 patient shards.
5. Apply on-the-fly normalization and transforms without returning to SQLite.

The runtime hot path is therefore:

- in-memory row metadata
- HDF5 reads
- normalization
- transforms

and never SQLite lookups.

## Required Tables

`master_manifest.sqlite` must contain at least these logical tables.

### `patches`

One immutable row per Stage 2 patch.

Required columns:

- `patch_id`: stable database primary key
- `source_hdf5_path`: text, not null
- `source_row_index`: integer, not null
- `filename`: text, not null
- `patient_id`: integer, not null
- `label`: integer, not null
- `slide_id`: text, nullable
- `source_signature`: text, nullable
- `artifact_coverage_fraction`: real, nullable
- `artifact_overlap_flag`: integer, nullable
- `artifact_exclusion_flag`: integer, nullable
- artifact provenance reference fields needed to validate the artifact source used by Stage 2

Required constraint:

- unique `("source_hdf5_path", "source_row_index")`

### `patch_stage_state`

One mutable row per `patch_id` holding the latest accepted stage-state fields.

Required columns:

- `patch_id`: primary/foreign key to `patches.patch_id`
- `cleaning_decision`: text, nullable
- `contamination_rate`: real, nullable
- `split`: text, nullable
- `normalization_method`: text, nullable
- `normalization_artifact_id`: integer, nullable
- `sampling_decision`: text, nullable
- `is_stage4_accepted`: integer, nullable
- `is_stage7_selected`: integer, nullable
- stage-specific lineage fields for the last run that updated the row

### `runs`

One row per stage execution.

Required columns:

- `run_id`: primary key
- `stage_name`: text, not null
- `config_path`: text, nullable
- `config_sha256`: text, nullable
- `started_at`: text, not null
- `completed_at`: text, nullable
- `code_version`: text, nullable
- `input_summary_json_path`: text, nullable
- `input_summary_sha256`: text, nullable

### `normalization_artifacts`

Normalization state owned by Stage 5 and referenced by runtime stages.

Required columns:

- `normalization_artifact_id`: primary key
- `run_id`: foreign key to `runs.run_id`
- `method`: text, not null
- `state_path`: text, not null
- `state_sha256`: text, not null
- `template_path`: text, nullable
- `template_sha256`: text, nullable
- `fit_scope`: text, not null

## Required Indexes

Minimum indexes:

- unique index on `(source_hdf5_path, source_row_index)`
- index on `patches.patient_id`
- index on `patches.filename`
- index on `patch_stage_state.cleaning_decision`
- index on `patch_stage_state.split`
- index on `patch_stage_state.sampling_decision`
- composite indexes added for common stage filters once the first runtime query shapes are implemented

## Update Rules By Stage

### Stage 2

- inserts into `patches`
- inserts the initial `patch_stage_state` rows
- inserts a `runs` row for the extraction/database build
- may insert artifact-derived patch metadata when available

### Stage 4.3

- updates `patch_stage_state` by joining on canonical row identity
- records cleaning decisions and contamination metadata
- inserts a `runs` row for the cleaning execution

### Stage 5

- updates `patch_stage_state` with split assignment and normalization references
- inserts normalization artifacts
- inserts a `runs` row for the split/normalization execution

### Stage 7.2

- updates `patch_stage_state` with sampling decisions only
- inserts a `runs` row for the sampling execution

### Stages 8-11

- read startup metadata from SQLite
- must not create or mutate patch identity rows during minibatch execution

## Out Of Scope For SQLite

The following must stay outside `master_manifest.sqlite`:

- image tensors and mask arrays
- raw HDF5 patch payloads
- raw artifact geometry
- full GeoJSON payloads
- derived files that are cheaper and safer to keep as immutable sidecars

## Exit Condition For Milestone 1

Milestone 1 is complete when later implementation work can rely on this document as the source of truth for:

- the single metadata database choice
- canonical row identity
- Stage 2 ownership of patch identity creation
- Stage 2 patient shards as the only pixel store
- startup-only SQLite use for runtime stages
