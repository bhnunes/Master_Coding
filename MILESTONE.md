# Migration Milestones

Use `--TBD` for pending milestones and replace it with `--DONE` once the milestone is implemented and validated.

## Milestone 1 --TBD

Define the master SQLite migration contract.

Scope:
- finalize `master_manifest.sqlite` as the single metadata source
- lock the canonical row identity around `(source_hdf5_path, source_row_index)`
- lock the rule that Stage 2 patient shards remain the only pixel store
- lock the rule that SQLite is orchestration metadata only, never a per-sample runtime dependency

Exit criteria:
- the schema and ownership rules are documented and agreed
- downstream stages have a stable row identity contract

## Milestone 2 --TBD

Implement Stage 2 master manifest creation.

Scope:
- make `2_database_manager.py` create `master_manifest.sqlite`
- insert one immutable patch row per extracted Stage 2 patch
- persist stable patch metadata needed downstream
- persist provenance fields needed to trace each patch back to source inputs

Exit criteria:
- Stage 2 creates and populates the SQLite master manifest
- every extracted patch can be resolved by canonical source identity

## Milestone 3 --TBD

Move artifact patch metadata into SQLite.

Scope:
- replace the separate artifact parquet with SQLite-backed patch metadata
- store compact one-to-one artifact-derived values on patch rows
- keep heavy raw geometry and GeoJSON outside the hot metadata tables
- preserve artifact provenance and signatures

Exit criteria:
- artifact coverage fields live in `master_manifest.sqlite`
- downstream consumers no longer require artifact parquet

## Milestone 4 --TBD

Delete Stage 3 from the active architecture.

Scope:
- replace Stage 3 filtered HDF5 source-of-truth behavior with SQLite row-state updates
- remove `3_pack_splits_to_hdf5.py`
- remove obsolete Stage 3 packaging-only migration code that exists only for rewrite workflows

Exit criteria:
- no downstream stage depends on Stage 3 rewritten HDF5 outputs
- accepted-row state is represented in SQLite instead

## Milestone 5 --TBD

Migrate Stage 4.3 cleaning decisions to SQLite.

Scope:
- write cleaning decision fields into `master_manifest.sqlite`
- preserve contamination-rate metadata and stable row joins
- keep accepted/rejected logic scientifically identical

Exit criteria:
- Stage 4.3 updates the correct Stage 2 rows in SQLite
- cleaning state is queryable without any separate row manifest

## Milestone 6 --TBD

Stop Stage 5 split HDF5 rewrites.

Scope:
- keep current split logic unchanged
- keep TRAIN-only normalization fitting unchanged
- write split assignment and normalization metadata into SQLite
- emit normalization sidecars without materializing split HDF5 files

Exit criteria:
- split definition exists in SQLite only
- Stage 5 no longer writes TRAIN/VALIDATION/TEST HDF5 outputs

## Milestone 7 --TBD

Remove Stage 6.5 / `7_1_patient_shards.py` from the pipeline.

Scope:
- delete the redundant post-Stage-2 patient resharing step
- remove dependencies on Stage 6.5 outputs
- update docs and configs to point directly to Stage 2 patient shards

Exit criteria:
- no active workflow depends on Stage 6.5 / 7.1 outputs
- Stage 2 patient shards are the only shard contract in the pipeline

## Milestone 8 --TBD

Convert Stage 7.2 smart sampling to SQLite row-state updates.

Scope:
- replace filtered TRAIN shard materialization with sampling decision columns in SQLite
- keep patient-wise label-aware sampling behavior unchanged
- emit only summary and lineage sidecars

Exit criteria:
- Stage 7.2 no longer rewrites filtered TRAIN shard HDF5 files
- selected training rows are resolved directly from SQLite

## Milestone 9 --TBD

Build the shared SQLite-backed loader foundation.

Scope:
- create shared stage-specific SQLite query helpers
- create the shared canonical-row dataset abstraction under `helpers/training/`
- support worker-safe HDF5 handle reuse
- enforce startup-only SQLite access with in-memory row caching

Exit criteria:
- runtime loaders query SQLite once at startup only
- runtime hot path uses only in-memory metadata plus HDF5 reads

## Milestone 10 --TBD

Implement shared on-the-fly stain normalization.

Scope:
- use `torch-staintools` as the main backend
- support `none`, `reinhard`, `ruifrok`, `macenko`, and `vahadane`
- apply normalization before augmentations and before ImageNet normalization
- keep Stage 5 as the owner of TRAIN-fitted normalization state

Exit criteria:
- a shared normalization abstraction is available to all model-facing stages
- no migrated path depends on TIAToolbox

## Milestone 11 --TBD

Implement the GPU Ruifrok path.

Scope:
- implement `ruifrok` as fixed-matrix stain deconvolution/reconvolution on GPU
- keep it aligned with the Macenko/Vahadane-style formulation using a fixed matrix
- integrate it into the shared normalization abstraction

Exit criteria:
- `ruifrok` is available as an on-the-fly normalization option
- no dedicated external Ruifrok class is required

## Milestone 12 --TBD

Migrate Stage 8 LR finder to the new loader path.

Scope:
- make Stage 8 resolve rows from SQLite at startup
- make Stage 8 load pixels from Stage 2 patient shards
- make Stage 8 use shared on-the-fly normalization

Exit criteria:
- Stage 8 runs without split-specific HDF5 files

## Milestone 13 --TBD

Migrate Stage 9 training to the new loader path.

Scope:
- make Stage 9 resolve rows from SQLite at startup
- preserve current patient checks, weighting logic, and training semantics
- make Stage 9 use shared on-the-fly normalization and HDF5-backed reads

Exit criteria:
- Stage 9 runs without split-specific or filtered shard HDF5 files

## Milestone 14 --TBD

Migrate Stage 10 optimizer to the new loader path.

Scope:
- make Stage 10 resolve rows from SQLite at startup
- make Stage 10 use shared HDF5-backed reads and on-the-fly normalization

Exit criteria:
- Stage 10 runs without downstream-materialized HDF5 dependencies

## Milestone 15 --TBD

Migrate Stage 11 inference to the new loader path.

Scope:
- make Stage 11 resolve rows from SQLite at startup
- make Stage 11 use shared HDF5-backed reads and on-the-fly normalization

Exit criteria:
- Stage 11 runs without downstream-materialized HDF5 dependencies

## Milestone 16 --TBD

Validate scientific and metadata equivalence.

Scope:
- verify patient isolation is unchanged
- verify sample identity preservation against Stage 2 shards
- verify artifact metadata equivalence
- verify smart sampling equivalence
- verify normalization equivalence for all supported methods
- verify provenance fail-closed behavior

Exit criteria:
- required regression coverage is in place
- no scientific regression is detected

## Milestone 17 --TBD

Remove obsolete dependencies and cleanup leftover rewrite paths.

Scope:
- remove TIAToolbox from migrated paths
- remove dead code tied to Stage 3 and Stage 6.5 / 7.1 rewrite workflows
- remove leftover downstream-materialization assumptions from configs and docs

Exit criteria:
- the active pipeline matches `PLAN_MIGRATION.md`
- obsolete rewrite-only code paths are gone
