# Migration Milestones

Use `--TBD` for pending milestones and replace it with `--DONE` once the milestone is implemented and validated.

## Milestone 1 --DONE

Define the master SQLite migration contract.

Scope:
- finalize `master_manifest.sqlite` as the single metadata source
- lock the canonical row identity around `(source_hdf5_path, source_row_index)`
- lock the rule that Stage 2 patient shards remain the only pixel store
- lock the rule that SQLite is orchestration metadata only, never a per-sample runtime dependency

Exit criteria:
- the schema and ownership rules are documented and agreed
- downstream stages have a stable row identity contract

Implemented by:
- `MASTER_MANIFEST_CONTRACT.md` now defines the authoritative schema, ownership, canonical row identity, and runtime hot-path rules
- `PLAN_MIGRATION.md` now points to that contract as the Milestone 1 source of truth

## Milestone 2 --DONE

Implement Stage 2 master manifest creation.

Scope:
- make `2_database_manager.py` create `master_manifest.sqlite`
- insert one immutable patch row per extracted Stage 2 patch
- persist stable patch metadata needed downstream
- persist provenance fields needed to trace each patch back to source inputs

Exit criteria:
- Stage 2 creates and populates the SQLite master manifest
- every extracted patch can be resolved by canonical source identity

Implemented by:
- `helpers/extraction/master_manifest.py` now creates the SQLite schema and replaces one slide's Stage 2 rows using canonical `(source_hdf5_path, source_row_index)` identity
- `2_database_manager.py` now initializes `master_manifest.sqlite` and populates it during Stage 2 processing from the same patch records used to write the Stage 2 HDF5 shard
- `helpers/extraction/config.py` now exposes the canonical Stage 2 `master_manifest.sqlite` path under the project base directory
- `tests/test_extraction_master_manifest.py` adds regression coverage for row creation, provenance fields, and slide-row replacement behavior

## Milestone 3 --DONE

Move artifact patch metadata into SQLite.

Scope:
- replace the separate artifact parquet with SQLite-backed patch metadata
- store compact one-to-one artifact-derived values on patch rows
- keep heavy raw geometry and GeoJSON outside the hot metadata tables
- preserve artifact provenance and signatures

Exit criteria:
- artifact coverage fields live in `master_manifest.sqlite`
- downstream consumers no longer require artifact parquet

Implemented by:
- `2_database_manager.py` no longer writes the separate `artifact_patch_index.parquet` file in the active Stage 2 processing path
- `helpers/training/data.py` now loads artifact-aware loss coverage values from `master_manifest.sqlite`
- training config, provenance, and reporting now reference `master_manifest_path` instead of an artifact parquet path
- training-side regression tests now validate SQLite-backed artifact lookup and `master_manifest_path` provenance wiring

## Milestone 4 --DONE

Delete Stage 3 from the active architecture.

Scope:
- replace Stage 3 filtered HDF5 source-of-truth behavior with SQLite row-state updates
- remove `3_pack_splits_to_hdf5.py`
- remove obsolete Stage 3 packaging-only migration code that exists only for rewrite workflows

Exit criteria:
- no downstream stage depends on Stage 3 rewritten HDF5 outputs
- accepted-row state is represented in SQLite instead

Implemented by:
- Stage 4.3 now updates accepted/rejected cleaning state directly in `master_manifest.sqlite`
- Stage 5 now accepts a SQLite-backed source dataset and loads accepted rows directly from `master_manifest.sqlite`
- Stage 5 split writing now reads rows directly from their canonical Stage 2 `source_hdf5_path` and `source_row_index`, including multi-shard inputs
- `3_pack_splits_to_hdf5.py` is now an obsolete entrypoint that exits with a migration message instead of acting as an active pipeline stage

## Milestone 5 --DONE

Migrate Stage 4.3 cleaning decisions to SQLite.

Scope:
- write cleaning decision fields into `master_manifest.sqlite`
- preserve contamination-rate metadata and stable row joins
- keep accepted/rejected logic scientifically identical

Exit criteria:
- Stage 4.3 updates the correct Stage 2 rows in SQLite
- cleaning state is queryable without any separate row manifest

Implemented by:
- `helpers/extraction/master_manifest.py` now validates Stage 4.3 decision provenance against the canonical Stage 2 row before updating SQLite, including `filename`, `patient_id`, `slide_id`, and `source_signature`
- Stage 4.3 continues to persist `cleaning_decision`, `contamination_rate`, and `is_stage4_accepted` in `patch_stage_state`, but now fails closed if a stale or mismatched cleaning record targets the wrong canonical row
- `tests/test_graph_cleaning_pipeline.py` now covers successful SQLite updates plus regression cases for stale `source_signature` lineage and mismatched filename joins

## Milestone 6 --DONE

Stop Stage 5 split HDF5 rewrites.

Scope:
- keep current split logic unchanged
- keep TRAIN-only normalization fitting unchanged
- write split assignment and normalization metadata into SQLite
- emit normalization sidecars without materializing split HDF5 files

Exit criteria:
- split definition exists in SQLite only
- Stage 5 no longer writes TRAIN/VALIDATION/TEST HDF5 outputs

Implemented by:
- `helpers/crossfold/pipeline.py` no longer writes `TRAIN.h5`, `VALIDATION.h5`, or `TEST.h5`; Stage 5 now requires `master_manifest.sqlite` as its native source, preserves split selection and normalization fitting, writes sidecar artifacts, and then persists Stage 5 state into SQLite
- `helpers/extraction/master_manifest.py` now supports Stage 5 run recording, normalization artifact recording, and split-assignment updates in `patch_stage_state`, with canonical-row provenance validation on `filename`, `patient_id`, and `label`
- `helpers/crossfold/provenance.py` now writes `manifest.csv` as a canonical-row split manifest keyed by `source_hdf5_path` and `source_row_index` instead of split-HDF5 row coordinates
- regression tests now cover SQLite-backed Stage 5 split persistence, normalization artifact persistence, and the canonical-row Stage 5 manifest shape

## Milestone 7 --DONE

Remove Stage 6.5 / `7_1_patient_shards.py` from the pipeline.

Scope:
- delete the redundant post-Stage-2 patient resharing step
- remove dependencies on Stage 6.5 outputs
- update docs and configs to point directly to Stage 2 patient shards

Exit criteria:
- no active workflow depends on Stage 6.5 / 7.1 outputs
- Stage 2 patient shards are the only shard contract in the pipeline

Implemented by:
- `7_1_patient_shards.py` and the obsolete patient-sharding helper package were removed from the active codebase
- repository docs now describe Stage 2 patient shards plus `master_manifest.sqlite` as the active storage contract and mark Stage 6.5 / 7.1 as obsolete
- remaining shard-backed runtime loaders in Stages 7.2-11 are explicitly treated as transitional code to be migrated in later milestones, rather than as an active Stage 6.5 dependency in the pipeline architecture

## Milestone 8 --DONE

Convert Stage 7.2 smart sampling to SQLite row-state updates.

Scope:
- replace filtered TRAIN shard materialization with sampling decision columns in SQLite
- keep patient-wise label-aware sampling behavior unchanged
- emit only summary and lineage sidecars
- remove the transitional Stage 7.2 dependence on `TRAIN_shards` by resolving TRAIN rows from SQLite and canonical Stage 2 shard paths

Exit criteria:
- Stage 7.2 no longer rewrites filtered TRAIN shard HDF5 files
- selected training rows are resolved directly from SQLite

Implemented by:
- `helpers/smart_sampling/config.py` now requires `SMART_SAMPLER_MASTER_MANIFEST_PATH`; Stage 7.2 no longer takes `TRAIN_shards` / manifest inputs as its active source contract
- `helpers/smart_sampling/index.py` now builds the Stage 7 TRAIN patient index directly from `master_manifest.sqlite`, validating that each patient resolves to one canonical Stage 2 shard and carrying canonical `source_row_index` metadata
- `helpers/smart_sampling/pipeline.py` now runs selection directly against canonical Stage 2 patient shards, writes only sidecar artifacts, records a Stage 7.2 run in SQLite, and persists per-row `sampling_decision` plus `is_stage7_selected` back into `patch_stage_state`
- `7_2_smart_sampler.py` now reports the sidecar output directory instead of a filtered shard directory
- Stage 7.2 regression tests now cover SQLite-backed config loading, sidecar-only smart-sampling outputs, SQLite selection-state updates, local input staging, and the GIST selector path

## Milestone 9 --DONE

Build the shared SQLite-backed loader foundation.

Scope:
- create shared stage-specific SQLite query helpers
- create the shared canonical-row dataset abstraction under `helpers/training/`
- support worker-safe HDF5 handle reuse
- enforce startup-only SQLite access with in-memory row caching

Exit criteria:
- runtime loaders query SQLite once at startup only
- runtime hot path uses only in-memory metadata plus HDF5 reads

Implemented by:
- `helpers/training/master_manifest_queries.py` now provides shared startup-only SQLite query helpers for Stage 8/9 TRAIN rows, Stage 8-10 VALIDATION rows, and Stage 11 TEST rows, all resolved by canonical `(source_hdf5_path, source_row_index)` identity
- `helpers/training/canonical_dataset.py` now provides a shared canonical-row HDF5 dataset abstraction that holds in-memory row records, lazily opens canonical Stage 2 shards, reuses HDF5 handles per worker/process, and never queries SQLite during `__getitem__`
- the shared dataset supports the current downstream mask shapes needed by later migrations through `raw`, `binary`, and `two_channel` mask modes, while preserving patient-id and filename accessors for existing stage semantics
- `tests/test_training_canonical_dataset.py` now covers split-aware startup queries, no-SQLite-after-startup runtime access, worker-safe handle reopening, and canonical-row mask loading behavior

## Milestone 10 --DONE

Implement shared on-the-fly stain normalization.

Scope:
- use `torch-staintools` as the main backend
- support `none`, `reinhard`, `ruifrok`, `macenko`, and `vahadane`
- apply normalization before augmentations and before ImageNet normalization
- keep Stage 5 as the owner of TRAIN-fitted normalization state

Exit criteria:
- a shared normalization abstraction is available to all model-facing stages
- no migrated path depends on TIAToolbox

Implemented by:
- `helpers/training/stain_normalization.py` now provides the shared runtime stain-normalization abstraction, including startup-only SQLite artifact loading, fail-closed normalization-state hash validation, `torch-staintools`-backed `reinhard` / `macenko` / `vahadane`, and `none` bypass handling
- `helpers/training/canonical_dataset.py` now accepts an optional shared image normalizer and applies it to canonical Stage 2 RGB patches before Albumentations-style transforms run
- `tests/test_training_stain_normalization.py` now covers startup artifact loading, state-hash validation, split-metadata consistency checks, and `reinhard` runtime state restoration without any TIAToolbox dependency in the migrated runtime path
- `tests/test_training_canonical_dataset.py` now verifies that canonical datasets invoke the shared stain normalizer before augmentation

## Milestone 11 --DONE

Implement the GPU Ruifrok path.

Scope:
- implement `ruifrok` as fixed-matrix stain deconvolution/reconvolution on GPU
- keep it aligned with the Macenko/Vahadane-style formulation using a fixed matrix
- integrate it into the shared normalization abstraction

Exit criteria:
- `ruifrok` is available as an on-the-fly normalization option
- no dedicated external Ruifrok class is required

Implemented by:
- `helpers/training/stain_normalization.py` now routes `ruifrok` through the same shared stain-separation-style runtime path as the other model-facing normalizers, using fixed-matrix deconvolution/reconvolution on torch tensors and supporting an optional persisted `stain_matrix_source` override while defaulting to the canonical H&E Ruifrok matrix
- the shared Ruifrok runtime path is device-aware, so its fixed source matrix, target stain matrix, and target concentration statistics all live on the requested torch device for on-the-fly execution
- `tests/test_training_stain_normalization.py` now covers both baseline Ruifrok runtime support and persisted fixed-source-matrix restoration on the requested device

## Milestone 12 --DONE

Migrate Stage 8 LR finder to the new loader path.

Scope:
- make Stage 8 resolve rows from SQLite at startup
- make Stage 8 load pixels from Stage 2 patient shards
- make Stage 8 use shared on-the-fly normalization
- remove Stage 8 dependence on `TRAIN_shards`, `TRAIN_FILTERED_shards`, and `VALIDATION_shards`

Exit criteria:
- Stage 8 runs without split-specific HDF5 files

Implemented by:
- `helpers/lr_finder/data.py` now resolves Stage 8 TRAIN and VALIDATION rows from `master_manifest.sqlite` at startup, builds the runtime dataset on top of `helpers/training/canonical_dataset.py`, stages canonical Stage 2 shards locally when requested, and records manifest-backed split provenance instead of shard-manifest provenance
- `helpers/lr_finder/data.py` now applies the shared on-the-fly stain normalizer through the canonical dataset path before augmentation while preserving Stage 8 subset selection and weighted-sampling behavior
- `helpers/lr_finder/config.py`, `helpers/lr_finder/pipeline.py`, `README.md`, and `.env_example` now use the native `LR_FINDER_MASTER_MANIFEST_PATH` contract instead of `LR_FINDER_HDF5_DRIVE_DIR`
- `tests/test_lr_finder_data.py` now covers SQLite-backed smart-sampling row resolution, canonical Stage 2 pixel loading, shared stain-normalizer integration, subset behavior, and weighted train-loader construction
- `tests/test_lr_finder_config.py`, `tests/test_lr_finder_pipeline.py`, and `tests/test_lr_finder_runner.py` were updated to the new Stage 8 manifest-backed config contract

## Milestone 13 --DONE

Migrate Stage 9 training to the new loader path.

Scope:
- make Stage 9 resolve rows from SQLite at startup
- preserve current patient checks, weighting logic, and training semantics
- make Stage 9 use shared on-the-fly normalization and HDF5-backed reads
- remove Stage 9 dependence on `TRAIN_shards`, `TRAIN_FILTERED_shards`, and `VALIDATION_shards`

Exit criteria:
- Stage 9 runs without split-specific or filtered shard HDF5 files

Implemented by:
- `helpers/training/data.py` now resolves Stage 9 TRAIN and VALIDATION rows from `master_manifest.sqlite` at startup, rebuilds the runtime datasets on top of `helpers/training/canonical_dataset.py`, preserves patient-separation checks plus weighted sampling semantics, and records manifest-backed split provenance instead of shard-manifest provenance
- `helpers/training/data.py` now applies the shared split stain normalizer through the canonical dataset path before augmentation while preserving optional Stage 9 subset selection and artifact-aware loss covariates
- `helpers/training/config.py`, `9_training_ensemble.py`, `README.md`, and `.env_example` now use the native `TRAINING_MASTER_MANIFEST_PATH` contract instead of `TRAINING_HDF5_DRIVE_DIR`
- `tests/test_training_config.py` and `tests/test_training_data.py` now cover the manifest-backed Stage 9 config contract, smart-sampling row resolution, canonical Stage 2 pixel loading, shared stain-normalizer integration, artifact-aware loss covariates, subset behavior, patient isolation, and manifest-backed provenance

## Milestone 14 --DONE

Migrate Stage 10 optimizer to the new loader path.

Scope:
- make Stage 10 resolve rows from SQLite at startup
- make Stage 10 use shared HDF5-backed reads and on-the-fly normalization
- remove Stage 10 dependence on `VALIDATION_shards`

Exit criteria:
- Stage 10 runs without downstream-materialized HDF5 dependencies

Implemented by:
- `helpers/ensemble_optimizer/data.py` now resolves Stage 10 VALIDATION rows from `master_manifest.sqlite` at startup, builds the optimizer dataset on top of `helpers/training/canonical_dataset.py`, preserves patient-id filtering plus two-channel mask semantics, and records manifest-backed validation provenance instead of `VALIDATION_shards` manifest provenance
- `helpers/ensemble_optimizer/data.py` now applies the shared split stain normalizer through the canonical dataset path before augmentation while preserving worker-safe Stage 2 shard access and optional local shard staging
- `helpers/ensemble_optimizer/config.py`, `helpers/ensemble_optimizer/pipeline.py`, `README.md`, and `.env_example` now use the native `ENSEMBLE_OPT_MASTER_MANIFEST_PATH` contract instead of `ENSEMBLE_OPT_HDF5_DRIVE_DIR`
- `tests/test_ensemble_optimizer_config.py`, `tests/test_ensemble_optimizer_data.py`, `tests/test_ensemble_optimizer_pipeline.py`, and `tests/test_ensemble_optimizer_optimization.py` were updated for the manifest-backed Stage 10 config and runtime path

## Milestone 15 --DONE

Migrate Stage 11 inference to the new loader path.

Scope:
- make Stage 11 resolve rows from SQLite at startup
- make Stage 11 use shared HDF5-backed reads and on-the-fly normalization
- remove Stage 11 dependence on `TEST_shards`

Exit criteria:
- Stage 11 runs without downstream-materialized HDF5 dependencies

Implemented by:
- `helpers/ensemble_inference/data.py` now resolves Stage 11 TEST rows from `master_manifest.sqlite` at startup, builds the inference dataset on top of `helpers/training/canonical_dataset.py`, preserves patient-id and filename reporting, and records manifest-backed TEST provenance instead of `TEST_shards` manifest provenance
- `helpers/ensemble_inference/data.py` now applies the shared split stain normalizer through the canonical dataset path before augmentation while preserving worker-safe Stage 2 shard access and optional local shard staging
- `helpers/ensemble_inference/config.py`, `helpers/ensemble_inference/pipeline.py`, `README.md`, and `.env_example` now use the native `ENSEMBLE_INFER_MASTER_MANIFEST_PATH` contract instead of `ENSEMBLE_INFER_HDF5_DRIVE_DIR`
- `helpers/ensemble_optimizer/data.py` and `helpers/ensemble_optimizer/reporting.py` now emit manifest-backed validation lineage metadata so Stage 10 recipes and Stage 11 inference compare the shared manifest/runtime-normalization contract instead of legacy shard attributes
- `tests/test_ensemble_inference_config.py`, `tests/test_ensemble_inference_data.py`, `tests/test_ensemble_inference_pipeline.py`, and `tests/test_ensemble_optimizer_reporting.py` were updated for the manifest-backed Stage 11 config and runtime path

## Milestone 16 --DONE

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

Implemented by:
- `tests/test_training_canonical_dataset.py` now verifies that startup-only runtime split queries preserve patient isolation across TRAIN/VALIDATION/TEST and that every manifest-backed runtime record still resolves to the exact Stage 2 `label`, `patient_id`, and `filename` at its canonical `(source_hdf5_path, source_row_index)` identity
- `tests/test_training_data.py` and `tests/test_training_canonical_dataset.py` already cover SQLite-backed artifact coverage lookup plus artifact-aware runtime covariates, and Milestone 16 keeps that path as the artifact-metadata equivalence regression anchor
- `tests/test_smart_sampling_pipeline.py` now verifies that Stage 7.2 published sidecars and SQLite-selected runtime TRAIN rows agree on the exact canonical identities selected for downstream smart-sampled loading
- `tests/test_training_stain_normalization.py` now covers runtime support for all shared normalization families used by migrated stages, including explicit `MACENKO` and `VAHADANE` state restoration in addition to existing `NOT_NORMALIZED`, `REINHARD`, and `RUIFROK` coverage
- `tests/test_ensemble_inference_pipeline.py` now adds another fail-closed Stage 11 provenance regression by rejecting recipes whose `validation_lineage` omits required manifest-backed runtime-normalization keys

## Milestone 17 --DONE

Remove obsolete dependencies and cleanup leftover rewrite paths.

Scope:
- remove TIAToolbox from migrated paths
- remove dead code tied to Stage 3 and Stage 6.5 / 7.1 rewrite workflows
- remove leftover downstream-materialization assumptions from configs and docs
- delete `helpers/patient_shards/*`, obsolete Stage 6.5 / 7.1 tests, and remaining shard-layout-only references once Stages 7.2-11 no longer require them

Exit criteria:
- the active pipeline matches `PLAN_MIGRATION.md`
- obsolete rewrite-only code paths are gone

Implemented by:
- `helpers/patient_shards/*` and the obsolete Stage 6.5 / 7.1 test modules were deleted now that no active runtime stage depends on the removed patient-sharding rewrite workflow
- the obsolete `7_1_patient_shards.py` shim was deleted now that the migrated pipeline no longer needs a fail-closed compatibility entrypoint
- `README.md` now describes the active metadata-first architecture: Stage 3 and Stage 6.5 / 7.1 are obsolete, Stage 7.2 writes SQLite state plus sidecars instead of filtered shards, and Stages 8-11 resolve runtime rows from `master_manifest.sqlite` while reading pixels from canonical Stage 2 patient shards
- `.env_example` no longer exposes removed `PATIENT_SHARDS_*` or pre-migration Stage 7 shard-input variables and now documents the active `SMART_SAMPLER_MASTER_MANIFEST_PATH` contract
- README dependency documentation now references `torch-staintools` instead of TIAToolbox for runtime stain normalization
