# Stage 2 Scientific Validation Review

## Summary of Scientific Objective

This review evaluated `2_database_manager.py` and its related extraction helpers for scientific flaws that could invalidate downstream results. The primary scientific concern is whether Stage 2 produces reproducible, leakage-safe patch datasets and provenance suitable for later patient-level splitting, model selection, and evaluation.

## Detected Threats to Validity

### Critical: Synthetic patient IDs can break true patient-level isolation

- `helpers/extraction/repository.py:75-86` defines `PATIENT` as `TEXT NOT NULL UNIQUE`.
- `helpers/extraction/repository.py:163-167` seeds new patient identifiers as sequential synthetic values.
- `helpers/extraction/repository.py:221-231` assigns one new patient ID per image during ingestion.
- `helpers/crossfold/discovery.py:32-35,48-57,62-66` loads those stored `patient_ids` from HDF5.
- `helpers/crossfold/splitting.py:15-23` performs downstream grouping and stratification by `patient_id`.

Impact:

- If one biological patient contributes multiple slides, Stage 2 currently treats those slides as different patients.
- Downstream Stage 5 patient-level splits can therefore place slides from the same real patient into different splits while reporting that patient-level isolation was preserved.
- This is a direct data leakage risk and can inflate reported performance.

Severity: Critical

### Major: `hiseg_xml_coord_level` is missing from the persisted processing signature

- `helpers/extraction/artifact_lookup.py:74-109` builds the Stage 2 `processing_signature`.
- That signature includes image, annotation, GeoJSON, and several extraction settings, but not `hiseg_xml_coord_level`.
- `helpers/extraction/data_handlers.py:138-142,176-198` uses `hiseg_xml_coord_level` to rescale HISEG XML annotations.
- `2_database_manager.py:223-246` relies on `processing_signature` to decide whether completed outputs are stale.

Impact:

- Changing `HISEG_XML_COORD_LEVEL` changes annotation geometry and therefore masks and labels.
- Because that setting is not included in the signature, previously completed slides can be incorrectly reused after a scientifically meaningful parsing change.
- This threatens reproducibility and can silently mix stale labels into later experiments.

Severity: Major

### Major: contradictory defaults for `USE_ADVANCED_ARTIFACT_FILTERING`

- `helpers/extraction/config.py:94-97` defaults `USE_ADVANCED_ARTIFACT_FILTERING` to `True`.
- `helpers/extraction/image_reader_service.py:100-107` defaults the same setting to `False`.
- `2_database_manager.py:96-106` resolves GeoJSONs based on the config-side interpretation.
- `2_database_manager.py:156-163` and `helpers/extraction/image_reader_service.py:183-187` pass the runtime-side interpretation into extraction.

Impact:

- If the env var is unset, Stage 2 orchestration can behave as if artifact filtering is enabled while extraction behaves as if it is disabled.
- That can silently change whether artifact coverage metadata is actually computed.
- Downstream artifact-aware analyses or ablations may therefore be based on incorrect metadata generation assumptions.

Severity: Major

### Moderate: HDF5 shard filenames can collide across same-stem slides

- `2_database_manager.py:145` writes each shard to `PATCHES/HDF5_SHARDS/<image_stem>.h5`.
- `helpers/extraction/image_reader_service.py:22-27` supports multiple input image extensions.

Impact:

- Two supported files sharing the same stem but different extensions can target the same shard path.
- One slide can overwrite another, silently dropping data and compromising dataset completeness and reproducibility.

Severity: Moderate

## Recommended Corrections

1. Replace synthetic per-slide patient IDs with a true patient identifier derived from source metadata or naming rules.
2. Add `hiseg_xml_coord_level` to `build_processing_signature(...)` and mark completed slides stale when it changes.
3. Unify `USE_ADVANCED_ARTIFACT_FILTERING` defaults across Stage 2 config and runtime loading.
4. Make Stage 2 shard filenames collision-safe, for example by including extension or a stable hash.

## Reproducibility Improvements

1. Add a regression test proving that two slides from the same real patient retain the same stored patient ID.
2. Add a stale-lineage test showing that changing `hiseg_xml_coord_level` invalidates prior completed outputs.
3. Add a config-consistency test for shared Stage 2 env vars such as `USE_ADVANCED_ARTIFACT_FILTERING`.
4. Add a shard-collision test for same-stem different-extension inputs.

## Bottom Line

I would not treat Stage 2-derived downstream metrics as scientifically reliable until at least these are addressed:

1. true patient-level identity preservation
2. processing-signature coverage for HISEG coordinate-level parsing

Those two issues are sufficient to invalidate claims about leakage-safe evaluation and strict reproducibility.

## Addendum: Stage 3 and Stage 5 Scientific Validation Review

### Summary of Scientific Objective

This addendum evaluated `3_pack_splits_to_hdf5.py` and `5_crossfold.py` together with the helper modules they orchestrate. The scientific question is whether Stage 3 preserves the post-cleaning patch cohort faithfully and whether Stage 5 creates leakage-safe, evaluation-fair, reproducible TRAIN/VALIDATION/TEST splits suitable for defensible downstream model claims.

`3_pack_splits_to_hdf5.py` and `5_crossfold.py` themselves are thin wrappers. The substantive scientific behavior lives in `helpers/packaging/*` and `helpers/crossfold/*`.

### Detected Threats to Validity

#### Major: Stage 3 accepted-manifest filtering is bound to mutable row indices, not immutable source identity

- `helpers/graph/cleaning_pipeline.py:229-260` writes accepted/rejected manifests with `source_hdf5_path` and `source_row_index`, but no source-HDF5 hash or row-level identity check.
- `helpers/packaging/writer.py:320-348` loads accepted rows and only verifies that `source_hdf5_path` string matches the current input path.
- `helpers/packaging/writer.py:358-375` builds the output signature from the current source-file hash plus manifest hash plus selected row indices, but does not verify that the manifest rows still refer to the same patch identities in the current source file.
- `helpers/packaging/writer.py:436-494` then copies rows strictly by current `source_row_index`.

Impact:

- If the Stage 3 source HDF5 is regenerated or reordered in place at the same path, an older accepted manifest can silently select the wrong rows.
- That means Stage 4.3 cleaning decisions can be applied to different patches than the ones that were actually reviewed.
- The result can silently reintroduce rejected/contaminated patches or drop accepted ones, breaking reproducibility and potentially biasing downstream training and evaluation.

Severity: Major

#### Major: Stage 5 split search cherry-picks one split from many candidates using a full-cohort entropy objective

- `helpers/crossfold/config.py:101-109` sets the entropy-based objective fields, and `helpers/crossfold/config.py:217-233` enables that objective by default.
- `helpers/crossfold/pipeline.py:54-73` computes entropy on the full source dataset before split search and passes patient-level entropy into the splitter.
- `helpers/crossfold/splitting.py:220-320` samples up to `max_tries` randomized patient splits and keeps the single best one by objective score.
- The default objective is `score_split="TRAIN"` with `maximize=True`, so the search preferentially allocates higher-entropy patients into TRAIN.

Impact:

- This is not a neutral random split. It is a data-dependent split selection procedure that searches many feasible partitions and retains the one with the most favorable entropy profile.
- Because the default objective maximizes TRAIN difficulty, the held-out VALIDATION/TEST sets can become systematically easier on the same full cohort budget.
- Reported downstream metrics can therefore be inflated relative to a neutral patient-level split, especially on the currently documented small-cohort regime.
- A reviewer could reasonably classify this as split cherry-picking unless the manuscript explicitly justifies the objective and reports results against fixed neutral baselines.

Severity: Major

#### Moderate: Stage 5 does not implement cross-validation despite the crossfold framing, and it emits no uncertainty estimate

- `helpers/crossfold/pipeline.py:67-140` creates one TRAIN/VALIDATION/TEST split and writes one run directory.
- `helpers/crossfold/splitting.py:147-359` searches for a single feasible split and returns only that split.

Impact:

- If a manuscript describes this stage as cross-validation or implies fold-averaged robustness, the code does not support that claim.
- On the documented small-patient setting, one selected split without confidence intervals or repeated resampling gives a high-variance estimate of performance.
- This weakens statistical defensibility even when the code is otherwise correct.

Severity: Moderate

#### Moderate: Stage 5 run provenance does not fully persist the split-selection objective configuration

- `helpers/crossfold/pipeline.py:43-48` logs the objective configuration at runtime.
- `helpers/crossfold/provenance.py:184-206` writes `run_config.json`, but it stores constraints, random seed, selected patients, and objective score without persisting the full objective configuration used to search splits.
- Missing persisted fields include whether objective search was enabled and the settings that change the search behavior, such as optimization direction and entropy-generation parameters.

Impact:

- Two runs can produce scientifically different split-selection procedures while leaving an incomplete audit trail in the saved artifacts.
- That makes independent reproduction and peer-review reconstruction harder, particularly because the split itself is chosen by optimization rather than simple random partitioning.

Severity: Moderate

### Recommended Corrections

1. Make Stage 3 accepted-manifest application fail closed unless the manifest records and matches an immutable source-HDF5 identity, ideally including the source HDF5 SHA256 and row-level metadata checks for filename/patient/slide.
2. Disable the Stage 5 entropy objective by default for reported benchmark datasets, or justify it explicitly and compare against a neutral fixed split strategy.
3. If entropy-guided split design is kept, pre-register the objective, persist the full objective config in `run_config.json`, and report results against multiple seeds or folds rather than a single selected split.
4. Avoid describing Stage 5 as cross-validation unless true multi-fold evaluation is implemented and reported.

### Reproducibility Improvements

1. Add a regression test proving that a stale accepted manifest from an older source HDF5 is rejected rather than silently reused by row index.
2. Add a provenance test requiring Stage 5 `run_config.json` to persist the entire objective configuration used during split search.
3. Add a scientific regression benchmark that compares downstream split difficulty and class/patient balance with objective-enabled versus neutral splitting.
4. If publication claims rely on robust generalization, add repeated-seed or fold-level reporting with confidence intervals.

### Bottom Line

The most serious new concerns are:

1. Stage 3 can silently misapply Stage 4.3 acceptance decisions if the source HDF5 is regenerated in place.
2. Stage 5 currently performs data-dependent split optimization by default, which can bias held-out evaluation difficulty and invite reviewer concerns about split cherry-picking.

I would not present Stage 5-held-out metrics as fully publication-safe until those two points are addressed or very explicitly justified in the experimental methods.
