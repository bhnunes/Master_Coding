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
