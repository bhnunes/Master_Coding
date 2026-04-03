# Stage 2 Scientific Validation Review

## Summary of Scientific Objective

This review evaluated `2_database_manager.py` and its related extraction helpers for scientific flaws that could invalidate downstream results. The primary scientific concern is whether Stage 2 produces reproducible, leakage-safe patch datasets and provenance suitable for later patient-level splitting, model selection, and evaluation.

## Detected Threats to Validity

### Critical: True patient-level isolation still depends on an external dataset assumption

- `helpers/extraction/repository.py` still assigns one persisted patient identifier per ingested slide rather than deriving a verified biological patient identifier from source metadata.
- `helpers/crossfold/discovery.py` and `helpers/crossfold/splitting.py` therefore preserve isolation only with respect to the Stage 2 stored patient IDs.

Impact:

- If one biological patient contributes multiple slides, downstream Stage 5 patient-level splits can still place those slides into different partitions while appearing leakage-safe.
- This remains a direct data leakage risk and can inflate reported performance.
- In the current project state, this risk is being accepted based on the external operational assumption that each WSI corresponds to a unique patient.
- That assumption is not enforced or validated in code.

Severity: Critical

### Resolved: `hiseg_xml_coord_level` is now included in the persisted processing signature

- `helpers/extraction/artifact_lookup.py` now includes `hiseg_xml_coord_level` in the Stage 2 `processing_signature`.
- `2_database_manager.py` now passes `config.hiseg_xml_coord_level` into that signature builder.

Impact:

- Changing `HISEG_XML_COORD_LEVEL` now invalidates the persisted processing signature as it should.
- This mitigates the stale-output reuse risk for scientifically meaningful HISEG XML parsing changes.

Severity: Resolved

### Resolved: `USE_ADVANCED_ARTIFACT_FILTERING` defaults are now aligned

- `helpers/extraction/config.py` defaults `USE_ADVANCED_ARTIFACT_FILTERING` to `True`.
- `helpers/extraction/image_reader_service.py` now also defaults the same setting to `True`.
- `2_database_manager.py` therefore now resolves GeoJSONs, persists signatures, and invokes extraction with a consistent interpretation when the env var is unset.

Impact:

- This removes the previous split-brain behavior between orchestration and runtime extraction.
- Downstream artifact-aware analyses and ablations now have a consistent default Stage 2 behavior when the env var is omitted.

Severity: Resolved

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

#### Resolved: Stage 3 accepted-manifest filtering now binds rows to immutable source identity

- `helpers/graph/cleaning_pipeline.py` now writes `source_hdf5_sha256` into accepted/rejected manifests.
- `helpers/packaging/writer.py` now requires each accepted-manifest row to match both `source_hdf5_path` and `source_hdf5_sha256`.
- `helpers/packaging/writer.py` now validates row-level identity against the current source HDF5 using `filename`, `patient_id`, and `slide_id` before copying rows.

Impact:

- If the source HDF5 is regenerated, reordered, or row identities drift, Stage 3 now fails closed instead of silently applying stale cleaning decisions to the wrong rows.
- This preserves the earlier Stage 3 performance gains because validation happens once up front and does not change the batched image/mask copy path.

Severity: Resolved

#### Resolved for TEST: Stage 5 entropy-guided split search is now confined to TRAIN / VALIDATION

- `helpers/crossfold/splitting.py` now freezes `TEST` first with a neutral patient-level stratified split.
- `helpers/crossfold/splitting.py` applies entropy-guided optimization only when allocating the remaining TRAIN / VALIDATION patients.
- `helpers/crossfold/pipeline.py` persists the split-selection method in provenance so the procedure is auditable.

Impact:

- This removes the main scientific objection for held-out benchmarking because `TEST` is no longer influenced by entropy-guided search.
- Remaining caveat: TRAIN / VALIDATION allocation is still a development-time heuristic when enabled, so methods/reporting should state that clearly.

Severity: Mitigated for TEST; still requires transparent reporting for TRAIN / VALIDATION

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

1. Stage 2 synthetic patient IDs can still break true patient-level isolation upstream of Stage 5.


I would now treat the previous Stage 5 split-cherry-picking concern, the Stage 3 accepted-manifest provenance concern, and the HISEG coordinate-level signature concern as substantially mitigated. The strongest remaining publication risk is still upstream patient identity correctness in Stage 2.

### Accepted Assumption

For now, the pipeline is proceeding under the dataset-provider assumption that each WSI corresponds to a unique patient and that no patient contributes multiple slides. Because WSI filenames do not provide a reliable patient-key pattern and no external patient-mapping manifest is available, this assumption is not currently enforced in code.

Implication:

- If that source-data assumption is false, patient-level isolation claims for downstream Stage 5 evaluation may still be scientifically invalid despite the current code safeguards.
